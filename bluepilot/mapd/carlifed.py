"""
carlifed — receive CarLife Companion dual-channel map stream.

Channel A (UDP :8888): MapDataPacket JSON → cereal carLifeMapData (HUD)
Channel B (UDP :8889): CLVF JPEG fragments → decode → shared-memory RGBA

Display only; never feeds control. No Amap tile fetch / rotation / crop.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from io import BytesIO
from typing import Any

import cereal.messaging as messaging
import numpy as np
from PIL import Image

from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from bluepilot.mapd.carlife_ipc import CarLifeFrameShm, MAX_H, MAX_W, STATUS_PATH

DATA_PORT = 8888
VIDEO_PORT = 8889
DATA_RECV_BUF = 4096   # JSON payload ≤2048; room for oversized datagrams
VIDEO_RECV_BUF = 2048  # CLVF packets ≤1400
HDR_FMT = ">4sBBHIQHHHHH"
HDR_SIZE = struct.calcsize(HDR_FMT)  # 30
assert HDR_SIZE == 30

PENDING_MAX_FRAMES = 8
FRAME_STALE_S = 1.0
STATE_HOLD_S = 3.0
PUBLISH_HZ = 20.0
NAV_PATH_MAX = 24

VALID_DIRS = frozenset({"left", "straight", "right", "uturn", "unknown"})
VALID_LIGHTS = frozenset({"red", "green", "yellow", "unknown"})
VALID_TURNS = frozenset({"left", "right", "straight", "unknown"})
VALID_LANE_CHANGE = frozenset({"left", "right", "none", "unknown"})
VALID_SPEED_ACTION = frozenset({"decelerate", "start", "maintain", "unknown"})


def _lower_priority() -> None:
  try:
    os.nice(10)
  except Exception:
    pass
  try:
    with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
      f.write("500")
  except Exception:
    pass
  try:
    os.system(f"ionice -c 2 -n 7 -p {os.getpid()} >/dev/null 2>&1")
  except Exception:
    pass


def _norm_enum(v: Any, valid: frozenset[str], default: str = "unknown") -> str:
  s = str(v) if v is not None else default
  return s if s in valid else default


def _norm_dir(v: Any) -> str:
  return _norm_enum(v, VALID_DIRS)


def _norm_light(v: Any) -> str:
  return _norm_enum(v, VALID_LIGHTS)


def _as_int(v: Any, default: int = -1) -> int:
  try:
    return int(v)
  except (TypeError, ValueError):
    return default


def _as_float(v: Any, default: float = 0.0) -> float:
  try:
    return float(v)
  except (TypeError, ValueError):
    return default


def _as_float_or_none(v: Any) -> float | None:
  if v is None:
    return None
  try:
    return float(v)
  except (TypeError, ValueError):
    return None


def _clamp01(v: float) -> float:
  return max(0.0, min(1.0, v))


def _parse_point(raw: Any) -> dict[str, float] | None:
  if not isinstance(raw, dict):
    return None
  try:
    x = _clamp01(float(raw.get("x")))
    y = _clamp01(float(raw.get("y")))
  except (TypeError, ValueError):
    return None
  return {"x": x, "y": y}


def _parse_lanes(raw_lanes: Any) -> list[dict[str, Any]]:
  if not isinstance(raw_lanes, list):
    return []
  out = []
  for item in raw_lanes:
    if not isinstance(item, dict):
      continue
    out.append({
      "index": _as_int(item.get("index"), 0),
      "directions": [_norm_dir(x) for x in (item.get("directions") or [])],
      "highlighted": bool(item.get("highlighted", False)),
    })
  out.sort(key=lambda x: x["index"])
  return out


def _parse_recommended(raw: Any, lanes: list[dict[str, Any]]) -> list[int]:
  ids: list[int] = []
  if isinstance(raw, list):
    for v in raw:
      i = _as_int(v, -1)
      if i >= 0:
        ids.append(i)
  if not ids:
    ids = [lane["index"] for lane in lanes if lane.get("highlighted")]
  return ids


def _parse_nav_path(raw: Any) -> list[dict[str, float]]:
  if not isinstance(raw, list):
    return []
  pts = []
  for item in raw[:NAV_PATH_MAX]:
    p = _parse_point(item)
    if p is not None:
      pts.append(p)
  return pts


class StateHold:
  """Keep last valid HUD fields during occlusion / unknown flashes."""

  _HOLD_KEYS = (
    "lightStatus", "countdown", "speedLimit", "laneCount", "currentLane",
    "lanes", "laneDirection", "recommendedLanes", "turnDirection", "laneChange",
    "speedAction", "intersectionDistance", "egoCar", "navPath",
  )

  def __init__(self, hold_s: float = STATE_HOLD_S):
    self.hold_s = hold_s
    self._lock = threading.Lock()
    self._pkt: dict[str, Any] | None = None
    self._held: dict[str, Any] | None = None
    self._held_until = 0.0

  def update(self, raw: dict[str, Any]) -> dict[str, Any]:
    now = time.monotonic()
    light = _norm_light(raw.get("lightStatus"))
    occluded = bool(raw.get("isOccluded", False))
    speed = _as_int(raw.get("speedLimit", -1))
    countdown = _as_int(raw.get("countdown", -1))
    lane_count = _as_int(raw.get("laneCount", -1))
    current_lane = _as_int(raw.get("currentLane", -1))
    lanes = _parse_lanes(raw.get("lanes"))
    lane_direction = _norm_dir(raw.get("laneDirection"))
    turn = _norm_enum(raw.get("turnDirection"), VALID_TURNS)
    lane_change = _norm_enum(raw.get("laneChange"), VALID_LANE_CHANGE, "none")
    speed_action = _norm_enum(raw.get("speedAction"), VALID_SPEED_ACTION)
    recommended = _parse_recommended(raw.get("recommendedLanes"), lanes)
    dist = _as_float_or_none(raw.get("intersectionDistance"))
    ego = _parse_point(raw.get("egoCar"))
    nav_path = _parse_nav_path(raw.get("navPath"))

    with self._lock:
      fresh: dict[str, Any] = {}
      if light != "unknown":
        fresh["lightStatus"] = light
      if countdown >= 0:
        fresh["countdown"] = countdown
      if speed > 0:
        fresh["speedLimit"] = speed
      if lane_count >= 1:
        fresh["laneCount"] = lane_count
      if current_lane >= 0:
        fresh["currentLane"] = current_lane
      if lanes:
        fresh["lanes"] = lanes
      if recommended:
        fresh["recommendedLanes"] = recommended
      if lane_direction != "unknown":
        fresh["laneDirection"] = lane_direction
      if turn != "unknown":
        fresh["turnDirection"] = turn
      if lane_change not in ("none", "unknown"):
        fresh["laneChange"] = lane_change
      if speed_action != "unknown":
        fresh["speedAction"] = speed_action
      if dist is not None:
        fresh["intersectionDistance"] = dist
      if ego is not None:
        fresh["egoCar"] = ego
      if nav_path:
        fresh["navPath"] = nav_path

      if fresh and not occluded:
        base = dict(self._held or {})
        base.update(fresh)
        self._held = base
        self._held_until = now + self.hold_s

      out: dict[str, Any] = {
        "schemaVersion": _as_int(raw.get("schemaVersion"), 1),
        "timestamp": _as_int(raw.get("timestamp"), 0),
        "lightStatus": light,
        "countdown": countdown,
        "speedLimit": speed,
        "laneCount": lane_count,
        "currentLane": current_lane,
        "lanes": lanes,
        "recommendedLanes": recommended,
        "laneDirection": lane_direction,
        "turnDirection": turn,
        "laneChange": lane_change,
        "speedAction": speed_action,
        "isOccluded": occluded,
        "confidence": _as_float(raw.get("confidence")),
        "lightConfidence": _as_float(raw.get("lightConfidence")),
        "speedConfidence": _as_float(raw.get("speedConfidence")),
        "laneConfidence": _as_float(raw.get("laneConfidence")),
        "actionConfidence": _as_float(raw.get("actionConfidence")),
        "pathConfidence": _as_float(raw.get("pathConfidence")),
        "intersectionDistance": dist,
        "curveCurvature": _as_float_or_none(raw.get("curveCurvature")),
        "egoCar": ego,
        "navPath": nav_path,
      }

      need_hold = occluded or light == "unknown" or speed <= 0 or not lanes or turn == "unknown"
      if self._held and now <= self._held_until and need_hold:
        held = self._held
        for key in self._HOLD_KEYS:
          cur = out.get(key)
          hv = held.get(key)
          if hv is None:
            continue
          if key == "lightStatus" and (cur == "unknown" or occluded):
            out[key] = hv
          elif key == "countdown" and ((cur is not None and cur < 0) or occluded):
            out[key] = hv
          elif key == "speedLimit" and ((cur is not None and cur <= 0) or occluded):
            out[key] = hv
          elif key == "laneCount" and ((cur is not None and cur < 1) or occluded):
            out[key] = hv
          elif key == "currentLane" and ((cur is not None and cur < 0) or occluded):
            out[key] = hv
          elif key in ("lanes", "recommendedLanes", "navPath") and (not cur or occluded):
            out[key] = hv
          elif key in ("laneDirection", "turnDirection", "speedAction") and (cur == "unknown" or occluded):
            out[key] = hv
          elif key == "laneChange" and (cur in ("none", "unknown") or occluded):
            out[key] = hv
          elif key == "intersectionDistance" and (cur is None or occluded):
            out[key] = hv
          elif key == "egoCar" and (cur is None or occluded):
            out[key] = hv

      self._pkt = out
      return out

  def latest(self) -> dict[str, Any] | None:
    with self._lock:
      return dict(self._pkt) if self._pkt else None


class VideoAssembler:
  def __init__(self):
    self._lock = threading.Lock()
    self._pending: dict[int, dict[str, Any]] = {}
    self._latest_jpeg: bytes | None = None
    self._latest_meta: tuple[int, int, int] | None = None  # ts, w, h
    self._last_complete_seq = -1

  def ingest(self, packet: bytes) -> tuple[bytes, int, int, int] | None:
    if len(packet) < HDR_SIZE:
      return None
    try:
      magic, ver, codec, _flags, seq, ts, w, h, fi, fc, plen = struct.unpack(
        HDR_FMT, packet[:HDR_SIZE]
      )
    except struct.error:
      return None
    if magic != b"CLVF" or ver != 1 or codec != 1:
      return None
    if fc <= 0 or fi >= fc or plen <= 0:
      return None
    payload = packet[HDR_SIZE:HDR_SIZE + plen]
    if len(payload) != plen:
      return None

    now = time.monotonic()
    with self._lock:
      # Drop stale incomplete frames
      stale = [s for s, slot in self._pending.items() if now - slot["t0"] > FRAME_STALE_S]
      for s in stale:
        self._pending.pop(s, None)
      while len(self._pending) > PENDING_MAX_FRAMES:
        oldest = min(self._pending.items(), key=lambda kv: kv[1]["t0"])[0]
        self._pending.pop(oldest, None)

      slot = self._pending.get(seq)
      if slot is None:
        slot = {"ts": ts, "w": w, "h": h, "fc": fc, "parts": {}, "t0": now}
        self._pending[seq] = slot
      elif slot["fc"] != fc:
        return None

      slot["parts"][fi] = payload
      if len(slot["parts"]) < fc:
        return None

      try:
        jpeg = b"".join(slot["parts"][i] for i in range(fc))
      except KeyError:
        self._pending.pop(seq, None)
        return None
      self._pending.pop(seq, None)

      # Prefer newest complete frame only
      if seq < self._last_complete_seq:
        return None
      self._last_complete_seq = seq
      self._latest_jpeg = jpeg
      self._latest_meta = (int(ts), int(w), int(h))
      return jpeg, int(ts), int(w), int(h)

  def latest(self) -> tuple[bytes, int, int, int] | None:
    with self._lock:
      if self._latest_jpeg is None or self._latest_meta is None:
        return None
      ts, w, h = self._latest_meta
      return self._latest_jpeg, ts, w, h


def _publish_map_data(pm: messaging.PubMaster, pkt: dict[str, Any]) -> None:
  msg = messaging.new_message("carLifeMapData")
  d = msg.carLifeMapData
  d.schemaVersion = int(pkt.get("schemaVersion") or 1)
  d.timestamp = int(pkt.get("timestamp") or 0)
  d.laneCount = int(pkt.get("laneCount", -1))
  d.currentLane = int(pkt.get("currentLane", -1))
  d.lightStatus = str(pkt.get("lightStatus") or "unknown")
  d.countdown = int(pkt.get("countdown", -1))
  dist = pkt.get("intersectionDistance")
  if dist is None:
    d.hasIntersectionDistance = False
    d.intersectionDistance = 0.0
  else:
    d.hasIntersectionDistance = True
    d.intersectionDistance = float(dist)
  d.speedLimit = int(pkt.get("speedLimit", -1))
  kappa = pkt.get("curveCurvature")
  if kappa is None:
    d.hasCurveCurvature = False
    d.curveCurvature = 0.0
  else:
    d.hasCurveCurvature = True
    d.curveCurvature = float(kappa)
  d.laneDirection = str(pkt.get("laneDirection") or "unknown")
  d.turnDirection = str(pkt.get("turnDirection") or "unknown")
  d.laneChange = str(pkt.get("laneChange") or "none")
  d.speedAction = str(pkt.get("speedAction") or "unknown")
  d.confidence = float(pkt.get("confidence") or 0.0)
  d.lightConfidence = float(pkt.get("lightConfidence") or 0.0)
  d.speedConfidence = float(pkt.get("speedConfidence") or 0.0)
  d.laneConfidence = float(pkt.get("laneConfidence") or 0.0)
  d.actionConfidence = float(pkt.get("actionConfidence") or 0.0)
  d.pathConfidence = float(pkt.get("pathConfidence") or 0.0)
  d.isOccluded = bool(pkt.get("isOccluded", False))

  raw_lanes = [item for item in (pkt.get("lanes") or []) if isinstance(item, dict)]
  lanes = d.init("lanes", len(raw_lanes))
  for i, item in enumerate(raw_lanes):
    lanes[i].index = _as_int(item.get("index"), 0)
    lanes[i].highlighted = bool(item.get("highlighted", False))
    dirs_src = [_norm_dir(x) for x in (item.get("directions") or [])]
    dirs = lanes[i].init("directions", len(dirs_src))
    for j, name in enumerate(dirs_src):
      dirs[j] = name

  rec = [int(x) for x in (pkt.get("recommendedLanes") or [])]
  rl = d.init("recommendedLanes", len(rec))
  for i, idx in enumerate(rec):
    rl[i] = idx

  ego = pkt.get("egoCar")
  if isinstance(ego, dict):
    d.hasEgoCar = True
    d.egoCarX = float(ego.get("x") or 0.0)
    d.egoCarY = float(ego.get("y") or 0.0)
  else:
    d.hasEgoCar = False
    d.egoCarX = 0.0
    d.egoCarY = 0.0

  path = [p for p in (pkt.get("navPath") or []) if isinstance(p, dict)]
  np_list = d.init("navPath", len(path))
  for i, p in enumerate(path):
    np_list[i].x = float(p.get("x") or 0.0)
    np_list[i].y = float(p.get("y") or 0.0)

  pm.send("carLifeMapData", msg)


def _decode_jpeg_rgba(jpeg: bytes, width_hint: int, height_hint: int) -> tuple[bytes, int, int] | None:
  try:
    img = Image.open(BytesIO(jpeg))
    img = img.convert("RGBA")
  except Exception:
    return None
  w, h = img.size
  if w <= 0 or h <= 0 or w > MAX_W or h > MAX_H:
    # Downscale if somehow oversized
    if w <= 0 or h <= 0:
      return None
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
      nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
      img = img.resize((nw, nh), Image.BILINEAR)
      w, h = nw, nh
    else:
      return None
  arr = np.asarray(img, dtype=np.uint8)
  return arr.tobytes(), w, h


class CarLifeD:
  def __init__(self):
    self.shm = CarLifeFrameShm(create=True)
    self.pm = messaging.PubMaster(["carLifeMapData"])
    self.state = StateHold()
    self.video = VideoAssembler()
    self._stop = threading.Event()
    self._data_pkts = 0
    self._video_pkts = 0
    self._video_frames = 0
    self._last_data_addr = ""
    self._last_video_addr = ""

  @staticmethod
  def _udp_socket(port: int, bufsize: int) -> socket.socket:
    """Bind UDP for unicast + LAN broadcast (phone→AP→car)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    except OSError:
      pass
    try:
      # Absorb bursty 20Hz CLVF fragments without drop
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
    except OSError:
      pass
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    cloudlog.info("carlifed UDP bound 0.0.0.0:%d (rcvbuf~%d)", port, bufsize)
    return sock

  def _data_loop(self) -> None:
    sock = self._udp_socket(DATA_PORT, DATA_RECV_BUF)
    while not self._stop.is_set():
      try:
        raw, addr = sock.recvfrom(DATA_RECV_BUF)
      except socket.timeout:
        continue
      except OSError:
        break
      self._data_pkts += 1
      self._last_data_addr = f"{addr[0]}:{addr[1]}"
      if self._data_pkts == 1:
        cloudlog.warning("carlifed first DATA packet from %s (%d bytes)", self._last_data_addr, len(raw))
      try:
        pkt = json.loads(raw.decode("utf-8"))
      except Exception:
        continue
      if not isinstance(pkt, dict):
        continue
      held = self.state.update(pkt)
      try:
        _publish_map_data(self.pm, held)
      except Exception:
        cloudlog.exception("carlifed publish map data failed")
    sock.close()

  def _video_loop(self) -> None:
    sock = self._udp_socket(VIDEO_PORT, VIDEO_RECV_BUF)
    while not self._stop.is_set():
      try:
        raw, addr = sock.recvfrom(VIDEO_RECV_BUF)
      except socket.timeout:
        continue
      except OSError:
        break
      self._video_pkts += 1
      self._last_video_addr = f"{addr[0]}:{addr[1]}"
      if self._video_pkts == 1:
        cloudlog.warning("carlifed first VIDEO packet from %s (%d bytes)", self._last_video_addr, len(raw))
      complete = self.video.ingest(raw)
      if complete is None:
        continue
      jpeg, ts, w, h = complete
      decoded = _decode_jpeg_rgba(jpeg, w, h)
      if decoded is None:
        continue
      rgba, dw, dh = decoded
      enable = 1
      hdr = self.shm.read_header()
      if hdr is not None:
        enable = int(hdr.enable)
      try:
        self.shm.publish_frame(rgba, dw, dh, timestamp_ms=ts, ready=True, enable=enable)
        self._video_frames += 1
      except Exception:
        cloudlog.exception("carlifed shm publish failed")
    sock.close()

  def _write_status(self) -> None:
    """UI-readable RX counters (survives without cereal)."""
    payload = {
      "data_pkts": int(self._data_pkts),
      "video_pkts": int(self._video_pkts),
      "video_frames": int(self._video_frames),
      "last_data_addr": self._last_data_addr or "",
      "last_video_addr": self._last_video_addr or "",
      "ts": time.time(),
    }
    try:
      tmp = STATUS_PATH + ".tmp"
      with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
      os.replace(tmp, STATUS_PATH)
    except Exception:
      pass

  def run(self) -> None:
    t_data = threading.Thread(target=self._data_loop, name="carlife-data", daemon=True)
    t_video = threading.Thread(target=self._video_loop, name="carlife-video", daemon=True)
    t_data.start()
    t_video.start()
    rk = Ratekeeper(1.0, print_delay_threshold=None)
    last_d = last_v = last_f = 0
    while True:
      # 1Hz status — helps diagnose phone target IP / stream issues
      d, v, f = self._data_pkts, self._video_pkts, self._video_frames
      self._write_status()
      if d != last_d or v != last_v or f != last_f:
        cloudlog.info(
          "carlifed rx data=%d(+%d) from=%s video_pkts=%d(+%d) frames=%d(+%d) from=%s",
          d, d - last_d, self._last_data_addr or "-",
          v, v - last_v, f, f - last_f, self._last_video_addr or "-",
        )
        last_d, last_v, last_f = d, v, f
      elif d == 0 and v == 0:
        # Periodic reminder while idle — easy to grep
        cloudlog.info("carlifed waiting for UDP on :%d/:%d (unicast to this host preferred)", DATA_PORT, VIDEO_PORT)
      rk.keep_time()


def main() -> None:
  _lower_priority()
  cloudlog.info("carlifed starting (UDP %d/%d)", DATA_PORT, VIDEO_PORT)
  CarLifeD().run()


if __name__ == "__main__":
  main()
