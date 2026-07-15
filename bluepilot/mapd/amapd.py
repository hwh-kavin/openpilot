"""
amapd — isolated low-priority Amap tile fetch + stitch/rotate/crop at 2Hz.

Pipeline (independent of screen resolution for downloads):
  1. Fetch a fixed 2x2 (4) tiles around GPS at native 256px
  2. Stitch → rotate heading-up around GPS
  3. Crop to display aspect ratio (as large as content allows)
  4. Upscale the crop to the UI panel size

Writes a finished RGBA viewport into shared memory for the UI to blit.
Never touches OpenGL; must not interfere with modeld/selfdrived.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections import OrderedDict
from io import BytesIO

import cereal.messaging as messaging
import numpy as np
import requests
from PIL import Image

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from bluepilot.mapd.amap_ipc import AmapFrameShm, MAX_H, MAX_W

TILE_URL = "https://webrd0{s}.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scale=1&style=8"
REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"
TILE_SIZE = 256
GRID = 2                    # always 2x2 = 4 tiles
STITCH_SIZE = TILE_SIZE * GRID  # 512
DEFAULT_ZOOM = 17
UPDATE_HZ = 2.0
MAX_CACHED_TILES = 16
HTTP_TIMEOUT = 5
REGEO_TIMEOUT = 3
REGEO_MIN_INTERVAL_S = 3.0
REGEO_MIN_MOVE_M = 40.0
TILE_CACHE_DIR = "/tmp/amap_tiles"
BG = (30, 30, 30, 255)
# Rotation void color — unique so we can detect "black edge" after rotate.
FILL = (255, 0, 254, 255)
FILL_MASK = 0  # mask fill after rotate (0 = void, 255 = map content)

_GCJ_A = 6378245.0
_GCJ_EE = 0.00669342162296594323


def _lower_priority() -> None:
  """Run at the lowest practical priority so autonomy stays unperturbed."""
  try:
    os.nice(19)
  except Exception:
    pass
  try:
    with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
      f.write("1000")
  except Exception:
    pass
  # Best-effort SCHED_IDLE (Linux) — never raise into FIFO/RR
  try:
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    class SchedParam(ctypes.Structure):
      _fields_ = [("sched_priority", ctypes.c_int)]

    SCHED_IDLE = 5
    libc.sched_setscheduler(0, SCHED_IDLE, ctypes.byref(SchedParam(0)))
  except Exception:
    pass
  try:
    os.system(f"ionice -c 3 -p {os.getpid()} >/dev/null 2>&1")
  except Exception:
    pass


def _wgs84_to_gcj02(lat: float, lon: float) -> tuple[float, float]:
  if not (-180 <= lon <= 180 and -90 <= lat <= 90):
    return lat, lon

  def _trans_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

  def _trans_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

  dlat = _trans_lat(lon - 105.0, lat - 35.0)
  dlon = _trans_lon(lon - 105.0, lat - 35.0)
  radlat = lat / 180.0 * math.pi
  magic = math.sin(radlat)
  magic = 1.0 - _GCJ_EE * magic * magic
  sqrtmagic = math.sqrt(magic)
  dlat = (dlat * 180.0) / ((_GCJ_A * (1.0 - _GCJ_EE)) / (magic * sqrtmagic) * math.pi)
  dlon = (dlon * 180.0) / (_GCJ_A / sqrtmagic * math.cos(radlat) * math.pi)
  return lat + dlat, lon + dlon


def _latlon_to_tile_fractional(lat: float, lon: float, zoom: int) -> tuple[int, int, float, float]:
  n = 2.0 ** zoom
  fx = (lon + 180.0) / 360.0 * n
  lat_rad = math.radians(lat)
  fy = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
  tx = int(fx)
  ty = int(fy)
  return tx, ty, fx - tx, fy - ty


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  r = 6371000.0
  p1, p2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlmb = math.radians(lon2 - lon1)
  a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
  return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _amap_str(val) -> str:
  """Amap often returns [] for missing string fields."""
  return val if isinstance(val, str) and val else ""


def _parse_amap_road_name(data: dict) -> str:
  regeocode = data.get("regeocode") or {}
  roads = regeocode.get("roads") or []
  best_name = ""
  best_dist = float("inf")
  if isinstance(roads, list):
    for road in roads:
      if not isinstance(road, dict):
        continue
      name = _amap_str(road.get("name"))
      if not name:
        continue
      try:
        dist = float(road.get("distance") or 1e9)
      except (TypeError, ValueError):
        dist = 1e9
      if dist < best_dist:
        best_dist = dist
        best_name = name
  if best_name:
    return best_name

  addr = regeocode.get("addressComponent") or {}
  if isinstance(addr, dict):
    street_num = addr.get("streetNumber") or {}
    if isinstance(street_num, dict):
      street = _amap_str(street_num.get("street"))
      if street:
        return street
    township = _amap_str(addr.get("township"))
    if township:
      return township
  return ""


class AmapWorker:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster([
      "gpsLocationExternal", "gpsLocation", "deviceState", "livePose",
    ])
    self.shm = AmapFrameShm(create=True)
    self.zoom = DEFAULT_ZOOM
    self._tile_images: OrderedDict[tuple[int, int, int], Image.Image] = OrderedDict()
    self._last_bearing = 0.0
    self._have_bearing = False
    self._road_name = ""
    self._road_name_lat = 0.0
    self._road_name_lon = 0.0
    self._road_name_ts = 0.0
    os.makedirs(TILE_CACHE_DIR, exist_ok=True)

  def _wifi_ok(self) -> bool:
    if not self.sm.valid.get("deviceState"):
      return False
    from cereal import log
    return self.sm["deviceState"].networkType == log.DeviceState.NetworkType.wifi

  @staticmethod
  def _bearing_from_vned(g) -> float | None:
    """Course-over-ground from NED velocity when moving fast enough."""
    try:
      speed = float(g.speed)
      vned = g.vNED
      if speed < 1.0 or len(vned) < 2:
        return None
      vn, ve = float(vned[0]), float(vned[1])
      if abs(vn) < 1e-3 and abs(ve) < 1e-3:
        return None
      return math.degrees(math.atan2(ve, vn)) % 360.0
    except Exception:
      return None

  @staticmethod
  def _bearing_from_gps_field(g) -> float | None:
    """Use reported bearingDeg only when accuracy looks usable."""
    try:
      acc = float(getattr(g, "bearingAccuracyDeg", 180.0) or 180.0)
      # 180° is the conventional "invalid / unknown" sentinel in openpilot GPS.
      if acc >= 45.0:
        return None
      return float(g.bearingDeg) % 360.0
    except Exception:
      return None

  def _bearing_from_live_pose(self) -> float | None:
    if not self.sm.valid.get("livePose"):
      return None
    try:
      orient = self.sm["livePose"].orientationNED
      if not orient.valid:
        return None
      # NED yaw: 0 = north, positive toward east (same as GPS bearing).
      return math.degrees(float(orient.z)) % 360.0
    except Exception:
      return None

  def _resolve_bearing(self, g) -> float:
    bearing = self._bearing_from_vned(g)
    if bearing is None:
      bearing = self._bearing_from_gps_field(g)
    if bearing is None:
      bearing = self._bearing_from_live_pose()
    if bearing is None:
      return self._last_bearing if self._have_bearing else 0.0
    self._last_bearing = bearing
    self._have_bearing = True
    return bearing

  def _get_gps(self) -> tuple[float, float, float, bool]:
    for key in ("gpsLocationExternal", "gpsLocation"):
      if not self.sm.valid.get(key):
        continue
      g = self.sm[key]
      if hasattr(g, "hasFix") and not g.hasFix:
        continue
      lat, lon = float(g.latitude), float(g.longitude)
      if abs(lat) < 1e-7 and abs(lon) < 1e-7:
        continue
      return lat, lon, self._resolve_bearing(g), True
    return 0.0, 0.0, self._last_bearing if self._have_bearing else 0.0, False

  def _fetch_road_name(self, lat: float, lon: float, api_key: str) -> str:
    """Resolve nearby road name via Amap reverse geocoding (GCJ-02)."""
    now = time.monotonic()
    moved = _haversine_m(lat, lon, self._road_name_lat, self._road_name_lon)
    if (self._road_name_ts > 0 and
        (now - self._road_name_ts) < REGEO_MIN_INTERVAL_S and
        moved < REGEO_MIN_MOVE_M):
      return self._road_name
    if not api_key or not self._wifi_ok():
      return self._road_name

    self._road_name_lat = lat
    self._road_name_lon = lon
    self._road_name_ts = now
    try:
      gcj_lat, gcj_lon = _wgs84_to_gcj02(lat, lon)
      resp = requests.get(
        REGEO_URL,
        params={
          "key": api_key,
          "location": f"{gcj_lon:.6f},{gcj_lat:.6f}",
          "extensions": "all",
          "roadlevel": "0",
          "radius": "100",
        },
        timeout=REGEO_TIMEOUT,
      )
      resp.raise_for_status()
      data = resp.json()
      if str(data.get("status")) != "1":
        return self._road_name
      name = _parse_amap_road_name(data)
      if name:
        self._road_name = name
      return self._road_name
    except Exception:
      return self._road_name

  def _load_tile(self, x: int, y: int, zoom: int) -> Image.Image | None:
    key = (x, y, zoom)
    if key in self._tile_images:
      self._tile_images.move_to_end(key)
      return self._tile_images[key]

    cache_path = os.path.join(TILE_CACHE_DIR, f"{zoom}_{x}_{y}.png")
    try:
      if os.path.exists(cache_path) and os.path.getsize(cache_path) >= 8:
        with open(cache_path, "rb") as f:
          if f.read(4) == b"\x89PNG":
            img = Image.open(cache_path).convert("RGBA")
            self._remember_tile(key, img)
            return img
        os.remove(cache_path)

      if not self._wifi_ok():
        return None

      subdomain = random.randint(1, 4)
      url = TILE_URL.format(s=subdomain, x=x, y=y, z=zoom)
      resp = requests.get(url, timeout=HTTP_TIMEOUT)
      resp.raise_for_status()
      if not resp.content.startswith(b"\x89PNG"):
        return None
      tmp = cache_path + ".tmp"
      with open(tmp, "wb") as f:
        f.write(resp.content)
      os.replace(tmp, cache_path)
      img = Image.open(BytesIO(resp.content)).convert("RGBA")
      self._remember_tile(key, img)
      return img
    except Exception:
      return None

  def _remember_tile(self, key: tuple[int, int, int], img: Image.Image) -> None:
    self._tile_images[key] = img
    while len(self._tile_images) > MAX_CACHED_TILES:
      self._tile_images.popitem(last=False)

  @staticmethod
  def _tile_block_origin(cx: int, cy: int, frac_x: float, frac_y: float) -> tuple[int, int, float, float]:
    """Pick the 2x2 tile block that keeps GPS as near the center as possible.

    Returns (tx0, ty0, gps_px, gps_py) in the 512x512 stitch space.
    """
    fx = cx + frac_x
    fy = cy + frac_y
    tx0 = int(math.floor(fx - 0.5))
    ty0 = int(math.floor(fy - 0.5))
    gps_px = (fx - tx0) * TILE_SIZE
    gps_py = (fy - ty0) * TILE_SIZE
    return tx0, ty0, gps_px, gps_py

  def _load_block(self, tx0: int, ty0: int) -> tuple[list[tuple[int, int]], int]:
    """Load the 4 tiles of the 2x2 block. Returns (keys, loaded_count)."""
    keys = []
    loaded = 0
    for dy in range(GRID):
      for dx in range(GRID):
        key = (tx0 + dx, ty0 + dy, self.zoom)
        keys.append(key)
        if self._load_tile(*key) is not None:
          loaded += 1
    return keys, loaded

  def _has_block(self, tx0: int, ty0: int) -> bool:
    for dy in range(GRID):
      for dx in range(GRID):
        if (tx0 + dx, ty0 + dy, self.zoom) not in self._tile_images:
          return False
    return True

  @staticmethod
  def _crop_box(gps_x: float, gps_y: float, aspect: float, half_h: float,
                img_w: int, img_h: int) -> tuple[int, int, int, int]:
    hh = half_h
    hw = hh * aspect
    left = int(math.floor(gps_x - hw))
    top = int(math.floor(gps_y - hh))
    right = int(math.ceil(gps_x + hw))
    bottom = int(math.ceil(gps_y + hh))
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)
    return left, top, right, bottom

  @staticmethod
  def _crop_has_void(mask: np.ndarray, box: tuple[int, int, int, int], threshold: int = 200) -> bool:
    """True if the crop window still contains rotation fill (black edge)."""
    left, top, right, bottom = box
    if right - left < 2 or bottom - top < 2:
      return True
    region = mask[top:bottom, left:right]
    # Any void/soft-edge pixel means we still have black border risk.
    return bool(region.min() < threshold)

  @classmethod
  def _crop_no_black(cls, img: Image.Image, mask_img: Image.Image,
                     gps_x: float, gps_y: float, aspect: float) -> Image.Image:
    """Largest aspect crop centered on GPS with no rotation fill; shrink (=zoom in) until clean."""
    w, h = img.size
    mask = np.asarray(mask_img, dtype=np.uint8)

    max_hh = min(gps_y, h - gps_y, w / (2.0 * aspect), h / 2.0)
    if max_hh < 4:
      max_hh = 4.0

    # Binary search largest half-height with no void pixels in the crop.
    lo, hi = 4.0, float(max_hh)
    best_box = cls._crop_box(gps_x, gps_y, aspect, lo, w, h)
    for _ in range(16):
      mid = (lo + hi) * 0.5
      box = cls._crop_box(gps_x, gps_y, aspect, mid, w, h)
      if cls._crop_has_void(mask, box):
        hi = mid  # too large → black edge → zoom in (smaller crop)
      else:
        best_box = box
        lo = mid

    # Final safety: if best still voids (degenerate), keep shrinking.
    box = best_box
    hh = max(4.0, (box[3] - box[1]) * 0.5)
    for _ in range(12):
      if not cls._crop_has_void(mask, box):
        break
      hh *= 0.85
      box = cls._crop_box(gps_x, gps_y, aspect, hh, w, h)

    return img.crop(box)

  def _render_viewport(self, view_w: int, view_h: int, lat: float, lon: float, bearing: float) -> tuple[bytes | None, bool]:
    """Returns (rgba_bytes, ready). Downloads at most 4 native tiles."""
    gcj_lat, gcj_lon = _wgs84_to_gcj02(lat, lon)
    cx, cy, frac_x, frac_y = _latlon_to_tile_fractional(gcj_lat, gcj_lon, self.zoom)
    tx0, ty0, gps_px, gps_py = self._tile_block_origin(cx, cy, frac_x, frac_y)

    self._load_block(tx0, ty0)
    ready = self._has_block(tx0, ty0)

    # 1) Stitch 2x2 at native tile resolution (512x512)
    canvas = Image.new("RGBA", (STITCH_SIZE, STITCH_SIZE), BG)
    content_mask = Image.new("L", (STITCH_SIZE, STITCH_SIZE), 255)
    for dy in range(GRID):
      for dx in range(GRID):
        tile = self._tile_images.get((tx0 + dx, ty0 + dy, self.zoom))
        x0 = dx * TILE_SIZE
        y0 = dy * TILE_SIZE
        if tile is None:
          patch = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (50, 50, 50, 255))
          canvas.paste(patch, (x0, y0))
        else:
          canvas.paste(tile, (x0, y0))

    # 2) Rotate heading-up around GPS (void corners get FILL / mask=0)
    rot_kwargs = dict(
      resample=Image.BILINEAR,
      expand=False,
      center=(gps_px, gps_py),
    )
    rotated = canvas.rotate(-bearing, fillcolor=FILL, **rot_kwargs)
    rotated_mask = content_mask.rotate(-bearing, fillcolor=FILL_MASK, **rot_kwargs)

    # 3) Aspect crop; if black/void edges remain, zoom in until clean
    aspect = view_w / max(view_h, 1)
    cropped = self._crop_no_black(rotated, rotated_mask, gps_px, gps_py, aspect)

    # 4) Upscale to panel resolution
    out = cropped.resize((view_w, view_h), Image.BILINEAR)
    arr = np.ascontiguousarray(np.array(out, dtype=np.uint8))
    return arr.tobytes(), ready

  def step(self) -> None:
    self.sm.update(0)
    hdr = self.shm.read_header()
    enable = bool(hdr and hdr.enable)
    api_key = self.params.get("AmapApiKey") or ""
    if not enable or not api_key:
      # Keep header alive but mark not ready
      if hdr is not None:
        hdr.ready = 0
        self.shm.write_header(hdr)
      return

    view_w = max(64, min(hdr.request_w or 640, MAX_W))
    view_h = max(64, min(hdr.request_h or 640, MAX_H))

    lat, lon, bearing, gps_valid = self._get_gps()
    if not gps_valid:
      # Publish dark placeholder so UI can still show "No GPS"
      blank = np.full((view_h, view_w, 4), (30, 30, 30, 255), dtype=np.uint8)
      self.shm.publish_frame(
        blank.tobytes(), view_w, view_h,
        ready=False, gps_valid=False, bearing=0.0, road_name="",
        request_w=view_w, request_h=view_h, enable=1,
      )
      return

    road_name = self._fetch_road_name(lat, lon, api_key)
    rgba, ready = self._render_viewport(view_w, view_h, lat, lon, bearing)
    if rgba is None:
      return

    self.shm.publish_frame(
      rgba, view_w, view_h,
      ready=ready, gps_valid=True, bearing=bearing, road_name=road_name,
      request_w=view_w, request_h=view_h, enable=1,
    )


def main() -> None:
  _lower_priority()
  cloudlog.info("amapd starting (nice=19, %.1f Hz)" % UPDATE_HZ)

  worker = AmapWorker()
  rk = Ratekeeper(UPDATE_HZ, print_delay_threshold=None)
  while True:
    try:
      worker.step()
    except Exception:
      cloudlog.exception("amapd step failed")
    rk.keep_time()


if __name__ == "__main__":
  main()
