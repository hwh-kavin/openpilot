"""
amapd — isolated low-priority Amap tile fetch + stitch/crop at 2Hz.

Pipeline (independent of screen resolution for downloads):
  1. Fetch a fixed 2x2 (4) tiles around GPS at native 256px
  2. Stitch north-up (no map rotation); UI rotates the GPS arrow by bearing
  3. Crop to display aspect ratio centered on GPS
  4. Upscale the crop to the UI panel size

Road names use Amap Web服务 reverse geocoding (AmapWebServiceKey).

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
from PIL import Image, ImageDraw

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from bluepilot.mapd.amap_ipc import AmapFrameShm, MAX_H, MAX_W
from bluepilot.mapd.coords import wgs84_to_gcj02
from bluepilot.mapd import nav_params as navp

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
ROUTE_COLOR = (30, 144, 255, 230)
DEST_COLOR = (220, 40, 40, 240)


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
  return wgs84_to_gcj02(lat, lon)


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


def _parse_amap_address(data: dict) -> str:
  """Pick a short on-road label from Amap regeo (Web服务)."""
  regeocode = data.get("regeocode") or {}
  addr = regeocode.get("addressComponent") or {}
  if isinstance(addr, dict):
    street_num = addr.get("streetNumber") or {}
    if isinstance(street_num, dict):
      street = _amap_str(street_num.get("street"))
      number = _amap_str(street_num.get("number"))
      if street and number:
        return f"{street}{number}"
      if street:
        return street

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

  if isinstance(addr, dict):
    township = _amap_str(addr.get("township"))
    if township:
      return township
    district = _amap_str(addr.get("district"))
    if district:
      return district

  formatted = _amap_str(regeocode.get("formatted_address"))
  if formatted:
    # Keep within shm budget (~40 CJK chars with 128-byte field).
    return formatted if len(formatted) <= 24 else formatted[:23] + "…"
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

  def _network_ok(self) -> bool:
    """Any online network (wifi/cell/ethernet) — regeo is tiny vs tile downloads."""
    if not self.sm.valid.get("deviceState"):
      return False
    from cereal import log
    return self.sm["deviceState"].networkType != log.DeviceState.NetworkType.none

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

  def _fetch_road_name(self, lat: float, lon: float) -> str:
    """Resolve nearby address via Amap Web服务 reverse geocoding (GCJ-02)."""
    now = time.monotonic()
    moved = _haversine_m(lat, lon, self._road_name_lat, self._road_name_lon)
    if (self._road_name_ts > 0 and
        (now - self._road_name_ts) < REGEO_MIN_INTERVAL_S and
        moved < REGEO_MIN_MOVE_M):
      return self._road_name

    web_key = navp.get_web_service_key(self.params)
    if not web_key or not self._network_ok():
      return self._road_name

    self._road_name_lat = lat
    self._road_name_lon = lon
    self._road_name_ts = now
    try:
      gcj_lat, gcj_lon = _wgs84_to_gcj02(lat, lon)
      params = {
        "key": web_key,
        "location": f"{gcj_lon:.6f},{gcj_lat:.6f}",
        "extensions": "all",
        "roadlevel": "0",
        "radius": "100",
      }
      resp = requests.get(REGEO_URL, params=params, timeout=REGEO_TIMEOUT)
      resp.raise_for_status()
      data = resp.json()
      if str(data.get("status")) != "1":
        cloudlog.warning("amapd regeo failed: %s %s", data.get("infocode"), data.get("info"))
        return self._road_name
      name = _parse_amap_address(data)
      if name:
        self._road_name = name
      return self._road_name
    except Exception:
      cloudlog.exception("amapd regeo request failed")
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
  def _crop_aspect(img: Image.Image, gps_x: float, gps_y: float, aspect: float) -> Image.Image:
    """Largest aspect crop centered on GPS that fits inside the north-up stitch."""
    w, h = img.size
    max_hh = min(gps_y, h - gps_y, w / (2.0 * aspect), h / 2.0)
    hh = max(4.0, float(max_hh))
    hw = hh * aspect
    left = max(0, int(math.floor(gps_x - hw)))
    top = max(0, int(math.floor(gps_y - hh)))
    right = min(w, int(math.ceil(gps_x + hw)))
    bottom = min(h, int(math.ceil(gps_y + hh)))
    if right - left < 2 or bottom - top < 2:
      return img
    return img.crop((left, top, right, bottom))

  def _gcj_to_stitch_px(self, lat: float, lon: float, tx0: int, ty0: int) -> tuple[float, float]:
    cx, cy, frac_x, frac_y = _latlon_to_tile_fractional(lat, lon, self.zoom)
    return (cx + frac_x - tx0) * TILE_SIZE, (cy + frac_y - ty0) * TILE_SIZE

  def _draw_route(self, canvas: Image.Image, tx0: int, ty0: int) -> None:
    route = navp.get_route_geometry(self.params)
    if not route:
      return
    coords = route.get("coordinates") or []
    if len(coords) < 2:
      return
    pts = []
    # Downsample for draw performance on large routes
    step = max(1, len(coords) // 300)
    for pt in coords[::step]:
      try:
        plat, plon = float(pt["latitude"]), float(pt["longitude"])
      except (KeyError, TypeError, ValueError):
        continue
      pts.append(self._gcj_to_stitch_px(plat, plon, tx0, ty0))
    if len(pts) < 2:
      return
    draw = ImageDraw.Draw(canvas)
    # Outline then fill for contrast
    draw.line(pts, fill=(255, 255, 255, 200), width=8)
    draw.line(pts, fill=ROUTE_COLOR, width=5)
    dest = route.get("destination") or {}
    try:
      dlat, dlon = float(dest["latitude"]), float(dest["longitude"])
      dx, dy = self._gcj_to_stitch_px(dlat, dlon, tx0, ty0)
      r = 10
      draw.ellipse((dx - r, dy - r, dx + r, dy + r), fill=DEST_COLOR, outline=(255, 255, 255, 220))
    except (KeyError, TypeError, ValueError):
      pass

  def _render_viewport(self, view_w: int, view_h: int, lat: float, lon: float) -> tuple[bytes | None, bool]:
    """Returns (rgba_bytes, ready). Downloads at most 4 native tiles. Map stays north-up."""
    gcj_lat, gcj_lon = _wgs84_to_gcj02(lat, lon)
    cx, cy, frac_x, frac_y = _latlon_to_tile_fractional(gcj_lat, gcj_lon, self.zoom)
    tx0, ty0, gps_px, gps_py = self._tile_block_origin(cx, cy, frac_x, frac_y)

    self._load_block(tx0, ty0)
    ready = self._has_block(tx0, ty0)

    # 1) Stitch 2x2 at native tile resolution (512x512), north-up
    canvas = Image.new("RGBA", (STITCH_SIZE, STITCH_SIZE), BG)
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

    # 1b) Paint route polyline in north-up stitch space
    try:
      self._draw_route(canvas, tx0, ty0)
    except Exception:
      pass

    # 2) Aspect crop centered on GPS (no rotation)
    aspect = view_w / max(view_h, 1)
    cropped = self._crop_aspect(canvas, gps_px, gps_py, aspect)

    # 3) Upscale to panel resolution
    out = cropped.resize((view_w, view_h), Image.BILINEAR)
    arr = np.ascontiguousarray(np.array(out, dtype=np.uint8))
    return arr.tobytes(), ready

  def step(self) -> None:
    self.sm.update(0)
    hdr = self.shm.read_header()
    enable = bool(hdr and hdr.enable)
    api_key = self.params.get("AmapApiKey") or ""
    security_code = self.params.get("AmapSecurityJsCode") or ""
    if not enable or not api_key or not security_code:
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

    road_name = self._fetch_road_name(lat, lon)
    rgba, ready = self._render_viewport(view_w, view_h, lat, lon)
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
