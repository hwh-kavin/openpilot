"""
navd — BluePilot navigation planner.

Watches NavDestination, calls Amap Web服务 driving API (AmapWebServiceKey),
writes NavRouteGeometry for amapd overlay.

Note: JS API 2.0 keys cannot call restapi.amap.com (USERKEY_PLAT_NOMATCH).
"""

from __future__ import annotations

import json
import math
import time

import cereal.messaging as messaging
import requests

from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog

from bluepilot.mapd import nav_params as np
from bluepilot.mapd.coords import wgs84_to_gcj02
from bluepilot.mapd.nav_guidance import compute_next_guidance

DRIVING_URL = "https://restapi.amap.com/v3/direction/driving"
UPDATE_HZ = 1.0
REPLAN_INTERVAL_S = 30.0
OFF_ROUTE_M = 80.0
HTTP_TIMEOUT = 8


def _amap_text(val) -> str:
  if val is None or isinstance(val, list):
    return ""
  return str(val).strip()


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  r = 6371000.0
  p1, p2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlmb = math.radians(lon2 - lon1)
  a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
  return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _parse_polyline(polyline: str) -> list[dict[str, float]]:
  coords: list[dict[str, float]] = []
  if not polyline or not isinstance(polyline, str):
    return coords
  for part in polyline.split(";"):
    part = part.strip()
    if not part:
      continue
    try:
      lon_s, lat_s = part.split(",", 1)
      coords.append({"latitude": float(lat_s), "longitude": float(lon_s)})
    except (TypeError, ValueError):
      continue
  return coords


def _min_dist_to_route_m(lat: float, lon: float, coords: list[dict[str, float]]) -> float:
  if not coords:
    return 1e9
  best = 1e9
  # Sample every point (routes usually < ~1k points after amap simplification)
  for pt in coords[:: max(1, len(coords) // 200)]:
    best = min(best, _haversine_m(lat, lon, pt["latitude"], pt["longitude"]))
  return best


class NavWorker:
  def __init__(self):
    self.params = Params()
    self.sm = messaging.SubMaster(["gpsLocationExternal", "gpsLocation", "deviceState"])
    self._last_dest_key = ""
    self._last_plan_ts = 0.0
    self._last_settings_key = ""

  def _wifi_ok(self) -> bool:
    if not self.sm.valid.get("deviceState"):
      return False
    from cereal import log
    return self.sm["deviceState"].networkType == log.DeviceState.NetworkType.wifi

  def _get_gps_wgs(self) -> tuple[float, float, bool]:
    for key in ("gpsLocationExternal", "gpsLocation"):
      if not self.sm.valid.get(key):
        continue
      g = self.sm[key]
      if hasattr(g, "hasFix") and not g.hasFix:
        continue
      lat, lon = float(g.latitude), float(g.longitude)
      if abs(lat) < 1e-7 and abs(lon) < 1e-7:
        continue
      return lat, lon, True
    return 0.0, 0.0, False

  def _fetch_route(self, origin_gcj: tuple[float, float], dest: dict,
                   api_key: str, settings: dict) -> dict | None:
    origin = f"{origin_gcj[1]:.6f},{origin_gcj[0]:.6f}"
    destination = f"{dest['longitude']:.6f},{dest['latitude']:.6f}"
    strategy = int(settings.get("strategy", 0))
    # Amap nostate / avoid flags via extensions & strategy family
    params = {
      "key": api_key,
      "origin": origin,
      "destination": destination,
      "extensions": "all",
      "strategy": str(strategy),
    }
    # Prefer avoid flags via strategy codes when set
    if settings.get("avoid_tolls") and settings.get("avoid_highway"):
      params["strategy"] = "8"  # 不走高速+少收费 approx
    elif settings.get("avoid_tolls"):
      params["strategy"] = "1"
    elif settings.get("avoid_highway"):
      params["strategy"] = "3"

    resp = requests.get(DRIVING_URL, params=params, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if str(data.get("status")) != "1":
      info = data.get("info") or "unknown"
      infocode = data.get("infocode") or ""
      cloudlog.warning(f"amap driving failed: {info} ({infocode})")
      if str(infocode) == "10009" or "PLAT_NOMATCH" in str(info):
        np.set_route_error(self.params, "Key类型错误：路径规划需「Web服务」Key")
      else:
        np.set_route_error(self.params, f"路线规划失败: {info}")
      return None
    route = (data.get("route") or {})
    paths = route.get("paths") or []
    if not paths:
      np.set_route_error(self.params, "未找到可行驶路线")
      return None
    path = paths[0]
    coords: list[dict[str, float]] = []
    steps_out = []
    for step in path.get("steps") or []:
      step_coords = _parse_polyline(step.get("polyline") or "")
      coords.extend(step_coords)
      # Downsample per-step polyline for guidance matching
      stored = step_coords
      if len(stored) > 24:
        stride = max(1, (len(stored) - 1) // 23)
        stored = stored[::stride]
        if step_coords and stored[-1] != step_coords[-1]:
          stored.append(step_coords[-1])
      steps_out.append({
        "instruction": _amap_text(step.get("instruction")),
        "road": _amap_text(step.get("road")),
        "action": _amap_text(step.get("action")),
        "assistant_action": _amap_text(step.get("assistant_action")),
        "distance_m": float(step.get("distance") or 0),
        "duration_s": float(step.get("duration") or 0),
        "coordinates": stored[:32],
      })
    if not coords:
      np.set_route_error(self.params, "路线数据为空")
      return None
    return {
      "provider": "amap",
      "coord_type": "gcj02",
      "distance_m": float(path.get("distance") or 0),
      "duration_s": float(path.get("duration") or 0),
      "coordinates": coords,
      "steps": steps_out[:48],
      "destination": {
        "latitude": dest["latitude"],
        "longitude": dest["longitude"],
        "place_name": dest.get("place_name"),
      },
      "updated_at": time.time(),
    }

  def step(self) -> None:
    self.sm.update(0)
    settings = np.get_nav_settings(self.params)
    dest = np.get_place(self.params, "NavDestination")

    if not settings.get("enabled") or not dest:
      if self._last_dest_key:
        np.set_route_geometry(self.params, None)
        np.set_guidance(self.params, None)
        self._last_dest_key = ""
      return

    dest_key = json.dumps({
      "lat": round(dest["latitude"], 6),
      "lon": round(dest["longitude"], 6),
      "settings": settings,
    }, sort_keys=True)
    settings_key = json.dumps(settings, sort_keys=True)

    api_key = np.get_web_service_key(self.params)
    if not api_key:
      # Do not spam; surface once when destination changes
      if dest_key != self._last_dest_key:
        np.set_route_error(self.params, "请配置高德 Web 服务 Key（路径规划）")
        self._last_dest_key = dest_key
      return

    lat, lon, gps_ok = self._get_gps_wgs()
    if not gps_ok:
      return

    route = np.get_route_geometry(self.params)
    # Always refresh turn guidance from current GPS when a route exists
    try:
      gcj_lat, gcj_lon = wgs84_to_gcj02(lat, lon)
      guidance = compute_next_guidance(route, gcj_lat, gcj_lon)
      np.set_guidance(self.params, guidance)
    except Exception:
      cloudlog.exception("navd guidance update failed")

    now = time.monotonic()
    need_plan = False
    if dest_key != self._last_dest_key or settings_key != self._last_settings_key:
      need_plan = True
    elif not route or not route.get("coordinates"):
      need_plan = True
    elif (now - self._last_plan_ts) >= REPLAN_INTERVAL_S:
      need_plan = True
    else:
      # dest already GCJ; GPS WGS→GCJ for off-route check against GCJ polyline
      gcj_lat, gcj_lon = wgs84_to_gcj02(lat, lon)
      if _min_dist_to_route_m(gcj_lat, gcj_lon, route.get("coordinates") or []) > OFF_ROUTE_M:
        need_plan = True

    if not need_plan:
      return
    if not self._wifi_ok():
      return

    try:
      origin_gcj = wgs84_to_gcj02(lat, lon)
      # Destination from Portal/JS is GCJ; if marked otherwise leave as-is
      dest_gcj = dest
      if dest.get("coord_type") == "wgs84":
        dlat, dlon = wgs84_to_gcj02(dest["latitude"], dest["longitude"])
        dest_gcj = {**dest, "latitude": dlat, "longitude": dlon, "coord_type": "gcj02"}

      geometry = self._fetch_route(origin_gcj, dest_gcj, api_key, settings)
      self._last_plan_ts = now
      self._last_dest_key = dest_key
      self._last_settings_key = settings_key
      if geometry:
        np.set_route_geometry(self.params, geometry)
        cloudlog.info(f"navd planned route: {geometry['distance_m']:.0f}m / {geometry['duration_s']:.0f}s")
        try:
          g = compute_next_guidance(geometry, origin_gcj[0], origin_gcj[1])
          np.set_guidance(self.params, g)
        except Exception:
          pass
    except Exception as e:
      np.set_route_error(self.params, f"路线规划异常: {e}")
      cloudlog.exception("navd plan failed")


def main() -> None:
  cloudlog.info("navd starting (%.1f Hz)" % UPDATE_HZ)
  # Keep modest priority — below modeld but above amapd stitch if possible
  try:
    import os
    os.nice(10)
  except Exception:
    pass

  worker = NavWorker()
  rk = Ratekeeper(UPDATE_HZ, print_delay_threshold=None)
  while True:
    try:
      worker.step()
    except Exception:
      cloudlog.exception("navd step failed")
    rk.keep_time()


if __name__ == "__main__":
  main()
