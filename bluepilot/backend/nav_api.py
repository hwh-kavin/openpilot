"""BluePilot Portal navigation HTTP helpers."""

from __future__ import annotations

import json
from typing import Any

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

from bluepilot.mapd import nav_params as np
from bluepilot.mapd.coords import wgs84_to_gcj02


PLACE_SLOTS = {
  "destination": "NavDestination",
  "saved_destination": "NavSavedDestination",
  "home": "NavHome",
  "work": "NavWork",
}


def _masked_secret(value: str | None) -> str | None:
  if not value:
    return None
  if len(value) > 8:
    return value[:4] + "****" + value[-4:]
  return "****"


def build_nav_status(params: Params | None = None) -> dict[str, Any]:
  params = params or Params()
  settings = np.get_nav_settings(params)
  destination = np.get_place(params, "NavDestination")
  home = np.get_place(params, "NavHome")
  work = np.get_place(params, "NavWork")
  route = np.get_route_geometry(params)
  api_key = params.get("AmapApiKey") or ""
  security = params.get("AmapSecurityJsCode") or ""
  web_key = np.get_web_service_key(params)
  return {
    "success": True,
    "settings": settings,
    "destination": destination,
    "home": home,
    "work": work,
    "route": {
      "distance_m": route.get("distance_m") if route else None,
      "duration_s": route.get("duration_s") if route else None,
      "step_count": len(route.get("steps") or []) if route else 0,
      "has_geometry": bool(route and route.get("coordinates")),
      "provider": route.get("provider") if route else None,
      "error": np.get_route_error(params) or None,
    },
    "credentials": {
      "has_api_key": bool(api_key),
      "has_security_js_code": bool(security),
      "has_web_service_key": bool(web_key),
      "api_key_masked": _masked_secret(api_key if isinstance(api_key, str) else None),
    },
  }


def handle_get_status() -> tuple[dict[str, Any], int]:
  try:
    return build_nav_status(), 200
  except Exception as e:
    cloudlog.exception("nav status failed")
    return {"success": False, "error": str(e)}, 500


def handle_get_credentials() -> tuple[dict[str, Any], int]:
  """Return Amap JS API credentials for the portal map page (local network only)."""
  try:
    params = Params()
    api_key = params.get("AmapApiKey") or ""
    security = params.get("AmapSecurityJsCode") or ""
    return {
      "success": True,
      "api_key": api_key if isinstance(api_key, str) else "",
      "security_js_code": security if isinstance(security, str) else "",
    }, 200
  except Exception as e:
    return {"success": False, "error": str(e)}, 500


def _gps_from_live() -> tuple[float, float, str] | None:
  """Read a fresh WGS84 fix from cereal. Returns (lat, lon, source) or None."""
  try:
    import cereal.messaging as messaging
  except Exception:
    return None

  try:
    sm = messaging.SubMaster(["gpsLocationExternal", "gpsLocation"])
    for _ in range(20):
      sm.update(50)
      for key in ("gpsLocationExternal", "gpsLocation"):
        if not sm.valid.get(key):
          continue
        g = sm[key]
        if hasattr(g, "hasFix") and not g.hasFix:
          continue
        lat, lon = float(g.latitude), float(g.longitude)
        if abs(lat) < 1e-7 and abs(lon) < 1e-7:
          continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
          continue
        return lat, lon, "device_gps"
  except Exception:
    cloudlog.exception("nav live gps read failed")
  return None


def _gps_from_last_param(params: Params) -> tuple[float, float, str] | None:
  """Fallback to LastGPSPositionLLK (WGS84 JSON)."""
  raw = params.get("LastGPSPositionLLK")
  if not raw:
    return None
  try:
    if isinstance(raw, bytes):
      raw = raw.decode("utf-8", errors="ignore")
    data = json.loads(raw)
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    if abs(lat) < 1e-7 and abs(lon) < 1e-7:
      return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
      return None
    return lat, lon, "last_gps"
  except Exception:
    return None


def handle_get_location() -> tuple[dict[str, Any], int]:
  """Return device location in GCJ-02 for the portal map (live GPS, then last fix)."""
  try:
    params = Params()
    fix = _gps_from_live() or _gps_from_last_param(params)
    if not fix:
      payload = {"success": True, "valid": False}
      _write_nav_location_snapshot(payload)
      return payload, 200
    lat_wgs, lon_wgs, source = fix
    lat_gcj, lon_gcj = wgs84_to_gcj02(lat_wgs, lon_wgs)
    payload = {
      "success": True,
      "valid": True,
      "source": source,
      "latitude": lat_gcj,
      "longitude": lon_gcj,
      "coord_type": "gcj02",
      "wgs84": {"latitude": lat_wgs, "longitude": lon_wgs},
    }
    _write_nav_location_snapshot(payload)
    return payload, 200
  except Exception as e:
    cloudlog.exception("nav location failed")
    return {"success": False, "error": str(e)}, 500


def _write_nav_location_snapshot(payload: dict[str, Any]) -> None:
  """Write a static JSON the map page can read without a portal code reload."""
  try:
    from bluepilot.backend.config import WEBAPP_DIR
    path = WEBAPP_DIR / "nav-location.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
  except Exception:
    pass


def handle_put_settings(data: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
  try:
    params = Params()
    settings = np.set_nav_settings(params, data or {})
    return {"success": True, "settings": settings}, 200
  except Exception as e:
    cloudlog.exception("nav settings save failed")
    return {"success": False, "error": str(e)}, 500


def handle_put_place(slot: str, data: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
  key = PLACE_SLOTS.get(slot)
  if not key:
    return {"success": False, "error": f"Unknown place slot: {slot}"}, 400
  try:
    params = Params()
    if data is None or data == {}:
      np.set_place(params, key, None)
      if slot == "destination":
        # Also clear the onroad "目的地" shortcut target.
        np.set_place(params, "NavSavedDestination", None)
        np.clear_navigation(params, clear_destination=True)
      place = None
    else:
      place = np.set_place(params, key, data)
      if place is None:
        return {"success": False, "error": "Invalid place (need latitude/longitude)"}, 400
      if slot == "destination":
        # Keep a durable copy for the onroad shortcut (survives home/work / clear).
        np.set_place(params, "NavSavedDestination", place)
        # New destination invalidates previous geometry until navd recomputes.
        np.set_route_geometry(params, None)
    return {"success": True, "slot": slot, "place": place}, 200
  except Exception as e:
    cloudlog.exception("nav place save failed")
    return {"success": False, "error": str(e)}, 500


def handle_navigate_to_saved(slot: str) -> tuple[dict[str, Any], int]:
  """Copy NavHome/NavWork/NavSavedDestination into NavDestination."""
  if slot not in ("home", "work", "destination"):
    return {"success": False, "error": "slot must be home, work, or destination"}, 400
  try:
    params = Params()
    src_key = "NavSavedDestination" if slot == "destination" else PLACE_SLOTS[slot]
    place = np.get_place(params, src_key)
    if not place:
      return {"success": False, "error": f"{slot} address is not set"}, 400
    saved = np.set_place(params, "NavDestination", place)
    np.set_route_geometry(params, None)
    return {"success": True, "destination": saved}, 200
  except Exception as e:
    cloudlog.exception("nav navigate-to-saved failed")
    return {"success": False, "error": str(e)}, 500


def handle_clear_navigation() -> tuple[dict[str, Any], int]:
  try:
    params = Params()
    np.clear_navigation(params, clear_destination=True)
    return {"success": True}, 200
  except Exception as e:
    return {"success": False, "error": str(e)}, 500
