"""Shared helpers for BluePilot navigation params (destination / home / work / settings / route)."""

from __future__ import annotations

import json
from typing import Any

from openpilot.common.params import Params

PLACE_KEYS = ("latitude", "longitude", "place_name", "place_details")

DEFAULT_NAV_SETTINGS: dict[str, Any] = {
  "enabled": True,
  "avoid_tolls": False,
  "avoid_highway": False,
  "strategy": 0,  # Amap driving: 0=speed, 1=cost, 2=distance, ...
}


def _as_dict(raw) -> dict[str, Any] | None:
  if raw is None:
    return None
  if isinstance(raw, dict):
    return raw
  if isinstance(raw, (bytes, bytearray)):
    raw = raw.decode("utf-8", errors="ignore")
  if isinstance(raw, str):
    raw = raw.strip()
    if not raw:
      return None
    try:
      data = json.loads(raw)
    except json.JSONDecodeError:
      return None
    return data if isinstance(data, dict) else None
  return None


def normalize_place(data: dict[str, Any] | None) -> dict[str, Any] | None:
  if not data:
    return None
  try:
    lat = float(data.get("latitude"))
    lon = float(data.get("longitude"))
  except (TypeError, ValueError):
    return None
  if abs(lat) < 1e-7 and abs(lon) < 1e-7:
    return None
  if not (-90 <= lat <= 90 and -180 <= lon <= 180):
    return None
  name = data.get("place_name")
  details = data.get("place_details")
  return {
    "latitude": lat,
    "longitude": lon,
    "place_name": str(name) if name else None,
    "place_details": str(details) if details else None,
    "coord_type": str(data.get("coord_type") or "gcj02"),
  }


def get_json_param(params: Params, key: str) -> dict[str, Any] | None:
  return _as_dict(params.get(key))


def put_json_param(params: Params, key: str, value: dict[str, Any] | None, *, block: bool = True) -> None:
  """Write JSON param with blocking put so it is fsync'd before return (survives power loss)."""
  if value is None:
    params.remove(key)
  else:
    # ParamKeyType.JSON expects a dict, not a JSON string.
    # block=True uses the synchronous put path which fsyncs file + directory.
    params.put(key, value, block=block)


def get_place(params: Params, key: str) -> dict[str, Any] | None:
  return normalize_place(get_json_param(params, key))


def set_place(params: Params, key: str, data: dict[str, Any] | None) -> dict[str, Any] | None:
  place = normalize_place(data) if data else None
  put_json_param(params, key, place)
  return place


def get_nav_settings(params: Params) -> dict[str, Any]:
  raw = get_json_param(params, "NavSettings") or {}
  out = dict(DEFAULT_NAV_SETTINGS)
  for k, default in DEFAULT_NAV_SETTINGS.items():
    if k not in raw:
      continue
    val = raw[k]
    if isinstance(default, bool):
      out[k] = bool(val)
    elif isinstance(default, int):
      try:
        out[k] = int(val)
      except (TypeError, ValueError):
        pass
    else:
      out[k] = val
  return out


def set_nav_settings(params: Params, data: dict[str, Any] | None) -> dict[str, Any]:
  merged = get_nav_settings(params)
  if data:
    for k in DEFAULT_NAV_SETTINGS:
      if k in data:
        merged[k] = data[k]
  # normalize types
  merged = get_nav_settings_from_dict(merged)
  put_json_param(params, "NavSettings", merged)
  return merged


def get_nav_settings_from_dict(data: dict[str, Any]) -> dict[str, Any]:
  out = dict(DEFAULT_NAV_SETTINGS)
  out["enabled"] = bool(data.get("enabled", True))
  out["avoid_tolls"] = bool(data.get("avoid_tolls", False))
  out["avoid_highway"] = bool(data.get("avoid_highway", False))
  try:
    out["strategy"] = int(data.get("strategy", 0))
  except (TypeError, ValueError):
    out["strategy"] = 0
  return out


def clear_navigation(params: Params, clear_destination: bool = True) -> None:
  if clear_destination:
    params.remove("NavDestination")
  params.remove("NavRouteGeometry")
  try:
    params.remove("NavRouteError")
  except Exception:
    pass
  try:
    params.remove("NavGuidance")
  except Exception:
    pass


def get_route_geometry(params: Params) -> dict[str, Any] | None:
  return get_json_param(params, "NavRouteGeometry")


def set_route_geometry(params: Params, data: dict[str, Any] | None) -> None:
  put_json_param(params, "NavRouteGeometry", data)
  if data:
    try:
      params.remove("NavRouteError")
    except Exception:
      pass
  else:
    try:
      params.remove("NavGuidance")
    except Exception:
      pass


def get_guidance(params: Params) -> dict[str, Any] | None:
  return get_json_param(params, "NavGuidance")


def set_guidance(params: Params, data: dict[str, Any] | None) -> None:
  put_json_param(params, "NavGuidance", data, block=False)


def set_route_error(params: Params, message: str | None) -> None:
  if not message:
    try:
      params.remove("NavRouteError")
    except Exception:
      pass
  else:
    params.put("NavRouteError", message, block=True)


def get_route_error(params: Params) -> str:
  try:
    return params.get("NavRouteError") or ""
  except Exception:
    return ""


def get_web_service_key(params: Params) -> str:
  """Web服务 Key required for REST driving/regeo (JS Key cannot call restapi)."""
  try:
    return params.get("AmapWebServiceKey") or ""
  except Exception:
    return ""
