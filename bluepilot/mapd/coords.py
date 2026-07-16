"""WGS84 / GCJ-02 coordinate helpers for BluePilot maps."""

from __future__ import annotations

import math

_GCJ_A = 6378245.0
_GCJ_EE = 0.00669342162296594323


def wgs84_to_gcj02(lat: float, lon: float) -> tuple[float, float]:
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
