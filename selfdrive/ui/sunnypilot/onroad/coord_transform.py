import math

_A = 6378245.0
_EE = 0.00669342162296594323


def _transform_lat(lng: float, lat: float) -> float:
  ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
  ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
  ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320.0 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
  return ret


def _transform_lng(lng: float, lat: float) -> float:
  ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
  ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
  ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
  return ret


def _out_of_china(lat: float, lng: float) -> bool:
  return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)


def wgs84_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
  if _out_of_china(lat, lng):
    return lat, lng
  dlat = _transform_lat(lng - 105.0, lat - 35.0)
  dlng = _transform_lng(lng - 105.0, lat - 35.0)
  radlat = lat / 180.0 * math.pi
  magic = math.sin(radlat)
  magic = 1 - _EE * magic * magic
  sqrtmagic = math.sqrt(magic)
  dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * math.pi)
  dlng = (dlng * 180.0) / (_A / sqrtmagic * math.cos(radlat) * math.pi)
  return lat + dlat, lng + dlng


def lat_lng_to_tile(lat: float, lng: float, zoom: int) -> tuple[int, int]:
  n = 2 ** zoom
  x = int((lng + 180.0) / 360.0 * n)
  lat_rad = math.radians(lat)
  y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
  x = max(0, min(x, n - 1))
  y = max(0, min(y, n - 1))
  return x, y
