import threading
import time

import requests

REGEO_URL = "https://restapi.amap.com/v3/geocode/regeo"
REQUEST_TIMEOUT = 10
CACHE_GRID = 0.00025  # ~25 m
MIN_FETCH_INTERVAL = 2.0


class AmapRegeoClient:
  def __init__(self):
    self._lock = threading.Lock()
    self._api_key = ""
    self._road_name = ""
    self._in_flight = False
    self._last_fetch_time = 0.0
    self._last_query_key: tuple[float, float] | None = None
    self._pending: tuple[float, float] | None = None

  def set_api_key(self, api_key: str) -> None:
    self._api_key = api_key or ""

  @property
  def road_name(self) -> str:
    with self._lock:
      return self._road_name

  def clear(self) -> None:
    with self._lock:
      self._road_name = ""
      self._last_query_key = None
      self._pending = None

  def update(self, lat: float, lng: float) -> None:
    if not self._api_key:
      return

    grid_lat = round(lat / CACHE_GRID) * CACHE_GRID
    grid_lng = round(lng / CACHE_GRID) * CACHE_GRID
    query_key = (grid_lat, grid_lng)
    if query_key == self._last_query_key:
      return

    now = time.monotonic()
    with self._lock:
      if self._in_flight:
        self._pending = (lat, lng)
        return
      if now - self._last_fetch_time < MIN_FETCH_INTERVAL:
        return
      self._in_flight = True
      self._last_fetch_time = now

    threading.Thread(target=self._fetch_thread, args=(lat, lng, query_key), daemon=True).start()

  def _fetch_thread(self, lat: float, lng: float, query_key: tuple[float, float]) -> None:
    road_name = ""
    try:
      response = requests.get(
        REGEO_URL,
        params={
          "key": self._api_key,
          "location": f"{lng:.6f},{lat:.6f}",
          "extensions": "all",
          "radius": 50,
          "roadlevel": 0,
        },
        timeout=REQUEST_TIMEOUT,
      )
      if response.status_code == 200:
        road_name = self._parse_road_name(response.json())
    except Exception:
      pass

    pending: tuple[float, float] | None = None
    with self._lock:
      self._in_flight = False
      if road_name:
        self._road_name = road_name
        self._last_query_key = query_key
      pending = self._pending
      self._pending = None

    if pending is not None:
      self.update(pending[0], pending[1])

  @staticmethod
  def _parse_road_name(data: dict) -> str:
    if data.get("status") != "1":
      return ""

    regeocode = data.get("regeocode") or {}
    roads = regeocode.get("roads") or []
    for road in roads:
      name = road.get("name") or ""
      if name:
        return name

    addr = regeocode.get("addressComponent") or {}
    street_number = addr.get("streetNumber") or {}
    street = street_number.get("street") or ""
    if street:
      number = street_number.get("number") or ""
      return f"{street}{number}" if number else street

    township = addr.get("township") or ""
    if township:
      return township

    formatted = regeocode.get("formatted_address") or ""
    return formatted
