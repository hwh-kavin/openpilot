import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyray as rl
import requests

from openpilot.selfdrive.ui.sunnypilot.onroad.coord_transform import lat_lng_to_tile
from openpilot.system.hardware.hw import Paths

TILE_SIZE = 256
MAP_STYLE = 6
MAX_GPU_TILES = 96
MAX_CONCURRENT = 4
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3

AMAP_TILE_ROOT = Path(Paths.mapd_root()) / "amap_tiles"


@dataclass(frozen=True)
class TileKey:
  style: int
  z: int
  x: int
  y: int


@dataclass
class TileImage:
  key: TileKey
  image: Any


def dir_size_bytes(root: Path) -> int:
  total = 0
  if not root.exists():
    return 0
  stack = [root]
  while stack:
    current = stack.pop()
    try:
      for entry in os.scandir(current):
        if entry.is_file():
          total += entry.stat().st_size
        elif entry.is_dir():
          stack.append(Path(entry.path))
    except OSError:
      pass
  return total


def clear_disk_cache() -> None:
  if AMAP_TILE_ROOT.exists():
    shutil.rmtree(AMAP_TILE_ROOT)


class AmapTileCache:
  def __init__(self):
    self._gpu: OrderedDict[TileKey, Any] = OrderedDict()
    self._in_flight: set[TileKey] = set()
    self._ready_queue: list[TileImage] = []
    self._lock = threading.Lock()
    self._server_idx = 0
    self._api_key = ""

  def set_api_key(self, api_key: str) -> None:
    self._api_key = api_key or ""

  def clear_gpu_cache(self) -> None:
    for texture in self._gpu.values():
      rl.unload_texture(texture)
    self._gpu.clear()
    with self._lock:
      self._in_flight.clear()
      self._ready_queue.clear()

  def clear_all(self) -> None:
    self.clear_gpu_cache()
    clear_disk_cache()

  @staticmethod
  def disk_path(key: TileKey) -> Path:
    return AMAP_TILE_ROOT / str(key.style) / str(key.z) / str(key.x) / f"{key.y}.png"

  @staticmethod
  def tile_url(key: TileKey, api_key: str, server: int) -> str:
    return (
      f"https://webrd0{server}.is.autonavi.com/appmaptile"
      f"?lang=zh_cn&size=1&scale=1&style={key.style}&x={key.x}&y={key.y}&z={key.z}&key={api_key}"
    )

  def has_gpu_tile(self, key: TileKey) -> bool:
    return key in self._gpu

  def get_texture(self, key: TileKey) -> Any | None:
    texture = self._gpu.get(key)
    if texture is not None:
      self._gpu.move_to_end(key)
    return texture

  def request_tile(self, key: TileKey) -> None:
    if key in self._gpu:
      return
    with self._lock:
      if key in self._in_flight:
        return
      if len(self._in_flight) >= MAX_CONCURRENT:
        return
      self._in_flight.add(key)
    threading.Thread(target=self._load_tile_thread, args=(key,), daemon=True).start()

  def _load_tile_thread(self, key: TileKey) -> None:
    png_bytes: bytes | None = None
    disk_path = self.disk_path(key)

    if disk_path.exists():
      try:
        png_bytes = disk_path.read_bytes()
      except OSError:
        png_bytes = None

    if png_bytes is None and self._api_key:
      png_bytes = self._download_tile(key)

    if png_bytes:
      try:
        image = rl.load_image_from_memory(".png", png_bytes, len(png_bytes))
        if image.data:
          with self._lock:
            self._ready_queue.append(TileImage(key=key, image=image))
          return
      except Exception:
        pass
      if disk_path.exists():
        try:
          disk_path.unlink()
        except OSError:
          pass

    with self._lock:
      self._in_flight.discard(key)

  def _download_tile(self, key: TileKey) -> bytes | None:
    for attempt in range(MAX_RETRIES):
      self._server_idx = (self._server_idx % 4) + 1
      url = self.tile_url(key, self._api_key, self._server_idx)
      try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200 or not response.content:
          continue
        disk_path = self.disk_path(key)
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(response.content)
        return response.content
      except Exception:
        if attempt + 1 < MAX_RETRIES:
          continue
    return None

  def process_ready_queue(self) -> None:
    with self._lock:
      pending = self._ready_queue
      self._ready_queue = []

    for item in pending:
      with self._lock:
        self._in_flight.discard(item.key)
      if item.key in self._gpu:
        rl.unload_image(item.image)
        continue
      while len(self._gpu) >= MAX_GPU_TILES:
        _, old_texture = self._gpu.popitem(last=False)
        rl.unload_texture(old_texture)
      texture = rl.load_texture_from_image(item.image)
      rl.unload_image(item.image)
      self._gpu[item.key] = texture

  def schedule_tiles(self, keys: list[TileKey]) -> None:
    for key in keys:
      if key in self._gpu:
        continue
      with self._lock:
        if key in self._in_flight:
          continue
        if len(self._in_flight) >= MAX_CONCURRENT:
          break
      self.request_tile(key)


def core_tile_keys(lat: float, lng: float, zoom: int, style: int = MAP_STYLE) -> list[TileKey]:
  cx, cy = lat_lng_to_tile(lat, lng, zoom)
  keys: list[TileKey] = []
  n = 2 ** zoom
  for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
      x = max(0, min(cx + dx, n - 1))
      y = max(0, min(cy + dy, n - 1))
      keys.append(TileKey(style=style, z=zoom, x=x, y=y))
  return keys
