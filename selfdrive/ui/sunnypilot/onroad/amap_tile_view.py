import math

import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.ui.sunnypilot.onroad.amap_regeo import AmapRegeoClient
from openpilot.selfdrive.ui.sunnypilot.onroad.amap_tile_cache import (
  AMAP_TILE_ROOT,
  AmapTileCache,
  MAP_STYLE,
  TILE_SIZE,
  TileKey,
  clear_disk_cache,
  core_tile_keys,
  dir_size_bytes,
)
from openpilot.selfdrive.ui.sunnypilot.onroad.coord_transform import lat_lng_to_tile, wgs84_to_gcj02
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui.elements import GpsInfoElement
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

_active_instance: "AmapTileView | None" = None


def get_amap_tile_view() -> "AmapTileView":
  global _active_instance
  if _active_instance is None:
    _active_instance = AmapTileView()
  return _active_instance


def _rotate_point(px: float, py: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
  rad = math.radians(angle_deg)
  dx, dy = px - cx, py - cy
  cos_a, sin_a = math.cos(rad), math.sin(rad)
  return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


class AmapTileView(Widget):
  def __init__(self):
    super().__init__()
    global _active_instance
    _active_instance = self

    self._params = Params()
    self._cache = AmapTileCache()
    self._regeo = AmapRegeoClient()
    self._zoom = 20
    self._lat = 0.0
    self._lng = 0.0
    self._bearing_deg = 0.0
    self._gps_valid = False
    self._core_keys: list[TileKey] = []
    self._render_rect: rl.Rectangle | None = None
    self._prev_lat = 0.0
    self._prev_lng = 0.0
    self._has_prev_position = False
    self._font_demi = gui_app.font(FontWeight.SEMI_BOLD)

  @staticmethod
  def cache_size_bytes() -> int:
    return dir_size_bytes(AMAP_TILE_ROOT)

  @staticmethod
  def clear_disk_cache() -> None:
    clear_disk_cache()
    if _active_instance is not None:
      _active_instance._cache.clear_gpu_cache()
      _active_instance._core_keys = []

  def _update_gps(self) -> None:
    gps_data, valid = GpsInfoElement.get_gps_data(ui_state.sm)
    if not valid or gps_data is None:
      self._gps_valid = False
      return

    self._gps_valid = True
    self._lat, self._lng = wgs84_to_gcj02(gps_data.latitude, gps_data.longitude)
    self._update_bearing(gps_data)

    v_ego = ui_state.sm['carState'].vEgo if ui_state.sm.valid['carState'] else 0.0
    if v_ego < 5.0:
      self._zoom = 21
    elif v_ego < 15.0:
      self._zoom = 20
    else:
      self._zoom = 19

  def _update_bearing(self, gps_data) -> None:
    if gps_data.bearingAccuracyDeg != 180.0:
      self._bearing_deg = gps_data.bearingDeg
    elif self._has_prev_position:
      dx = self._lng - self._prev_lng
      dy = self._lat - self._prev_lat
      if abs(dx) > 1e-7 or abs(dy) > 1e-7:
        self._bearing_deg = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    self._prev_lat = self._lat
    self._prev_lng = self._lng
    self._has_prev_position = True

  def _panel_rect(self) -> rl.Rectangle:
    if self._render_rect is not None:
      return self._render_rect
    return rl.Rectangle(0, 0, gui_app.width * 0.5, gui_app.height)

  def _tile_radius(self, rect: rl.Rectangle) -> tuple[int, int]:
    cover = math.sqrt(rect.width * rect.width + rect.height * rect.height) / 2.0
    radius = int(math.ceil(cover / TILE_SIZE)) + 2
    return radius, radius

  def _tile_keys_for_panel(self) -> list[TileKey]:
    if not self._gps_valid:
      return []
    rect = self._panel_rect()
    cx, cy = lat_lng_to_tile(self._lat, self._lng, self._zoom)
    radius_x, radius_y = self._tile_radius(rect)
    n = 2 ** self._zoom
    keys: list[TileKey] = []
    for dx in range(-radius_x, radius_x + 1):
      for dy in range(-radius_y, radius_y + 1):
        x = cx + dx
        y = cy + dy
        if 0 <= x < n and 0 <= y < n:
          keys.append(TileKey(style=MAP_STYLE, z=self._zoom, x=x, y=y))
    return keys

  def _core_keys_for_position(self) -> list[TileKey]:
    if not self._gps_valid:
      return []
    return core_tile_keys(self._lat, self._lng, self._zoom, MAP_STYLE)

  def tick(self) -> None:
    api_key = self._params.get("AmapApiKey") or ""
    self._cache.set_api_key(api_key)
    self._regeo.set_api_key(api_key)
    self._update_gps()
    self._cache.process_ready_queue()

    if not api_key or not self._gps_valid:
      self._core_keys = []
      return

    self._core_keys = self._core_keys_for_position()
    self._cache.schedule_tiles(self._tile_keys_for_panel())
    self._regeo.update(self._lat, self._lng)

  def is_ready(self) -> bool:
    if not self._core_keys:
      return False
    return all(self._cache.has_gpu_tile(key) for key in self._core_keys)

  def can_enter_map_split(self) -> bool:
    if not self._params.get("AmapApiKey"):
      return False
    if not self._gps_valid:
      return False
    return self.is_ready()

  def _render(self, rect: rl.Rectangle):
    self._render_rect = rect

    if not self._params.get("AmapApiKey"):
      rl.draw_rectangle_rec(rect, rl.Color(30, 30, 30, 255))
      self._draw_center_text(rect, "Set Amap API key in OSM settings")
      return

    if not self._gps_valid:
      rl.draw_rectangle_rec(rect, rl.Color(30, 30, 30, 255))
      self._draw_center_text(rect, "Waiting for GPS...")
      return

    cx, cy = lat_lng_to_tile(self._lat, self._lng, self._zoom)

    pixel_x = (self._lng + 180.0) / 360.0 * (2 ** self._zoom) * TILE_SIZE
    lat_rad = math.radians(self._lat)
    pixel_y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** self._zoom) * TILE_SIZE

    offset_x = rect.x + rect.width / 2 - (pixel_x - cx * TILE_SIZE)
    offset_y = rect.y + rect.height / 2 - (pixel_y - cy * TILE_SIZE)

    scx = rect.x + rect.width / 2
    scy = rect.y + rect.height / 2
    rotation = -self._bearing_deg

    n = 2 ** self._zoom
    radius_x, radius_y = self._tile_radius(rect)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    for dx in range(-radius_x, radius_x + 1):
      for dy in range(-radius_y, radius_y + 1):
        x = cx + dx
        y = cy + dy
        if x < 0 or y < 0 or x >= n or y >= n:
          continue
        key = TileKey(style=MAP_STYLE, z=self._zoom, x=x, y=y)
        texture = self._cache.get_texture(key)
        if texture is None:
          continue

        tx = offset_x + dx * TILE_SIZE
        ty = offset_y + dy * TILE_SIZE
        tile_cx, tile_cy = tx + TILE_SIZE / 2, ty + TILE_SIZE / 2
        rot_cx, rot_cy = _rotate_point(tile_cx, tile_cy, scx, scy, rotation)
        dest = rl.Rectangle(rot_cx - TILE_SIZE / 2, rot_cy - TILE_SIZE / 2, TILE_SIZE, TILE_SIZE)
        rl.draw_texture_pro(
          texture,
          rl.Rectangle(0, 0, TILE_SIZE, TILE_SIZE),
          dest,
          rl.Vector2(TILE_SIZE / 2, TILE_SIZE / 2),
          rotation,
          rl.WHITE,
        )

    rl.end_scissor_mode()
    self._draw_location_marker(rect, self._regeo.road_name)

  def _draw_location_marker(self, rect: rl.Rectangle, road_name: str) -> None:
    cx = rect.x + rect.width / 2
    cy = rect.y + rect.height / 2
    size = 18
    tip = rl.Vector2(cx, cy - size)
    left = rl.Vector2(cx - size * 0.55, cy + size * 0.45)
    right = rl.Vector2(cx + size * 0.55, cy + size * 0.45)
    rl.draw_triangle(tip, left, right, rl.RED)
    rl.draw_triangle_lines(tip, left, right, rl.WHITE)

    if not road_name:
      return

    font_size = 34
    text = road_name
    text_size = measure_text_cached(self._font_demi, text, font_size)
    max_width = rect.width - 40
    if text_size.x > max_width:
      while text_size.x > max_width and len(text) > 3:
        text = text[:-1]
        text_size = measure_text_cached(self._font_demi, text + "...", font_size)
      text = text + "..."

    text_size = measure_text_cached(self._font_demi, text, font_size)
    padding_x, padding_y = 16, 8
    label_width = text_size.x + padding_x * 2
    label_height = text_size.y + padding_y * 2
    label_y = cy + size + 10
    label_rect = rl.Rectangle(cx - label_width / 2, label_y, label_width, label_height)
    rl.draw_rectangle_rounded(label_rect, 0.25, 8, rl.Color(0, 0, 0, 170))
    text_pos = rl.Vector2(label_rect.x + padding_x, label_rect.y + padding_y)
    rl.draw_text_ex(self._font_demi, text, text_pos, font_size, 0, rl.WHITE)

  def _draw_center_text(self, rect: rl.Rectangle, text: str) -> None:
    font = gui_app.font(FontWeight.NORMAL)
    size = 36
    text_size = rl.measure_text_ex(font, text, size, 1.0)
    pos = rl.Vector2(rect.x + (rect.width - text_size.x) / 2, rect.y + (rect.height - text_size.y) / 2)
    rl.draw_text_ex(font, text, pos, size, 1.0, rl.LIGHTGRAY)
