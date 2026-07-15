"""
UI-side Amap viewer: blit shared-memory frames produced by amapd.

Heavy work (tile download, stitch, rotate, crop) runs in a separate
lowest-priority process at 2Hz. This widget only uploads/blits RGBA and
draws the GPS marker — no OpenGL FBO stitch.
"""

from __future__ import annotations

import numpy as np
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from bluepilot.mapd.amap_ipc import AmapFrameShm, MAX_H, MAX_W


class AmapView(Widget):
  def __init__(self):
    super().__init__()
    self._shm = AmapFrameShm(create=False)
    if not self._shm.available:
      # Ensure shm exists so requests reach amapd even before first publish.
      self._shm = AmapFrameShm(create=True)

    self._texture: rl.Texture | None = None
    self._tex_w = 0
    self._tex_h = 0
    self._last_seq = -1
    self._ready = False
    self._gps_valid = False
    self._bearing = 0.0
    self._road_name = ""
    self._enable = False

  def prepare(self) -> None:
    """Ask amapd to keep producing frames; poll ready flag."""
    self._enable = True
    # Use a modest default request until first render supplies panel size.
    h = self._shm.read_header()
    req_w = h.request_w if h and h.request_w else 640
    req_h = h.request_h if h and h.request_h else 720
    self._shm.update_request(req_w, req_h, enable=True)
    self._poll_header()

  def set_enabled(self, enabled: bool) -> None:
    self._enable = enabled
    h = self._shm.read_header()
    req_w = h.request_w if h and h.request_w else 640
    req_h = h.request_h if h and h.request_h else 720
    self._shm.update_request(req_w, req_h, enable=enabled)

  def is_ready(self) -> bool:
    self._poll_header()
    return self._ready and self._gps_valid

  def _poll_header(self) -> None:
    h = self._shm.read_header()
    if h is None:
      self._ready = False
      return
    self._ready = bool(h.ready)
    self._gps_valid = bool(h.gps_valid)
    self._bearing = float(h.bearing)
    self._road_name = h.road_name

  def _ensure_texture(self, width: int, height: int) -> None:
    if self._texture is not None and self._tex_w == width and self._tex_h == height:
      return
    if self._texture is not None:
      rl.unload_texture(self._texture)
      self._texture = None
    # Allocate empty GPU texture; updated via update_texture.
    image = rl.Image(None, width, height, 1, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
    self._texture = rl.load_texture_from_image(image)
    self._tex_w = width
    self._tex_h = height

  def _sync_texture_from_shm(self) -> None:
    h = self._shm.read_header()
    if h is None or h.width <= 0 or h.height <= 0:
      return
    if h.seq == self._last_seq:
      return

    seq1 = h.seq
    active = h.active
    width, height = h.width, h.height
    mv = self._shm.read_frame_bytes(active, width, height)
    if mv is None:
      return
    # Tear-check
    h2 = self._shm.read_header()
    if h2 is None or h2.seq != seq1:
      return

    self._ensure_texture(width, height)
    arr = np.frombuffer(mv, dtype=np.uint8).reshape((height, width, 4))
    cont = np.ascontiguousarray(arr)
    if self._texture is not None:
      rl.update_texture(self._texture, rl.ffi.cast("void *", cont.ctypes.data))
    self._last_seq = seq1
    self._ready = bool(h2.ready)
    self._gps_valid = bool(h2.gps_valid)
    self._bearing = float(h2.bearing)
    self._road_name = h2.road_name

  def _render(self, rect: rl.Rectangle):
    view_w = max(64, min(int(rect.width), MAX_W))
    view_h = max(64, min(int(rect.height), MAX_H))
    self._enable = True
    self._shm.update_request(view_w, view_h, enable=True)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(30, 30, 30, 255))

    self._sync_texture_from_shm()
    self._poll_header()

    if not self._gps_valid:
      self._draw_placeholder(rect, "No GPS")
      rl.end_scissor_mode()
      return

    if self._texture is not None:
      src = rl.Rectangle(0, 0, float(self._tex_w), float(self._tex_h))
      dst = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)
      rl.draw_texture_pro(self._texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
    else:
      self._draw_placeholder(rect, "Loading map...")

    self._draw_gps_marker(rect)
    rl.end_scissor_mode()

  def _draw_gps_marker(self, rect: rl.Rectangle):
    cx = rect.x + rect.width / 2.0
    cy = rect.y + rect.height / 2.0

    arrow_size = 54
    half = arrow_size / 2.0
    tip = rl.Vector2(cx, cy - arrow_size)
    left = rl.Vector2(cx - half, cy + half)
    right = rl.Vector2(cx + half, cy + half)
    rl.draw_triangle(tip, left, right, rl.Color(220, 40, 40, 240))
    rl.draw_triangle_lines(tip, left, right, rl.Color(255, 255, 255, 180))
    rl.draw_circle(int(cx), int(cy + half - 9), 12, rl.Color(220, 40, 40, 240))

    if self._road_name:
      font_size = 22
      font = gui_app.font(FontWeight.SEMI_BOLD)
      text_size = measure_text_cached(font, self._road_name, font_size)
      tx = cx - text_size.x / 2.0
      ty = cy + half + 12
      bg_pad = 6
      rl.draw_rectangle_rounded(
        rl.Rectangle(tx - bg_pad, ty - bg_pad, text_size.x + 2 * bg_pad, text_size.y + 2 * bg_pad),
        0.3, 8, rl.Color(0, 0, 0, 160),
      )
      rl.draw_text_ex(font, self._road_name, rl.Vector2(tx, ty), font_size, 0, rl.WHITE)

  def _draw_placeholder(self, rect: rl.Rectangle, message: str):
    font_size = 28
    font = gui_app.font(FontWeight.NORMAL)
    text_size = measure_text_cached(font, message, font_size)
    tx = rect.x + (rect.width - text_size.x) / 2.0
    ty = rect.y + rect.height / 2.0 - text_size.y / 2.0
    rl.draw_text_ex(font, message, rl.Vector2(tx, ty), font_size, 0, rl.Color(150, 150, 150, 255))
