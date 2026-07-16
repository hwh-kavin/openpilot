"""
UI-side Amap viewer: blit shared-memory frames produced by amapd.

Heavy work (tile download, stitch, rotate, crop) runs in a separate
lowest-priority process at 2Hz. This widget only uploads/blits RGBA and
draws the GPS marker — no OpenGL FBO stitch.

Top-right Destination / Home / Work shortcuts navigate using
NavSavedDestination / NavHome / NavWork.
"""

from __future__ import annotations

import time

import numpy as np
import pyray as rl

from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from bluepilot.mapd.amap_ipc import AmapFrameShm, MAX_H, MAX_W
from bluepilot.mapd import nav_params as navp

BTN_SIZE = 144
BTN_GAP = 32
BTN_MARGIN = 20

# shortcut slot → (param key, short label, toast label, rgba)
_SHORTCUTS = (
  ("destination", "NavSavedDestination", "目", "目的地", (230, 81, 0, 230)),
  ("home", "NavHome", "家", "家", (46, 125, 50, 230)),
  ("work", "NavWork", "公", "公司", (21, 101, 192, 230)),
)

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
    self._params = Params()
    self._dest_rect = rl.Rectangle(0, 0, BTN_SIZE, BTN_SIZE)
    self._home_rect = rl.Rectangle(0, 0, BTN_SIZE, BTN_SIZE)
    self._work_rect = rl.Rectangle(0, 0, BTN_SIZE, BTN_SIZE)
    self._toast = ""
    self._toast_until = 0.0
    self._pending_route_check_until = 0.0

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

  def _update_shortcut_rects(self, rect: rl.Rectangle) -> None:
    x = rect.x + rect.width - BTN_MARGIN - BTN_SIZE
    y = rect.y + BTN_MARGIN
    step = BTN_SIZE + BTN_GAP
    self._dest_rect = rl.Rectangle(x, y, BTN_SIZE, BTN_SIZE)
    self._home_rect = rl.Rectangle(x, y + step, BTN_SIZE, BTN_SIZE)
    self._work_rect = rl.Rectangle(x, y + 2 * step, BTN_SIZE, BTN_SIZE)

  def _shortcut_rect(self, slot: str) -> rl.Rectangle:
    if slot == "destination":
      return self._dest_rect
    if slot == "home":
      return self._home_rect
    return self._work_rect

  @staticmethod
  def _point_in_rect(pos: MousePos, r: rl.Rectangle) -> bool:
    return r.x <= pos.x <= r.x + r.width and r.y <= pos.y <= r.y + r.height

  def _show_toast(self, message: str, seconds: float = 2.0) -> None:
    self._toast = message
    self._toast_until = time.monotonic() + seconds

  def _navigate_to_slot(self, slot: str) -> None:
    meta = next((s for s in _SHORTCUTS if s[0] == slot), None)
    if meta is None:
      return
    _, key, _, toast_label, _ = meta
    place = navp.get_place(self._params, key)
    # Unconfigured shortcuts are gray and must do nothing when tapped.
    if not place:
      return
    if not navp.get_web_service_key(self._params):
      self._show_toast("请配置高德Web服务Key", seconds=3.0)
      return
    # Destination shortcut uses NavSavedDestination (portal "设为目的地").
    navp.set_route_error(self._params, None)
    navp.set_place(self._params, "NavDestination", place)
    navp.set_route_geometry(self._params, None)
    label = place.get("place_name") or toast_label
    self._show_toast(f"导航至{label}")
    self._pending_route_check_until = time.monotonic() + 8.0

  def _poll_route_error(self) -> None:
    until = getattr(self, "_pending_route_check_until", 0.0)
    if until <= 0 or time.monotonic() > until:
      return
    err = navp.get_route_error(self._params)
    if err:
      self._show_toast(err, seconds=3.5)
      self._pending_route_check_until = 0.0
      return
    route = navp.get_route_geometry(self._params)
    if route and route.get("coordinates"):
      self._pending_route_check_until = 0.0

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    for slot, key, *_ in _SHORTCUTS:
      if self._point_in_rect(mouse_pos, self._shortcut_rect(slot)):
        # Always consume the tap over the icon (even when gray / unset).
        if navp.get_place(self._params, key):
          self._navigate_to_slot(slot)
        return
    super()._handle_mouse_release(mouse_pos)

  def _render(self, rect: rl.Rectangle):
    view_w = max(64, min(int(rect.width), MAX_W))
    view_h = max(64, min(int(rect.height), MAX_H))
    self._enable = True
    self._shm.update_request(view_w, view_h, enable=True)
    self._update_shortcut_rects(rect)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(30, 30, 30, 255))

    self._sync_texture_from_shm()
    self._poll_header()

    if not self._gps_valid:
      self._draw_placeholder(rect, "No GPS")
      self._draw_guidance(rect)
      self._draw_shortcuts(rect)
      self._poll_route_error()
      self._draw_toast(rect)
      rl.end_scissor_mode()
      return

    if self._texture is not None:
      src = rl.Rectangle(0, 0, float(self._tex_w), float(self._tex_h))
      dst = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)
      rl.draw_texture_pro(self._texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
    else:
      self._draw_placeholder(rect, "Loading map...")

    self._draw_gps_marker(rect)
    self._draw_guidance(rect)
    self._draw_shortcuts(rect)
    self._poll_route_error()
    self._draw_toast(rect)
    rl.end_scissor_mode()

  def _draw_guidance(self, rect: rl.Rectangle) -> None:
    g = navp.get_guidance(self._params)
    if not g:
      return
    icon = g.get("icon") or ""
    action = g.get("action") or ""
    dist = g.get("distance_text") or ""
    road = g.get("road") or ""
    if not action and not dist:
      return

    icon_char = {
      "left": "←",
      "right": "→",
      "slight_left": "↖",
      "slight_right": "↗",
      "uturn": "↩",
      "arrive": "⚑",
    }.get(icon, "•")

    # Banner at top-left, clear of the right-side shortcut buttons
    margin = 16
    banner_h = 110
    max_w = max(200.0, rect.width - BTN_MARGIN - BTN_SIZE - margin * 3)
    banner = rl.Rectangle(rect.x + margin, rect.y + margin, max_w, banner_h)
    bg = rl.Color(20, 20, 20, 210)
    if g.get("off_route"):
      bg = rl.Color(80, 40, 0, 220)
    rl.draw_rectangle_rounded(banner, 0.2, 8, bg)
    rl.draw_rectangle_rounded_lines(banner, 0.2, 8, 2, rl.Color(255, 255, 255, 100))

    # Icon circle
    cx = banner.x + 48
    cy = banner.y + banner_h / 2.0
    accent = rl.Color(33, 150, 243, 255)
    if icon in ("left", "slight_left"):
      accent = rl.Color(76, 175, 80, 255)
    elif icon in ("right", "slight_right"):
      accent = rl.Color(255, 152, 0, 255)
    elif icon == "arrive":
      accent = rl.Color(244, 67, 54, 255)
    rl.draw_circle(int(cx), int(cy), 34, accent)

    font_icon = gui_app.font(FontWeight.BOLD)
    icon_size = 40
    isz = measure_text_cached(font_icon, icon_char, icon_size)
    rl.draw_text_ex(font_icon, icon_char, rl.Vector2(cx - isz.x / 2.0, cy - isz.y / 2.0), icon_size, 0, rl.WHITE)

    font = gui_app.font(FontWeight.SEMI_BOLD)
    text_x = banner.x + 100
    # Distance
    dist_size = 36
    rl.draw_text_ex(font, dist or "--", rl.Vector2(text_x, banner.y + 18), dist_size, 0, rl.WHITE)
    # Action + road
    line2 = action
    if road:
      line2 = f"{action} · {road}" if action else road
    if len(line2) > 18:
      line2 = line2[:17] + "…"
    rl.draw_text_ex(font, line2, rl.Vector2(text_x, banner.y + 62), 28, 0, rl.Color(220, 220, 220, 255))

  def _draw_shortcuts(self, _rect: rl.Rectangle) -> None:
    for slot, key, short_label, _, rgba in _SHORTCUTS:
      place = navp.get_place(self._params, key)
      self._draw_shortcut_btn(
        self._shortcut_rect(slot),
        short_label,
        bool(place),
        rl.Color(*rgba),
      )

  def _draw_shortcut_btn(self, r: rl.Rectangle, label: str, configured: bool, color: rl.Color) -> None:
    if configured:
      bg = color
      border = rl.Color(255, 255, 255, 180)
      text_color = rl.WHITE
    else:
      # Grayed-out / disabled look when address is not set
      bg = rl.Color(55, 55, 55, 180)
      border = rl.Color(120, 120, 120, 120)
      text_color = rl.Color(140, 140, 140, 180)
    rl.draw_rectangle_rounded(r, 0.35, 8, bg)
    rl.draw_rectangle_rounded_lines_ex(r, 0.35, 8, 2, border)
    font = gui_app.font(FontWeight.SEMI_BOLD)
    font_size = 56
    text_size = measure_text_cached(font, label, font_size)
    tx = r.x + (r.width - text_size.x) / 2.0
    ty = r.y + (r.height - text_size.y) / 2.0
    rl.draw_text_ex(font, label, rl.Vector2(tx, ty), font_size, 0, text_color)

  def _draw_toast(self, rect: rl.Rectangle) -> None:
    if not self._toast or time.monotonic() > self._toast_until:
      self._toast = ""
      return
    font_size = 24
    font = gui_app.font(FontWeight.NORMAL)
    text_size = measure_text_cached(font, self._toast, font_size)
    pad = 12
    tw = text_size.x + 2 * pad
    th = text_size.y + 2 * pad
    tx = rect.x + (rect.width - tw) / 2.0
    ty = rect.y + rect.height - th - 24
    rl.draw_rectangle_rounded(rl.Rectangle(tx, ty, tw, th), 0.3, 8, rl.Color(0, 0, 0, 180))
    rl.draw_text_ex(font, self._toast, rl.Vector2(tx + pad, ty + pad), font_size, 0, rl.WHITE)

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
