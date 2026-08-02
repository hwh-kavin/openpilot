"""
UI-side CarLife map viewer: blit shared-memory frames from carlifed.

Map assist fields (lanes / turn / path / speedAction / …) stay on cereal for
future lateral/longitudinal use — not drawn here; the phone map stream already
shows them. Phone screenshots are already oriented — no rotation or crop.
"""

from __future__ import annotations

import json

import numpy as np
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from bluepilot.mapd.carlife_ipc import CarLifeFrameShm

# Keep local — avoid coupling UI import to carlife_ipc constants (crash-loop risk).
_STATUS_PATH = "/dev/shm/bluepilot_carlife_status.json"


def _local_ipv4() -> str:
  """Best-effort WLAN IPv4 for the waiting placeholder (no internet needed)."""
  import fcntl
  import socket
  import struct

  # Prefer real iface addresses — UI may start before default route exists.
  for ifname in ("wlan0", "wlan1", "eth0", "enp0s3", "en0"):
    try:
      s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      try:
        ip = socket.inet_ntoa(fcntl.ioctl(
          s.fileno(),
          0x8915,  # SIOCGIFADDR
          struct.pack("256s", ifname[:15].encode("utf-8")),
        )[20:24])
      finally:
        s.close()
      if ip and not ip.startswith("127."):
        return ip
    except Exception:
      continue

  try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
      s.connect(("8.8.8.8", 80))
      ip = s.getsockname()[0]
    finally:
      s.close()
    if ip and not ip.startswith("127."):
      return ip
  except Exception:
    pass

  try:
    import subprocess
    out = subprocess.check_output(
      ["ip", "-4", "-o", "addr", "show", "scope", "global"],
      text=True, timeout=1,
    )
    for line in out.splitlines():
      parts = line.split()
      if "inet" in parts:
        idx = parts.index("inet")
        cand = parts[idx + 1].split("/")[0]
        if cand and not cand.startswith("127."):
          return cand
  except Exception:
    pass
  return "?.?.?.?"


def _subnet_broadcast(ip: str) -> str:
  """Assume /24 for hint text when we only know the host address."""
  try:
    a, b, c, _ = ip.split(".")
    return f"{a}.{b}.{c}.255"
  except Exception:
    return "x.x.x.255"


class CarLifeMapView(Widget):
  def __init__(self):
    super().__init__()
    self._shm = CarLifeFrameShm(create=False)
    if not self._shm.available:
      self._shm = CarLifeFrameShm(create=True)

    self._texture: rl.Texture | None = None
    self._tex_w = 0
    self._tex_h = 0
    self._last_seq = -1
    self._ready = False
    self._enable = False
    self._hint_ip = _local_ipv4()
    self._hint_refresh_at = 0.0
    self._rx_data = 0
    self._rx_video = 0
    self._rx_frames = 0
    self._rx_from = ""
    self._status_refresh_at = 0.0

  def prepare(self) -> None:
    self._enable = True
    self._shm.update_enable(True)
    self._poll_header()

  def set_enabled(self, enabled: bool) -> None:
    self._enable = enabled
    self._shm.update_enable(enabled)

  def is_ready(self) -> bool:
    """True once carlifed has published at least one decoded map frame (no GPS gate)."""
    self._poll_header()
    return self._ready

  def frame_size(self) -> tuple[int, int] | None:
    """Phone frame pixel size for split-pane layout (None if not ready)."""
    h = self._shm.read_header()
    if h is None or not h.ready or h.width <= 0 or h.height <= 0 or int(h.seq) <= 0:
      return None
    return int(h.width), int(h.height)

  def preferred_pane_size(self, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest pane that fits in max_w×max_h with frame aspect (no crop/stretch)."""
    frame = self.frame_size()
    fw, fh = frame if frame is not None else (9, 16)
    if fw <= 0 or fh <= 0 or max_w <= 0 or max_h <= 0:
      return max(1, max_w // 2), max(1, max_h)
    scale = min(max_w / fw, max_h / fh)
    return max(1, int(fw * scale)), max(1, int(fh * scale))

  def _poll_header(self) -> None:
    h = self._shm.read_header()
    if h is None:
      self._ready = False
      return
    self._ready = bool(h.ready) and h.width > 0 and h.height > 0 and int(h.seq) > 0

  def _ensure_texture(self, width: int, height: int) -> None:
    if self._texture is not None and self._tex_w == width and self._tex_h == height:
      return
    if self._texture is not None:
      rl.unload_texture(self._texture)
      self._texture = None
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
    h2 = self._shm.read_header()
    if h2 is None or h2.seq != seq1:
      return

    self._ensure_texture(width, height)
    arr = np.frombuffer(mv, dtype=np.uint8).reshape((height, width, 4))
    cont = np.ascontiguousarray(arr)
    if self._texture is not None:
      rl.update_texture(self._texture, rl.ffi.cast("void *", cont.ctypes.data))
    self._last_seq = seq1
    self._ready = bool(h2.ready) and width > 0 and height > 0

  def _render(self, rect: rl.Rectangle):
    try:
      self._render_inner(rect)
    except Exception:
      try:
        rl.end_scissor_mode()
      except Exception:
        pass
      rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(30, 30, 30, 255))
      self._draw_placeholder(rect, "Map render error")

  def _render_inner(self, rect: rl.Rectangle):
    self._enable = True
    self._shm.update_enable(True)

    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    rl.draw_rectangle(int(rect.x), int(rect.y), int(rect.width), int(rect.height), rl.Color(30, 30, 30, 255))

    self._sync_texture_from_shm()
    self._poll_header()

    if self._texture is not None and self._ready:
      self._draw_frame_contain(rect)
    else:
      self._draw_waiting(rect)

    rl.end_scissor_mode()

  def _draw_frame_contain(self, rect: rl.Rectangle) -> None:
    """Draw frame keeping aspect. Layout usually sizes the pane to match; no stretch."""
    if self._texture is None or self._tex_w <= 0 or self._tex_h <= 0:
      return
    if rect.width <= 0 or rect.height <= 0:
      return
    scale = min(rect.width / float(self._tex_w), rect.height / float(self._tex_h))
    dw = float(self._tex_w) * scale
    dh = float(self._tex_h) * scale
    # Flush to the left of the map pane so it joins the driving video with no gap.
    dx = rect.x
    dy = rect.y + (rect.height - dh) * 0.5
    src = rl.Rectangle(0, 0, float(self._tex_w), float(self._tex_h))
    dst = rl.Rectangle(dx, dy, dw, dh)
    rl.draw_texture_pro(self._texture, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)

  def displayed_size(self, max_w: float, max_h: float) -> tuple[float, float]:
    """Ready: height-fit keep aspect. Waiting: half width × full height (50/50)."""
    if max_w <= 0 or max_h <= 0:
      return 1.0, 1.0
    frame = self.frame_size()
    if frame is None:
      return max_w * 0.5, max_h
    fw, fh = float(frame[0]), float(frame[1])
    if fw <= 0 or fh <= 0:
      return max_w * 0.5, max_h
    # Prefer full height; width from aspect (caller may still clamp max_w).
    map_w = max_h * (fw / fh)
    if map_w > max_w:
      map_w = max_w
      map_h = map_w * (fh / fw)
    else:
      map_h = max_h
    return map_w, map_h

  def _refresh_rx_status(self) -> None:
    try:
      with open(_STATUS_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
      self._rx_data = int(st.get("data_pkts", 0))
      self._rx_video = int(st.get("video_pkts", 0))
      self._rx_frames = int(st.get("video_frames", 0))
      addr = st.get("last_video_addr") or st.get("last_data_addr") or ""
      self._rx_from = str(addr)
    except Exception:
      pass

  def _draw_waiting(self, rect: rl.Rectangle) -> None:
    import time
    now = time.monotonic()
    if now >= self._hint_refresh_at or self._hint_ip.startswith("?"):
      self._hint_ip = _local_ipv4()
      self._hint_refresh_at = now + 2.0
    if now >= self._status_refresh_at:
      self._refresh_rx_status()
      self._status_refresh_at = now + 1.0

    bcast = _subnet_broadcast(self._hint_ip)
    if self._rx_data == 0 and self._rx_video == 0:
      hint = "未收到UDP — 请单播到车机IP(非仅广播)"
    elif self._rx_frames == 0:
      hint = f"已收包 数据{self._rx_data} 视频{self._rx_video} — 等待完整帧"
    else:
      hint = f"已解码 {self._rx_frames} 帧 from {self._rx_from or '-'}"
    lines = (
      "等待手机地图...",
      f"车机  {self._hint_ip}:8888 / :8889",
      f"广播  {bcast}:8888 / :8889",
      hint,
    )
    font = gui_app.font(FontWeight.NORMAL)
    # Doubled vs previous (28/24/22/20) for readability on the waiting half-pane.
    sizes = (56, 48, 44, 40)
    gap = 16.0
    total_h = sum(sizes) + gap * (len(lines) - 1)
    y = rect.y + (rect.height - total_h) / 2.0
    for text, size in zip(lines, sizes, strict=True):
      sz = measure_text_cached(font, text, size)
      color = rl.Color(220, 220, 220, 255) if size >= 56 else rl.Color(150, 150, 150, 255)
      rl.draw_text_ex(
        font, text,
        rl.Vector2(rect.x + (rect.width - sz.x) / 2.0, y),
        size, 0, color,
      )
      y += size + gap

  def _draw_placeholder(self, rect: rl.Rectangle, message: str):
    font_size = 28
    font = gui_app.font(FontWeight.NORMAL)
    text_size = measure_text_cached(font, message, font_size)
    tx = rect.x + (rect.width - text_size.x) / 2.0
    ty = rect.y + rect.height / 2.0 - text_size.y / 2.0
    rl.draw_text_ex(font, message, rl.Vector2(tx, ty), font_size, 0, rl.Color(150, 150, 150, 255))
