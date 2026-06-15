"""
Shared helpers and widgets for BluePilot Portal QR access.
"""
import ipaddress
import time

import numpy as np
import pyray as rl
import qrcode

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import IconButton

try:
  from bluepilot.backend.config import DEFAULT_PORT as DEFAULT_PORTAL_PORT
except ImportError:
  DEFAULT_PORTAL_PORT = 80

try:
  from bluepilot.ui.widgets.install_status import InstallStatusTracker
except ImportError:
  InstallStatusTracker = None


class PortalQrMixin:
  QR_REFRESH_INTERVAL = 5  # seconds

  def _init_portal_qr_state(self) -> None:
    self._params = Params()
    self._portal_url = ""
    self._qr_texture: rl.Texture | None = None
    self._last_refresh = float("-inf")
    self._generate_qr_code()
    self._last_refresh = time.monotonic()

  def _get_portal_port(self) -> int:
    try:
      from bluepilot.backend.network.utils import get_portal_port
      return get_portal_port(self._params)
    except Exception:
      cloudlog.exception("Failed to read BPPortalPort, using default portal port")
      return DEFAULT_PORTAL_PORT

  def _get_wifi_ip(self) -> str | None:
    try:
      from bluepilot.backend.network.utils import get_wifi_ip
      return get_wifi_ip()
    except Exception:
      cloudlog.exception("Failed to get WiFi IP for portal QR")

    import subprocess
    try:
      result = subprocess.run(["ip", "addr", "show", "wlan0"], capture_output=True, text=True, timeout=2)
      for line in result.stdout.split("\n"):
        if "inet " in line:
          return line.strip().split()[1].split("/")[0]
    except Exception:
      pass
    return None

  def _normalize_host(self, wifi_ip: str) -> str:
    host = wifi_ip.strip()
    if host.startswith("http://"):
      host = host[7:]
    elif host.startswith("https://"):
      host = host[8:]
    if "/" in host:
      host = host.split("/", 1)[0]

    # Strip any existing port suffix (IPv4 only).
    try:
      ipaddress.IPv4Address(host.split(":", 1)[0])
      if ":" in host:
        base, _, maybe_port = host.rpartition(":")
        if maybe_port.isdigit():
          host = base
    except ValueError:
      pass
    return host

  def _format_portal_url(self, host: str, port: int) -> str:
    try:
      ipaddress.IPv4Address(host)
      if port == 80:
        return f"http://{host}/"
      return f"http://{host}:{port}/"
    except ValueError:
      if port == 80:
        return f"http://[{host}]/"
      return f"http://[{host}]:{port}/"

  def _get_portal_url(self) -> str:
    wifi_ip = self._get_wifi_ip()
    if not wifi_ip:
      return ""
    return self._format_portal_url(self._normalize_host(wifi_ip), self._get_portal_port())

  def regenerate_qr_code(self) -> None:
    self._generate_qr_code()
    self._last_refresh = time.monotonic()

  def _generate_qr_code(self) -> None:
    self._portal_url = self._get_portal_url()
    if not self._portal_url:
      if self._qr_texture and self._qr_texture.id != 0:
        rl.unload_texture(self._qr_texture)
      self._qr_texture = None
      return

    try:
      qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
      qr.add_data(self._portal_url)
      qr.make(fit=True)

      pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
      img_array = np.array(pil_img, dtype=np.uint8)

      if self._qr_texture and self._qr_texture.id != 0:
        rl.unload_texture(self._qr_texture)

      rl_image = rl.Image()
      rl_image.data = rl.ffi.cast("void *", img_array.ctypes.data)
      rl_image.width = pil_img.width
      rl_image.height = pil_img.height
      rl_image.mipmaps = 1
      rl_image.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8

      self._qr_texture = rl.load_texture_from_image(rl_image)
    except Exception:
      cloudlog.exception("Portal QR code generation failed")
      self._qr_texture = None

  def _check_qr_refresh(self) -> None:
    current_time = time.monotonic()
    if current_time - self._last_refresh >= self.QR_REFRESH_INTERVAL:
      self._generate_qr_code()
      self._last_refresh = current_time

  def _render_qr_texture(self, rect: rl.Rectangle) -> None:
    if not self._qr_texture:
      rl.draw_rectangle_rounded(rect, 0.1, 20, rl.Color(40, 40, 40, 255))
      error_font = gui_app.font(FontWeight.BOLD)
      message = tr("WiFi not connected") if not self._portal_url else tr("QR Code Error")
      rl.draw_text_ex(error_font, message, rl.Vector2(rect.x + 20, rect.y + rect.height // 2 - 15), 30, 0.0, rl.RED)
      return

    source = rl.Rectangle(0, 0, self._qr_texture.width, self._qr_texture.height)
    rl.draw_texture_pro(self._qr_texture, source, rect, rl.Vector2(0, 0), 0, rl.WHITE)

  def _unload_qr_texture(self) -> None:
    if self._qr_texture and self._qr_texture.id != 0:
      rl.unload_texture(self._qr_texture)


class PortalQrPanel(PortalQrMixin, Widget):
  """Inline Portal QR code for settings pages."""

  def __init__(self):
    super().__init__()
    self._init_portal_qr_state()
    self._install_status = InstallStatusTracker() if InstallStatusTracker else None
    self._padding = 20
    self._was_visible = False

  def show_event(self):
    super().show_event()
    self.regenerate_qr_code()

  def set_parent_rect(self, parent_rect: rl.Rectangle) -> None:
    super().set_parent_rect(parent_rect)
    self._rect.width = parent_rect.width
    self._rect.height = 420

  def _render(self, rect: rl.Rectangle) -> int:
    if not self._was_visible:
      self.regenerate_qr_code()
    self._was_visible = True
    self._check_qr_refresh()
    if self._install_status:
      self._install_status.update()

    content_x = rect.x + self._padding
    content_width = rect.width - self._padding * 2
    y = rect.y + self._padding

    if self._install_status and self._install_status.active:
      notice_font = gui_app.font(FontWeight.BOLD)
      notice = self._install_status.message or tr("Installing portal components. Please wait...")
      notice_wrapped = wrap_text(notice_font, notice, 34, int(content_width))
      rl.draw_text_ex(notice_font, "\n".join(notice_wrapped[:2]), rl.Vector2(content_x, y), 34, 0.0, rl.Color(255, 195, 0, 255))
      y += len(notice_wrapped[:2]) * 34 + 16
      if self._install_status.progress is not None:
        bar = rl.Rectangle(content_x, y, content_width, 8)
        rl.draw_rectangle_rounded(bar, 1, 8, rl.Color(40, 40, 40, 255))
        fill = rl.Rectangle(bar.x, bar.y, bar.width * self._install_status.progress / 100.0, bar.height)
        rl.draw_rectangle_rounded(fill, 1, 8, rl.Color(70, 91, 234, 255))
        y += 24

    title_font = gui_app.font(FontWeight.NORMAL)
    title = tr("Scan to open BluePilot Portal")
    title_wrapped = wrap_text(title_font, title, 42, int(content_width))
    rl.draw_text_ex(title_font, "\n".join(title_wrapped), rl.Vector2(content_x, y), 42, 0.0, rl.WHITE)
    y += len(title_wrapped) * 42 + 20

    qr_size = min(260, content_width - 40)
    qr_x = rect.x + (rect.width - qr_size) / 2
    qr_rect = rl.Rectangle(qr_x, y, qr_size, qr_size)
    self._render_qr_texture(qr_rect)
    y += qr_size + 20

    if self._portal_url:
      url_font = gui_app.font(FontWeight.BOLD)
      url_wrapped = wrap_text(url_font, self._portal_url, 34, int(content_width))
      rl.draw_text_ex(url_font, "\n".join(url_wrapped), rl.Vector2(content_x, y), 34, 0.0, rl.Color(170, 170, 170, 255))
    else:
      hint_font = gui_app.font(FontWeight.NORMAL)
      hint = tr("Connect your comma device to WiFi first")
      hint_wrapped = wrap_text(hint_font, hint, 34, int(content_width))
      rl.draw_text_ex(hint_font, "\n".join(hint_wrapped), rl.Vector2(content_x, y), 34, 0.0, rl.Color(170, 170, 170, 255))

    return -1

  def __del__(self):
    self._unload_qr_texture()


class PortalQrDialog(PortalQrMixin, Widget):
  """Full-screen dialog showing a QR code for the local BluePilot Portal URL."""

  def __init__(self):
    super().__init__()
    self._init_portal_qr_state()
    self._close_btn = IconButton(gui_app.texture("icons/close.png", 80, 80))
    self._close_btn.set_click_callback(gui_app.pop_widget)

  def _render(self, rect: rl.Rectangle) -> int:
    rl.clear_background(rl.Color(224, 224, 224, 255))
    self._check_qr_refresh()

    margin = 70
    content_rect = rl.Rectangle(rect.x + margin, rect.y + margin, rect.width - 2 * margin, rect.height - 2 * margin)
    y = content_rect.y

    close_size = 80
    pad = 20
    close_rect = rl.Rectangle(content_rect.x - pad, y - pad, close_size + pad * 2, close_size + pad * 2)
    self._close_btn.render(close_rect)
    y += close_size + 40

    title_font = gui_app.font(FontWeight.NORMAL)
    left_width = int(content_rect.width * 0.5 - 15)
    title = tr("Scan to open BluePilot Portal")
    title_wrapped = wrap_text(title_font, title, 75, left_width)
    rl.draw_text_ex(title_font, "\n".join(title_wrapped), rl.Vector2(content_rect.x, y), 75, 0.0, rl.BLACK)
    y += len(title_wrapped) * 75 + 40

    remaining_height = content_rect.height - (y - content_rect.y)
    right_width = content_rect.width // 2 - 20
    self._render_instructions(rl.Rectangle(content_rect.x, y, left_width, remaining_height))

    qr_size = min(right_width, content_rect.height) - 120
    qr_x = content_rect.x + left_width + 40 + (right_width - qr_size) // 2
    qr_y = content_rect.y
    qr_rect = rl.Rectangle(qr_x, qr_y, qr_size, qr_size)
    self._render_qr_texture(qr_rect)

    if self._portal_url:
      url_font = gui_app.font(FontWeight.BOLD)
      url_y = qr_rect.y + qr_rect.height + 20
      url_wrapped = wrap_text(url_font, self._portal_url, 34, int(right_width))
      rl.draw_text_ex(url_font, "\n".join(url_wrapped), rl.Vector2(qr_x, url_y), 34, 0.0, rl.BLACK)

    return -1

  def _render_instructions(self, rect: rl.Rectangle) -> None:
    from openpilot.system.ui.lib.text_measure import measure_text_cached

    if self._portal_url:
      instructions = [
        tr("Connect your phone to the same WiFi network as your comma device"),
        tr("Scan the QR code on the right with your phone camera"),
        tr("Or open this address in your browser:"),
        self._portal_url,
      ]
    else:
      instructions = [
        tr("Connect your comma device to WiFi first"),
        tr("Then return here to scan the QR code for BluePilot Portal"),
      ]

    font = gui_app.font(FontWeight.BOLD)
    y = rect.y
    for i, text in enumerate(instructions):
      circle_radius = 25
      circle_x = rect.x + circle_radius + 15
      text_x = rect.x + circle_radius * 2 + 40
      text_width = rect.width - (circle_radius * 2 + 40)

      wrapped = wrap_text(font, text, 47, int(text_width))
      text_height = len(wrapped) * 47
      circle_y = y + text_height // 2

      if i < 3:
        rl.draw_circle(int(circle_x), int(circle_y), circle_radius, rl.Color(70, 70, 70, 255))
        number = str(i + 1)
        number_size = measure_text_cached(font, number, 30)
        rl.draw_text_ex(font, number, (int(circle_x - number_size.x // 2), int(circle_y - number_size.y // 2)), 30, 0, rl.WHITE)

      rl.draw_text_ex(font, "\n".join(wrapped), rl.Vector2(text_x, y), 47, 0.0, rl.BLACK)
      y += text_height + 50

  def __del__(self):
    self._unload_qr_texture()
