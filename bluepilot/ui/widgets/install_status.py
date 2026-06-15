"""Install/deploy progress display for on-device Raylib UI."""
from __future__ import annotations

import time

import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.wrap_text import wrap_text

try:
  from bluepilot.backend.core import install_status as status_module
except ImportError:
  status_module = None

POLL_INTERVAL = 2.0
BANNER_HEIGHT = 92


class InstallStatusTracker:
  """Polls shared install status file and caches the latest state."""

  def __init__(self) -> None:
    self._last_poll = 0.0
    self.active = False
    self.message = ""
    self.progress: int | None = None
    self.status = ""
    self.phase = ""

  def update(self) -> bool:
    now = time.monotonic()
    if now - self._last_poll < POLL_INTERVAL:
      return self.active
    self._last_poll = now

    if status_module is None:
      self.active = False
      return False

    data = status_module.read_status()
    if not data:
      self.active = False
      return False

    self.status = str(data.get("status", ""))
    self.phase = str(data.get("phase", ""))
    self.message = str(data.get("message", ""))
    progress = data.get("progress")
    self.progress = int(progress) if isinstance(progress, (int, float)) else None
    self.active = self.status in ("installing", "restarting")
    return self.active


def _title_for(tracker: InstallStatusTracker) -> str:
  if tracker.status == "restarting":
    return "Restarting..."
  if tracker.phase == "bootstrap":
    return "Setting Up Device"
  if tracker.phase == "portal":
    return "Updating Portal"
  return "Installing..."


def draw_install_banner(rect: rl.Rectangle, tracker: InstallStatusTracker) -> None:
  """Draw install banner along the top edge of rect."""
  if not tracker.active:
    return

  banner = rl.Rectangle(rect.x, rect.y, rect.width, BANNER_HEIGHT)
  bg = rl.Color(70, 91, 234, 235)
  if tracker.status == "restarting":
    bg = rl.Color(255, 195, 0, 235)
  rl.draw_rectangle_rec(banner, bg)

  title_font = gui_app.font(FontWeight.SEMI_BOLD)
  body_font = gui_app.font(FontWeight.NORMAL)
  rl.draw_text_ex(title_font, _title_for(tracker), rl.Vector2(rect.x + 20, rect.y + 10), 30, 0, rl.WHITE)

  message = tracker.message or "Please wait. Do not power off the device."
  wrapped = wrap_text(body_font, message, 24, int(max(120, rect.width - 40)))
  if wrapped:
    rl.draw_text_ex(body_font, wrapped[0], rl.Vector2(rect.x + 20, rect.y + 44), 24, 0, rl.Color(255, 255, 255, 210))
    if len(wrapped) > 1:
      rl.draw_text_ex(body_font, wrapped[1], rl.Vector2(rect.x + 20, rect.y + 68), 22, 0, rl.Color(255, 255, 255, 180))

  if tracker.progress is not None:
    bar = rl.Rectangle(rect.x + 20, rect.y + BANNER_HEIGHT - 8, rect.width - 40, 5)
    rl.draw_rectangle_rec(bar, rl.Color(0, 0, 0, 90))
    fill_width = max(0.0, min(bar.width, bar.width * tracker.progress / 100.0))
    if fill_width > 0:
      fill = rl.Rectangle(bar.x, bar.y, fill_width, bar.height)
      rl.draw_rectangle_rec(fill, rl.WHITE)


def banner_height(tracker: InstallStatusTracker) -> float:
  return BANNER_HEIGHT if tracker.active else 0.0
