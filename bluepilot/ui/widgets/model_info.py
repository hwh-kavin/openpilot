"""
BluePilot Model Info Widget
Displays the current active driving model name on the offroad home screen.
Clickable to open the Models settings panel. Ported from the Qt ModelInfoWidget.
"""

import re
import pyray as rl
from collections.abc import Callable

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight, MousePos
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from bluepilot.ui.lib.colors import BPColors


# Card styling (matching DriveStats and Qt styling)
CARD_BG = rl.Color(38, 38, 38, 255)
CARD_BORDER = rl.Color(255, 255, 255, 26)
NAME_CONTAINER_BG = rl.Color(255, 255, 255, 13)
NAME_CONTAINER_BORDER = rl.Color(255, 255, 255, 26)
MODEL_NAME_COLOR = BPColors.ACCENT  # #18b4ff

# Regex to insert newline before date patterns like "(October 03, 2023)"
DATE_PATTERN = re.compile(r'\s+(\([A-Za-z]+\s+\d{1,2},\s+\d{4}\))')

DEFAULT_MODEL_NAME = "Default Model"


class ModelInfoWidget(Widget):
  def __init__(self):
    super().__init__()
    self._model_name = DEFAULT_MODEL_NAME
    self._click_callback: Callable[[], None] | None = None

  def set_click_callback(self, callback: Callable[[], None]):
    self._click_callback = callback

  def _update_state(self):
    self._model_name = self._get_active_model_name()

  def _get_active_model_name(self) -> str:
    try:
      model_manager = ui_state.sm["modelManagerSP"]
      if model_manager.activeBundle.ref:
        return model_manager.activeBundle.displayName
    except Exception:
      pass
    return DEFAULT_MODEL_NAME

  def _handle_mouse_release(self, mouse_pos: MousePos):
    super()._handle_mouse_release(mouse_pos)
    if self._click_callback and rl.check_collision_point_rec(mouse_pos, self._rect):
      self._click_callback()

  def _render(self, rect: rl.Rectangle):
    padding_x = 20
    padding_top = 20

    # Draw card background
    rl.draw_rectangle_rounded(rect, 0.05, 10, CARD_BG)
    rl.draw_rectangle_rounded_lines(rect, 0.05, 10, CARD_BORDER)

    # Title (centered)
    title_font = gui_app.font(FontWeight.BOLD)
    title_size = 48
    title_text = "Driving Model"
    title_text_size = measure_text_cached(title_font, title_text, title_size)
    title_x = rect.x + (rect.width - title_text_size.x) / 2
    title_pos = rl.Vector2(title_x, rect.y + padding_top)
    rl.draw_text_ex(title_font, title_text, title_pos, title_size, 0, rl.WHITE)

    # Model name container
    container_top = rect.y + padding_top + title_size + 10
    container_margin = 10
    container_height = rect.y + rect.height - container_top - container_margin
    container_rect = rl.Rectangle(
      rect.x + container_margin, container_top,
      rect.width - 2 * container_margin, container_height
    )
    rl.draw_rectangle_rounded(container_rect, 0.08, 10, NAME_CONTAINER_BG)
    rl.draw_rectangle_rounded_lines(container_rect, 0.08, 10, NAME_CONTAINER_BORDER)

    # Model name text (each line centered, with date pattern splitting and auto-scaling)
    display_name = DATE_PATTERN.sub(r'\n\1', self._model_name)
    lines = display_name.split('\n')

    name_font = gui_app.font(FontWeight.SEMI_BOLD)
    font_size = 46
    available_width = container_rect.width - 40  # padding inside container

    # Auto-shrink font to fit widest line
    for line in lines:
      line_size = measure_text_cached(name_font, line.strip(), font_size)
      while line_size.x > available_width and font_size > 28:
        font_size -= 2
        line_size = measure_text_cached(name_font, line.strip(), font_size)

    # Measure line height and compute total block height
    line_spacing = int(font_size * 0.3)
    sample_size = measure_text_cached(name_font, "Ag", font_size)
    line_height = sample_size.y
    total_height = line_height * len(lines) + line_spacing * (len(lines) - 1)

    # Draw each line individually centered
    start_y = container_rect.y + (container_rect.height - total_height) / 2
    for i, line in enumerate(lines):
      line_text = line.strip()
      line_size = measure_text_cached(name_font, line_text, font_size)
      line_x = container_rect.x + (container_rect.width - line_size.x) / 2
      line_y = start_y + i * (line_height + line_spacing)
      rl.draw_text_ex(name_font, line_text, rl.Vector2(line_x, line_y), font_size, 0, MODEL_NAME_COLOR)
