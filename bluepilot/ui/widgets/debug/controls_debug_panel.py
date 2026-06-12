"""
BluePilot Controls Debug Panel - Main Container
Slide-in overlay with tabbed sub-panels for lateral, longitudinal, and vehicle debug data.
Port of Qt OnroadControlsDebugPanel.
"""

import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget
from bluepilot.ui.widgets.debug.debug_colors import DebugColors
from bluepilot.ui.widgets.debug.lateral_debug_panel import LateralDebugPanel
from bluepilot.ui.widgets.debug.long_debug_panel import LongDebugPanel
from bluepilot.ui.widgets.debug.other_debug_panel import OtherDebugPanel


class ControlsDebugPanel(Widget):
  """Main onroad debug panel container with slide-in animation and tab navigation."""

  # Layout constants
  TAB_BAR_HEIGHT = 100
  CLOSE_BUTTON_SIZE = 60
  CLOSE_BUTTON_MARGIN = 15
  ANIMATION_SPEED = 5.0  # Progress units per second (~200ms to full open)

  # Tab definitions
  TAB_LABELS = ["Lateral", "Longitudinal", "Other"]

  def __init__(self):
    super().__init__()
    self._visible_state = False
    self._animation_progress = 0.0  # 0 = fully hidden, 1 = fully visible
    self._current_tab = 0

    # Sub-panels
    self._lateral_panel = LateralDebugPanel()
    self._long_panel = LongDebugPanel()
    self._other_panel = OtherDebugPanel()
    self._panels = [self._lateral_panel, self._long_panel, self._other_panel]

    # Fonts
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)

    # Track if close button or tab was clicked this frame (to avoid passthrough)
    self._consumed_click = False

  def toggle_visibility(self):
    """Toggle the debug panel open/closed."""
    self._visible_state = not self._visible_state

  @property
  def is_panel_visible(self) -> bool:
    """Returns True if the panel is visible or animating."""
    return self._visible_state or self._animation_progress > 0.01

  def _update_state(self):
    # Animate slide-in/out
    target = 1.0 if self._visible_state else 0.0
    dt = rl.get_frame_time()
    if dt <= 0:
      dt = 1.0 / 60.0  # Fallback

    if self._animation_progress < target:
      self._animation_progress = min(target, self._animation_progress + self.ANIMATION_SPEED * dt)
    elif self._animation_progress > target:
      self._animation_progress = max(target, self._animation_progress - self.ANIMATION_SPEED * dt)

    # Only update the active sub-panel when visible
    if self._animation_progress > 0.5:
      self._panels[self._current_tab]._update_state()

  def _render(self, rect: rl.Rectangle):
    if self._animation_progress < 0.01:
      return

    self._consumed_click = False

    # Calculate panel position (slides in from right)
    panel_w = rect.width
    x_offset = panel_w * (1.0 - self._animation_progress)
    panel_rect = rl.Rectangle(rect.x + x_offset, rect.y, panel_w, rect.height)

    # Clip to the available area
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    # Dark background
    rl.draw_rectangle_rec(panel_rect, DebugColors.PANEL_BG)

    # Content area (above tab bar)
    content_rect = rl.Rectangle(
      panel_rect.x, panel_rect.y,
      panel_rect.width, panel_rect.height - self.TAB_BAR_HEIGHT
    )

    # Render the active sub-panel
    if self._animation_progress > 0.5:
      self._panels[self._current_tab].render(content_rect)

    # Tab bar at bottom (includes close button on left)
    self._render_tab_bar(panel_rect)

    rl.end_scissor_mode()

    # Consume all mouse events within panel rect to prevent passthrough
    self._consume_mouse_events(panel_rect)

  def _render_tab_bar(self, panel_rect: rl.Rectangle):
    """Render bottom tab bar with close button on left and 3 tab buttons."""
    bar_y = panel_rect.y + panel_rect.height - self.TAB_BAR_HEIGHT
    bar_rect = rl.Rectangle(panel_rect.x, bar_y, panel_rect.width, self.TAB_BAR_HEIGHT)

    # Tab bar background (slightly darker)
    rl.draw_rectangle_rec(bar_rect, DebugColors.PANEL_BG_DARKER)

    # Separator line above tab bar
    rl.draw_line(int(bar_rect.x), int(bar_y),
                 int(bar_rect.x + bar_rect.width), int(bar_y),
                 DebugColors.TAB_BORDER)

    tab_count = len(self.TAB_LABELS)
    tab_spacing = 12
    tab_margin = 15
    tab_h = 64
    tab_y = bar_y + (self.TAB_BAR_HEIGHT - tab_h) / 2

    # Close button on left
    close_w = self.CLOSE_BUTTON_SIZE
    close_x = panel_rect.x + tab_margin
    close_rect = rl.Rectangle(close_x, tab_y, close_w, tab_h)

    rl.draw_rectangle_rounded(close_rect, 0.3, 8, DebugColors.CLOSE_BG)
    rl.draw_rectangle_rounded_lines_ex(close_rect, 0.3, 8, 1.5, DebugColors.CLOSE_BORDER)

    close_text = "X"
    close_text_size = measure_text_cached(self._font_bold, close_text, 28)
    rl.draw_text_ex(self._font_bold, close_text,
                    rl.Vector2(close_x + (close_w - close_text_size.x) / 2,
                               tab_y + (tab_h - close_text_size.y) / 2),
                    28, 0, DebugColors.LEGEND_TEXT)

    for mouse_event in gui_app.mouse_events:
      if mouse_event.left_released:
        if rl.check_collision_point_rec(mouse_event.pos, close_rect):
          self._visible_state = False
          self._consumed_click = True

    # Tab buttons to the right of close button
    tabs_start_x = close_x + close_w + tab_spacing
    available_w = (panel_rect.x + panel_rect.width - tab_margin) - tabs_start_x - (tab_count - 1) * tab_spacing
    tab_w = available_w / tab_count

    for i, label in enumerate(self.TAB_LABELS):
      tab_x = tabs_start_x + i * (tab_w + tab_spacing)
      tab_rect = rl.Rectangle(tab_x, tab_y, tab_w, tab_h)

      is_active = (i == self._current_tab)
      bg_color = DebugColors.TAB_ACTIVE if is_active else DebugColors.TAB_INACTIVE
      text_color = DebugColors.TAB_TEXT if is_active else DebugColors.TAB_TEXT_DIM

      rl.draw_rectangle_rounded(tab_rect, 0.3, 8, bg_color)
      if not is_active:
        rl.draw_rectangle_rounded_lines_ex(tab_rect, 0.3, 8, 1.0, DebugColors.TAB_BORDER)

      # Center text
      text_size = measure_text_cached(self._font_semi, label, 26)
      text_x = tab_x + (tab_w - text_size.x) / 2
      text_y_pos = tab_y + (tab_h - text_size.y) / 2
      rl.draw_text_ex(self._font_semi, label,
                      rl.Vector2(text_x, text_y_pos), 26, 0, text_color)

      # Click detection
      for mouse_event in gui_app.mouse_events:
        if mouse_event.left_released:
          if rl.check_collision_point_rec(mouse_event.pos, tab_rect):
            self._current_tab = i
            self._consumed_click = True

  def _consume_mouse_events(self, panel_rect: rl.Rectangle):
    """Prevent mouse events from passing through the debug panel to the onroad view."""
    # The Widget base class handles this via is_pressed tracking.
    # We set our rect so the parent knows we're consuming events in this area.
    if self._animation_progress > 0.5:
      self._rect = panel_rect
