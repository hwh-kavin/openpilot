"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui.elements import (
  UiElement, RelDistElement, RelSpeedElement, SteeringAngleElement,
  DesiredLateralAccelElement, ActualLateralAccelElement, DesiredSteeringAngleElement,
  DesiredSteeringPIDElement, CpuUsageElement, CpuTempElement, MemoryUsageElement, FreeSpaceElement,
  ModelTorqueElement, ModelAccelElement,
)
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


def get_bottom_dev_ui_offset():
  if ui_state.developer_ui in (DeveloperUiState.BOTTOM, DeveloperUiState.BOTH):
    return 60
  return 0


class DeveloperUiState(IntEnum):
  OFF = 0
  BOTTOM = 1
  RIGHT = 2
  BOTH = 3


class DeveloperUiRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self.dev_ui_mode = DeveloperUiState.OFF

    self.rel_dist_elem = RelDistElement()
    self.rel_speed_elem = RelSpeedElement()
    self.steering_angle_elem = SteeringAngleElement()
    self.desired_lat_accel_elem = DesiredLateralAccelElement()
    self.actual_lat_accel_elem = ActualLateralAccelElement()
    self.desired_steer_elem = DesiredSteeringAngleElement()
    self.desired_pid_steer_elem = DesiredSteeringPIDElement()
    self.cpu_usage_elem = CpuUsageElement()
    self.cpu_temp_elem = CpuTempElement()
    self.memory_usage_elem = MemoryUsageElement()
    self.free_space_elem = FreeSpaceElement()
    self.model_torque_elem = ModelTorqueElement()
    self.model_accel_elem = ModelAccelElement()

  def _update_state(self) -> None:
    self.dev_ui_mode = ui_state.developer_ui

  def render_bottom(self, rect: rl.Rectangle) -> None:
    self._update_state()
    if self.dev_ui_mode not in (DeveloperUiState.BOTTOM, DeveloperUiState.BOTH):
      return

    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      return

    self._draw_bottom_dev_ui(rect)

  def render_right(self, rect: rl.Rectangle) -> None:
    self._update_state()
    if self.dev_ui_mode not in (DeveloperUiState.RIGHT, DeveloperUiState.BOTH):
      return

    sm = ui_state.sm
    if sm.recv_frame["carState"] < ui_state.started_frame:
      return

    self._draw_right_dev_ui(rect)

  def _render(self, rect: rl.Rectangle) -> None:
    self.render_right(rect)

  def _draw_right_dev_ui(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    controls_state = sm['controlsState']

    UI_BORDER_SIZE = 20
    container_width = 184
    x = int(rect.x + rect.width - container_width - UI_BORDER_SIZE * 2)
    y = int(rect.y + UI_BORDER_SIZE * 1.5)

    elements = [
      self.rel_dist_elem.update(sm, ui_state.is_metric),
      self.rel_speed_elem.update(sm, ui_state.is_metric),
      self.steering_angle_elem.update(sm, ui_state.is_metric),
    ]
    if controls_state.lateralControlState.which() == 'torqueState':
      elements.append(self.desired_lat_accel_elem.update(sm, ui_state.is_metric))
    elif controls_state.lateralControlState.which() == 'angleState':
      elements.append(self.desired_steer_elem.update(sm, ui_state.is_metric))
    elif controls_state.lateralControlState.which() == 'pidState':
      elements.append(self.desired_pid_steer_elem.update(sm, ui_state.is_metric))

    elements.append(self.actual_lat_accel_elem.update(sm, ui_state.is_metric))

    current_y = y
    for element in elements:
      current_y += self._draw_right_dev_ui_element(x, current_y, element)

  def _draw_right_dev_ui_element(self, x: int, y: int, element: UiElement) -> int:
    x += 0
    y += 230
    container_width = 184
    label_size = 28
    value_size = 60
    unit_size = 28
    label_width = measure_text_cached(self._font_bold, element.label, label_size, 0).x
    centered_label_x = x + (container_width - label_width) / 2
    rl.draw_text_ex(self._font_bold, element.label, rl.Vector2(centered_label_x, y), label_size, 0, rl.WHITE)

    y += 45
    value_width = measure_text_cached(self._font_bold, element.value, value_size, 0).x
    centered_value_x = x + (container_width - value_width) / 2
    rl.draw_text_ex(self._font_bold, element.value, rl.Vector2(centered_value_x, y), value_size, 0, element.color)

    if element.unit:
      units_height = measure_text_cached(self._font_bold, element.unit, unit_size, 0).x

      units_x = x + container_width
      units_y = y + (value_size / 2) + (units_height / 2)

      rl.draw_text_pro(self._font_bold, element.unit, rl.Vector2(units_x, units_y), rl.Vector2(0, 0), -90.0, unit_size, 0, rl.WHITE)

    return 130

  def _draw_bottom_dev_ui(self, rect: rl.Rectangle) -> None:
    sm = ui_state.sm
    bar_height = 61
    y = int(rect.y + rect.height - bar_height)

    rl.draw_rectangle(int(rect.x), y, int(rect.width), bar_height,
                      rl.Color(0, 0, 0, 100))

    elements = [
      self.model_torque_elem.update(sm, ui_state.is_metric),
      self.cpu_usage_elem.update(sm, ui_state.is_metric),
      self.cpu_temp_elem.update(sm, ui_state.is_metric),
      self.memory_usage_elem.update(sm, ui_state.is_metric),
      self.free_space_elem.update(sm, ui_state.is_metric),
      self.model_accel_elem.update(sm, ui_state.is_metric),
    ]

    if not elements:
      return

    font_size = 38
    side_pad = 20
    element_widths = []
    for element in elements:
      element.measure(self._font_bold, font_size)
      element_widths.append(element.total_width)

    center_y = y + bar_height // 2
    n = len(elements)

    # First flush-left, last flush-right; middle items evenly spaced between them
    positions = [0.0] * n
    positions[0] = rect.x + side_pad
    positions[-1] = rect.x + rect.width - side_pad - element_widths[-1]

    if n > 2:
      middle_width = sum(element_widths[1:-1])
      available = positions[-1] - (positions[0] + element_widths[0]) - middle_width
      gap = available / (n - 1)
      current_x = positions[0] + element_widths[0] + gap
      for i in range(1, n - 1):
        positions[i] = current_x
        current_x += element_widths[i] + gap

    for i, element in enumerate(elements):
      element_center_x = int(positions[i] + element_widths[i] / 2)
      self._draw_bottom_dev_ui_element(element_center_x, center_y, element)

  def _draw_bottom_dev_ui_element(self, center_x: int, y: int, element: UiElement) -> None:
    font_size = 38
    start_x = center_x - element.total_width / 2

    rl.draw_text_ex(self._font_bold, element.label_text, rl.Vector2(start_x, y - font_size // 2), font_size, 0, rl.WHITE)
    rl.draw_text_ex(self._font_bold, element.val_text, rl.Vector2(start_x + element.label_width, y - font_size // 2), font_size, 0, element.color)

    if element.unit:
      rl.draw_text_ex(self._font_bold, element.unit_text, rl.Vector2(start_x + element.label_width + element.val_width, y - font_size // 2),
                      font_size, 0, rl.WHITE)
