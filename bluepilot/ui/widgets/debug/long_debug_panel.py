"""
BluePilot Longitudinal Debug Panel
Displays acceleration and gas/brake control signal time-series graphs.
Port of Qt LongDebugPanel + AccelGraphWidget + ControlGraphWidget.
"""

import time
import pyray as rl
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.ui_state import ui_state
from bluepilot.ui.widgets.debug.debug_colors import DebugColors
from bluepilot.ui.widgets.debug.debug_graph import TimeSeriesGraph, GraphConfig, GraphSeries

# Match Qt update rate: 20Hz = 50ms between data pushes
DATA_PUSH_INTERVAL = 0.05


class LongDebugPanel(Widget):
  """Longitudinal control debug panel with acceleration and control signal graphs."""

  MAX_DATA_POINTS = 100

  def __init__(self):
    super().__init__()

    # Acceleration graph (centered, auto-scaling)
    self._accel_graph = TimeSeriesGraph(
      config=GraphConfig(
        title="Acceleration",
        y_unit=" m/s\u00b2",
        max_data_points=self.MAX_DATA_POINTS,
        min_scale=1.0,
      ),
      series=[
        GraphSeries("Desired", DebugColors.DESIRED_GREEN, fill_alpha=20),
        GraphSeries("Actual", DebugColors.ACTUAL_YELLOW, fill_alpha=20),
      ]
    )

    # Control signals graph (0-1 range, zero at bottom)
    self._control_graph = TimeSeriesGraph(
      config=GraphConfig(
        title="Control Signals",
        y_unit="",
        max_data_points=self.MAX_DATA_POINTS,
        min_scale=1.0,
        auto_scale=False,
        zero_at_bottom=True,
      ),
      series=[
        GraphSeries("Gas", DebugColors.GAS_GREEN, fill_alpha=30),
        GraphSeries("Brake", DebugColors.BRAKE_RED, fill_alpha=30),
      ]
    )

    self._long_actuator_delay = 0.0
    self._should_stop = False
    self._allow_throttle = True
    self._allow_brake = True
    self._last_push_time = 0.0

  def _update_state(self):
    sm = ui_state.sm
    if sm is None:
      return

    # Throttle data pushes to ~20Hz to match Qt version
    now = time.monotonic()
    if now - self._last_push_time < DATA_PUSH_INTERVAL:
      return

    actual_accel = 0.0
    desired_accel = 0.0
    gas_signal = 0.0
    brake_signal = 0.0

    try:
      if sm.valid.get('carState', False):
        actual_accel = sm['carState'].aEgo

      if sm.valid.get('carControl', False):
        actuators = sm['carControl'].actuators
        gas_signal = actuators.gas
        brake_signal = actuators.brake

      if sm.valid.get('longitudinalPlan', False):
        plan = sm['longitudinalPlan']
        accels = list(plan.accels)
        if accels:
          desired_accel = accels[0]
        self._should_stop = plan.shouldStop
        self._allow_throttle = plan.allowThrottle
        self._allow_brake = plan.allowBrake

      if sm.valid.get('carParams', False):
        self._long_actuator_delay = sm['carParams'].longitudinalActuatorDelay

      # Push accel data
      self._accel_graph.push_data([desired_accel, actual_accel])
      self._accel_graph.set_extra_legend([
        ("Long Delay", f"{self._long_actuator_delay:.3f}s"),
      ])

      # Push control data
      self._control_graph.push_data([gas_signal, brake_signal])
      self._last_push_time = now

      # Status items in control graph legend
      stop_text = "Yes" if self._should_stop else "No"
      throttle_text = "Allowed" if self._allow_throttle else "Blocked"
      brake_text = "Allowed" if self._allow_brake else "Blocked"
      self._control_graph.set_extra_legend([
        ("Should Stop", stop_text),
        ("Throttle", throttle_text),
        ("Brake", brake_text),
      ])

    except (KeyError, AttributeError, ValueError):
      pass

  def _render(self, rect: rl.Rectangle):
    # Split rect vertically: top half for accel, bottom half for control
    gap = 8
    half_h = (rect.height - gap) / 2.0
    accel_rect = rl.Rectangle(rect.x, rect.y, rect.width, half_h)
    control_rect = rl.Rectangle(rect.x, rect.y + half_h + gap, rect.width, half_h)
    self._accel_graph.render(accel_rect)
    self._control_graph.render(control_rect)
