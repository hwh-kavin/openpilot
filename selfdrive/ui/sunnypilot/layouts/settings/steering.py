"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from cereal import car
from enum import IntEnum

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, simple_button_item_sp, option_item_sp, LineSeparatorSP
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.lane_change_settings import LaneChangeSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.mads_settings import MadsSettingsLayout
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.steering_sub_layouts.torque_settings import TorqueSettingsLayout


class PanelType(IntEnum):
  STEERING = 0
  MADS = 1
  LANE_CHANGE = 2
  TORQUE_CONTROL = 3


class SteeringLayout(Widget):
  def __init__(self):
    super().__init__()

    self._current_panel = PanelType.STEERING
    self._lane_change_settings_layout = LaneChangeSettingsLayout(lambda: self._set_current_panel(PanelType.STEERING))
    self._mads_settings_layout = MadsSettingsLayout(lambda: self._set_current_panel(PanelType.STEERING))
    self._torque_control_layout = TorqueSettingsLayout(lambda: self._set_current_panel(PanelType.STEERING))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=False, spacing=0)

  def _initialize_items(self):
    self._mads_base_desc = tr("Enable the beloved MADS feature. " +
                              "Disable toggle to revert back to stock sunnypilot engagement/disengagement.")
    self._mads_limited_desc = tr("This platform supports limited MADS settings.")
    self._mads_full_desc = tr("This platform supports all MADS settings.")
    self._mads_check_compat_desc = tr("Start the vehicle to check vehicle compatibility.")

    self._mads_toggle = toggle_item_sp(
      param="Mads",
      title=lambda: tr("Modular Assistive Driving System (MADS)"),
      description=self._mads_base_desc,
    )
    self._mads_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Customize MADS"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.MADS)
    )
    self._lane_change_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Customize Lane Change"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.LANE_CHANGE)
    )
    self._blinker_control_toggle = toggle_item_sp(
      param="BlinkerPauseLateralControl",
      description=lambda: tr("Pause lateral control with blinker when traveling below the desired speed selected."),
      title=lambda: tr("Pause Lateral Control with Blinker"),
    )
    self._blinker_control_options = option_item_sp(
      param="BlinkerMinLateralControlSpeed",
      title=lambda: tr("Minimum Speed to Pause Lateral Control"),
      min_value=0,
      max_value=255,
      value_change_step=5,
      description="",
      label_callback=lambda speed: f'{speed} {"km/h" if ui_state.is_metric else "mph"}',
    )
    self._blinker_reengage_delay = option_item_sp(
      param="BlinkerLateralReengageDelay",
      title=lambda: tr("Post-Blinker Delay"),
      min_value=0,
      max_value=10,
      value_change_step=1,
      description=lambda: tr("Delay before lateral control resumes after the turn signal ends."),
      label_callback=lambda delay: f'{delay} {"s"}'
    )
    self._torque_control_toggle = toggle_item_sp(
      param="EnforceTorqueControl",
      title=lambda: tr("Enforce Torque Lateral Control"),
      description=lambda: tr("Enable this to enforce sunnypilot to steer with Torque lateral control."),
    )
    self._torque_customization_button = simple_button_item_sp(
      button_text=lambda: tr("Customize Torque Params"),
      button_width=850,
      callback=lambda: self._set_current_panel(PanelType.TORQUE_CONTROL)
    )
    self._nnlc_toggle = toggle_item_sp(
      param="NeuralNetworkLateralControl",
      title=lambda: tr("Neural Network Lateral Control (NNLC)"),
      description=""
    )
    self._htd_toggle = toggle_item_sp(
      param="dp_htd_enabled",
      title=lambda: tr("启用人工转弯检测HTD"),
      description=lambda: tr("人工大幅度转弯，或出弯后期望角已小但与实际角差值仍大时，暂停横向以借助车身回正，对齐后恢复。"),
    )
    self._htd_turn_angle_threshold = option_item_sp(
      param="dp_htd_turn_angle_threshold",
      title=lambda: tr("HTD触发转角门限"),
      min_value=15,
      max_value=90,
      value_change_step=5,
      description=lambda: tr("手握方向盘时，超过该转角将触发人工转弯检测。"),
      label_callback=lambda angle: f"{angle}°",
    )
    self._htd_resume_angle_diff = option_item_sp(
      param="dp_htd_resume_angle_diff",
      title=lambda: tr("HTD恢复角度差"),
      min_value=1,
      max_value=30,
      value_change_step=1,
      description=lambda: tr("人工转弯：模型期望角与实际角差值低于该值可恢复横向控制。"),
      label_callback=lambda diff: f"{diff}°",
    )
    self._htd_resume_delay = option_item_sp(
      param="dp_htd_resume_delay_ms",
      title=lambda: tr("HTD恢复延迟"),
      min_value=0,
      max_value=2000,
      value_change_step=50,
      description=lambda: tr("人工转弯满足恢复条件后，延迟该时间再恢复横向控制。"),
      label_callback=lambda delay: f"{delay} ms",
    )
    self._htd_curve_exit_toggle = toggle_item_sp(
      param="dp_htd_curve_exit_enabled",
      title=lambda: tr("出弯自动释放横向"),
      description=lambda: tr("出弯后模型已近直行且实际角与期望角差值偏大时释放横向，车身回正至接近期望后立即接回。"),
    )
    self._htd_curve_latch = option_item_sp(
      param="dp_htd_curve_latch_angle",
      title=lambda: tr("弯道确认转角"),
      min_value=8,
      max_value=40,
      value_change_step=1,
      description=lambda: tr("模型期望转角持续达到该值以上，才开始累计弯道确认距离（推荐约16度）。"),
      label_callback=lambda angle: f"{angle}°",
    )
    self._htd_curve_latch_distance = option_item_sp(
      param="dp_htd_curve_latch_distance",
      title=lambda: tr("弯道确认距离"),
      min_value=5,
      max_value=50,
      value_change_step=5,
      description=lambda: tr("期望转角持续超门限并行驶不少于该距离，才确认有效弯道，用于过滤模型短暂抖动（推荐约10米）。"),
      label_callback=lambda dist: f"{dist} m",
    )
    self._htd_curve_exit_model = option_item_sp(
      param="dp_htd_curve_exit_model_angle",
      title=lambda: tr("模型直行门限"),
      min_value=2,
      max_value=15,
      value_change_step=1,
      description=lambda: tr("模型期望转角不高于该值（视为已近直行）才触发出弯释放（推荐约6度）。"),
      label_callback=lambda angle: f"{angle}°",
    )
    self._htd_curve_exit_error = option_item_sp(
      param="dp_htd_curve_exit_error",
      title=lambda: tr("触发释放角度差"),
      min_value=5,
      max_value=25,
      value_change_step=1,
      description=lambda: tr("实际转角与模型期望的差值不低于该值时触发出弯释放（推荐约8度）。"),
      label_callback=lambda diff: f"{diff}°",
    )
    self._htd_curve_exit_resume_error = option_item_sp(
      param="dp_htd_curve_exit_resume_error",
      title=lambda: tr("横向接回角度差"),
      min_value=2,
      max_value=15,
      value_change_step=1,
      description=lambda: tr("实际角与期望角差值低于该值时立即接回横向（推荐约5度，应小于释放角度差）。"),
      label_callback=lambda diff: f"{diff}°",
    )

    items = [
      self._mads_toggle,
      self._mads_settings_button,
      LineSeparatorSP(40),
      self._lane_change_settings_button,
      LineSeparatorSP(40),
      self._blinker_control_toggle,
      self._blinker_control_options,
      self._blinker_reengage_delay,
      LineSeparatorSP(40),
      self._torque_control_toggle,
      self._torque_customization_button,
      LineSeparatorSP(40),
      self._nnlc_toggle,
      LineSeparatorSP(40),
      self._htd_toggle,
      self._htd_turn_angle_threshold,
      self._htd_resume_angle_diff,
      self._htd_resume_delay,
      self._htd_curve_exit_toggle,
      self._htd_curve_latch,
      self._htd_curve_latch_distance,
      self._htd_curve_exit_model,
      self._htd_curve_exit_error,
      self._htd_curve_exit_resume_error,
    ]
    return items

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel

  def _update_state(self):
    super()._update_state()

    torque_allowed = ui_state.CP is not None and ui_state.CP.steerControlType != car.CarParams.SteerControlType.angle
    if ui_state.CP is not None:
      mads_main_desc = self._mads_limited_desc if self._mads_settings_layout._mads_limited_settings() else self._mads_full_desc
      self._mads_toggle.set_description(f"<b>{mads_main_desc}</b><br><br>{self._mads_base_desc}")
    else:
      self._mads_toggle.set_description(f"<b>{self._mads_check_compat_desc}</b><br><br>{self._mads_base_desc}")

    self._mads_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._mads_settings_button.action_item.set_enabled(ui_state.is_offroad() and self._mads_toggle.action_item.get_state())
    self._blinker_control_options.set_visible(self._blinker_control_toggle.action_item.get_state())
    self._blinker_reengage_delay.set_visible(self._blinker_control_toggle.action_item.get_state())

    enforce_torque_enabled = self._torque_control_toggle.action_item.get_state()
    nnlc_enabled = self._nnlc_toggle.action_item.get_state()
    self._nnlc_toggle.action_item.set_enabled(ui_state.is_offroad() and torque_allowed and not enforce_torque_enabled)
    self._torque_control_toggle.action_item.set_enabled(ui_state.is_offroad() and torque_allowed and not nnlc_enabled)
    self._torque_customization_button.action_item.set_enabled(self._torque_control_toggle.action_item.get_state())

    htd_enabled = self._htd_toggle.action_item.get_state()
    self._htd_turn_angle_threshold.set_visible(htd_enabled)
    self._htd_resume_angle_diff.set_visible(htd_enabled)
    self._htd_resume_delay.set_visible(htd_enabled)
    self._htd_curve_exit_toggle.set_visible(htd_enabled)
    curve_exit_enabled = htd_enabled and self._htd_curve_exit_toggle.action_item.get_state()
    self._htd_curve_exit_error.set_visible(curve_exit_enabled)
    self._htd_curve_exit_resume_error.set_visible(curve_exit_enabled)
    self._htd_curve_exit_model.set_visible(curve_exit_enabled)
    self._htd_curve_latch.set_visible(curve_exit_enabled)
    self._htd_curve_latch_distance.set_visible(curve_exit_enabled)

  def _render(self, rect):
    if self._current_panel == PanelType.LANE_CHANGE:
      self._lane_change_settings_layout.render(rect)
    elif self._current_panel == PanelType.MADS:
      self._mads_settings_layout.render(rect)
    elif self._current_panel == PanelType.TORQUE_CONTROL:
      self._torque_control_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.STEERING)
    self._scroller.show_event()
