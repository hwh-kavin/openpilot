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
    # --- HTD ---
    self._htd_enabled = toggle_item_sp(
      param="dp_htd_enabled",
      title=lambda: tr("Enable Human Turn Detection"),
      description=lambda: tr("Pause lateral control when the driver turns the wheel with hands on wheel past the trigger angle, then resume after recovery conditions are met."),
    )

    self._htd_turn_angle_threshold = option_item_sp(
      param="dp_htd_turn_angle_threshold",
      title=lambda: tr("Trigger angle"),
      description=lambda: tr("Driver steering angle that triggers HTD (degrees)."),
      min_value=40,
      max_value=90,
      value_change_step=1,
      label_callback=lambda angle: f'{angle} °',
    )

    self._htd_resume_angle_diff = option_item_sp(
      param="dp_htd_resume_angle_diff",
      title=lambda: tr("Resume angle difference"),
      description=lambda: tr("Maximum filtered angle difference between model desired steering and wheel angle to resume (degrees)."),
      min_value=5,
      max_value=20,
      value_change_step=1,
      label_callback=lambda angle: f'{angle} °',
    )

    self._htd_resume_delay = option_item_sp(
      param="dp_htd_resume_delay_ms",
      title=lambda: tr("Resume delay"),
      description=lambda: tr("Delay before lateral control resumes after recovery conditions are met."),
      min_value=0,
      max_value=5000,
      value_change_step=100,
      label_callback=lambda delay: f'{delay} ms',
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

    items = [
      self._mads_toggle,
      self._mads_settings_button,
      LineSeparatorSP(40),
      self._lane_change_settings_button,
      LineSeparatorSP(40),
      self._htd_enabled,
      self._htd_turn_angle_threshold,
      self._htd_resume_angle_diff,
      self._htd_resume_delay,
      LineSeparatorSP(40),
      self._torque_control_toggle,
      self._torque_customization_button,
      LineSeparatorSP(40),
      self._nnlc_toggle,
    ]
    return items

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel

  def _update_state(self):
    super()._update_state()

    torque_allowed = True
    if ui_state.CP is not None:
      mads_main_desc = self._mads_limited_desc if self._mads_settings_layout._mads_limited_settings() else self._mads_full_desc
      self._mads_toggle.set_description(f"<b>{mads_main_desc}</b><br><br>{self._mads_base_desc}")

      if ui_state.CP.steerControlType == car.CarParams.SteerControlType.angle:
        ui_state.params.remove("EnforceTorqueControl")
        ui_state.params.remove("NeuralNetworkLateralControl")
        torque_allowed = False
    else:
      self._mads_toggle.set_description(f"<b>{self._mads_check_compat_desc}</b><br><br>{self._mads_base_desc}")
      ui_state.params.remove("EnforceTorqueControl")
      ui_state.params.remove("NeuralNetworkLateralControl")
      torque_allowed = False

    self._mads_toggle.action_item.set_enabled(ui_state.is_offroad())
    self._mads_settings_button.action_item.set_enabled(ui_state.is_offroad() and self._mads_toggle.action_item.get_state())

    enforce_torque_enabled = self._torque_control_toggle.action_item.get_state()
    nnlc_enabled = self._nnlc_toggle.action_item.get_state()
    self._nnlc_toggle.action_item.set_enabled(ui_state.is_offroad() and torque_allowed and not enforce_torque_enabled)
    self._torque_control_toggle.action_item.set_enabled(ui_state.is_offroad() and torque_allowed and not nnlc_enabled)
    self._torque_customization_button.action_item.set_enabled(self._torque_control_toggle.action_item.get_state())

    htd_is_enabled = self._htd_enabled.action_item.get_state()
    self._htd_turn_angle_threshold.set_visible(htd_is_enabled)
    self._htd_resume_angle_diff.set_visible(htd_is_enabled)
    self._htd_resume_delay.set_visible(htd_is_enabled)

    offroad = ui_state.is_offroad()
    self._htd_enabled.action_item.set_enabled(offroad)
    self._htd_turn_angle_threshold.action_item.set_enabled(offroad)
    self._htd_resume_angle_diff.action_item.set_enabled(offroad)
    self._htd_resume_delay.action_item.set_enabled(offroad)

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
