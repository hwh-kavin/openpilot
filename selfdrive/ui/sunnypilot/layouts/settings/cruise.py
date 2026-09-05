"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import SpeedLimitSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  CRUISE = 0
  SLA = 1


ICBM_DESC = tr_noop("When enabled, sunnypilot will attempt to manage the built-in cruise control buttons " +
                    "by emulating button presses for limited longitudinal control.")
ICMB_UNAVAILABLE = tr_noop("Intelligent Cruise Button Management is currently unavailable on this platform.")
ICMB_UNAVAILABLE_LONG_AVAILABLE = tr_noop("Disable the sunnypilot Longitudinal Control (alpha) toggle to allow Intelligent Cruise Button Management.")
ICMB_UNAVAILABLE_LONG_UNAVAILABLE = tr_noop("sunnypilot Longitudinal Control is the default longitudinal control for this platform.")

ACC_ENABLED_DESCRIPTION = tr_noop("Enable custom Short & Long press increments for cruise speed increase/decrease.")
ACC_NOLONG_DESCRIPTION = tr_noop("This feature can only be used with sunnypilot longitudinal control enabled.")
ACC_PCMCRUISE_DISABLED_DESCRIPTION = tr_noop("This feature is not supported on this platform due to vehicle limitations.")
ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")


class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.CRUISE
    self._speed_limit_layout = SpeedLimitSettingsLayout(lambda: self._set_current_panel(PanelType.CRUISE))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

    # Cache UI state so _update_state does not redo work every frame
    self._capability_key: tuple | None = None
    self._custom_acc_desc_key: tuple | None = None
    self._custom_acc_items_visible = False

  def _initialize_items(self):

    self.icbm_toggle = toggle_item_sp(
      title=tr("Intelligent Cruise Button Management (ICBM) (Alpha)"),
      description="",
      param="IntelligentCruiseButtonManagement")

    self.bp_long_bypass_toggle = toggle_item_sp(
      title=tr("Bypass BP Longitudinal Control"),
      description=tr("Disable BluePilot's follow-aware longitudinal control (lead classification, gas/accel limits, "
                     "brake/precharge hysteresis). Above ~50 mph with a lead it smooths following; disable to use "
                     "stock openpilot longitudinal only."),
      param="disable_BP_long_UI")

    self.bp_downhill_comp_toggle = toggle_item_sp(
      title=tr("Disable Downhill Compensation"),
      description=tr("Ignore negative pitch compensation so downhill deceleration is not added."),
      param="disable_downhill_comp_UI")

    self.scc_v_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Vision"),
      description=tr("Use vision path predictions to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlVision")

    self.scc_m_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Map"),
      description=tr("Use map data to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlMap")

    self.custom_acc_toggle = toggle_item_sp(
      title=tr("Custom ACC Speed Increments"),
      description="",
      param="CustomAccIncrementsEnabled",
      callback=self._on_custom_acc_toggle)

    self.custom_acc_short_increment = option_item_sp(
      title=tr("Short Press Increment"),
      param="CustomAccShortPressIncrement",
      min_value=1, max_value=10, value_change_step=1,
      inline=True)

    self.custom_acc_long_increment = option_item_sp(
      title=tr("Long Press Increment"),
      param="CustomAccLongPressIncrement",
      value_map={1: 1, 2: 5, 3: 10},
      min_value=1, max_value=3, value_change_step=1,
      inline=True)

    self.sla_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Speed Limit"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SLA)
    )

    self.dec_toggle = toggle_item_sp(
      title=tr("Enable Dynamic Experimental Control"),
      description=tr("Enable toggle to allow the model to determine when to use sunnypilot ACC or sunnypilot End to End Longitudinal."),
      param="DynamicExperimentalControl")

    items = [
      self.icbm_toggle,
      self.dec_toggle,
      self.bp_long_bypass_toggle,
      self.bp_downhill_comp_toggle,
      self.scc_v_toggle,
      self.scc_m_toggle,
      self.custom_acc_toggle,
      self.custom_acc_short_increment,
      self.custom_acc_long_increment,
      self.sla_settings_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.SLA:
      self._speed_limit_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    super().show_event()
    self._capability_key = None
    self._custom_acc_desc_key = None
    self._set_current_panel(PanelType.CRUISE)
    self._scroller.show_event()
    self.icbm_toggle.show_description(True)
    self.custom_acc_toggle.show_description(True)

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.SLA:
      self._speed_limit_layout.show_event()

  def _update_state(self):
    super()._update_state()

    if ui_state.CP is not None and ui_state.CP_SP is not None:
      has_icbm = ui_state.has_icbm
      has_long = ui_state.has_longitudinal_control
      capability_key = (
        has_icbm,
        has_long,
        ui_state.CP_SP.intelligentCruiseButtonManagementAvailable,
        ui_state.CP.alphaLongitudinalAvailable,
        ui_state.CP.pcmCruise,
        ui_state.CP.brand,
        ui_state.is_offroad(),
      )
    else:
      has_icbm = has_long = False
      capability_key = (False, False, False, False, False, None, ui_state.is_offroad())

    if capability_key != self._capability_key:
      self._capability_key = capability_key
      self._apply_capability_state(has_icbm, has_long)

    custom_acc_desc_key = (ui_state.is_offroad(), has_long, has_icbm,
                           ui_state.CP.pcmCruise if ui_state.CP is not None else False)
    if custom_acc_desc_key != self._custom_acc_desc_key:
      self._custom_acc_desc_key = custom_acc_desc_key
      self._apply_custom_acc_description(has_long, has_icbm)

    custom_acc_enabled = self.custom_acc_toggle.action_item.get_state()
    if custom_acc_enabled != self._custom_acc_items_visible:
      self._custom_acc_items_visible = custom_acc_enabled
      self._on_custom_acc_toggle(custom_acc_enabled)

  def _apply_capability_state(self, has_icbm: bool, has_long: bool):
    # Param cleanup is handled by ui_state._enforce_constraints(); do not remove here every frame.

    if ui_state.CP is not None and ui_state.CP_SP is not None:
      if ui_state.CP_SP.intelligentCruiseButtonManagementAvailable and not has_long:
        self.icbm_toggle.action_item.set_enabled(ui_state.is_offroad())
        self.icbm_toggle.set_description(tr(ICBM_DESC))
      else:
        self.icbm_toggle.action_item.set_enabled(False)

        long_desc = ICMB_UNAVAILABLE
        if has_long:
          if ui_state.CP.alphaLongitudinalAvailable:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_AVAILABLE
          else:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_UNAVAILABLE

        new_desc = "<b>" + tr(long_desc) + "</b>\n\n" + tr(ICBM_DESC)
        if self.icbm_toggle.description != new_desc:
          self.icbm_toggle.set_description(new_desc)
          self.icbm_toggle.show_description(True)

      is_ford = ui_state.CP.brand == "ford"
      self.bp_long_bypass_toggle.set_visible(is_ford)
      self.bp_downhill_comp_toggle.set_visible(is_ford)

      if has_long or has_icbm:
        self.custom_acc_toggle.action_item.set_enabled(((has_long and not ui_state.CP.pcmCruise) or has_icbm) and ui_state.is_offroad())
        self.dec_toggle.action_item.set_enabled(has_long)
        self.bp_long_bypass_toggle.action_item.set_enabled(has_long and is_ford)
        self.bp_downhill_comp_toggle.action_item.set_enabled(has_long and is_ford)
        self.scc_v_toggle.action_item.set_enabled(True)
        self.scc_m_toggle.action_item.set_enabled(True)
      else:
        self.custom_acc_toggle.action_item.set_enabled(False)
        self.dec_toggle.action_item.set_enabled(False)
        self.bp_long_bypass_toggle.action_item.set_enabled(False)
        self.bp_downhill_comp_toggle.action_item.set_enabled(False)
        self.scc_v_toggle.action_item.set_enabled(False)
        self.scc_m_toggle.action_item.set_enabled(False)
    else:
      self.icbm_toggle.action_item.set_enabled(False)
      self.icbm_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))
      self.bp_long_bypass_toggle.set_visible(False)
      self.bp_long_bypass_toggle.action_item.set_enabled(False)
      self.bp_downhill_comp_toggle.set_visible(False)
      self.bp_downhill_comp_toggle.action_item.set_enabled(False)

  def _apply_custom_acc_description(self, has_long: bool, has_icbm: bool):
    show_custom_acc_desc = False

    if ui_state.is_offroad():
      new_custom_acc_desc = tr(ONROAD_ONLY_DESCRIPTION)
      show_custom_acc_desc = True
    elif has_long or has_icbm:
      if has_long and ui_state.CP is not None and ui_state.CP.pcmCruise:
        new_custom_acc_desc = tr(ACC_PCMCRUISE_DISABLED_DESCRIPTION)
        show_custom_acc_desc = True
      else:
        new_custom_acc_desc = tr(ACC_ENABLED_DESCRIPTION)
    else:
      new_custom_acc_desc = tr(ACC_NOLONG_DESCRIPTION)
      show_custom_acc_desc = True
      self.custom_acc_toggle.action_item.set_state(False)

    if self.custom_acc_toggle.description != new_custom_acc_desc:
      self.custom_acc_toggle.set_description(new_custom_acc_desc)
      if show_custom_acc_desc:
        self.custom_acc_toggle.show_description(True)

  def _on_custom_acc_toggle(self, state):
    self.custom_acc_short_increment.set_visible(state)
    self.custom_acc_long_increment.set_visible(state)
    self.custom_acc_short_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
    self.custom_acc_long_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
