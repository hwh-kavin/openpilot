"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import messaging, custom

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

GREEN_LIGHT_X_THRESHOLD = 30
LEAD_DEPART_DIST_THRESHOLD = 1.0   # m, lead must move this much further before alerting (anti-jitter range)
LEAD_DEPART_ARM_DIST = 12.0        # m, arm the alert when the lead stops this close ahead
LEAD_DEPART_HOLD_DIST = 30.0       # m, keep the armed state while the lead is still tracked this close
LEAD_DEPART_DIST_EMA = 0.3         # EMA factor smoothing the radar lead distance (detection jitter)
TRIGGER_TIMER_THRESHOLD = 0.3      # s, sustained departure before the chime fires


class E2EStates:
  INACTIVE = 0
  ARMED = 1
  CONSUMED = 2


class E2EAlertsHelper:
  def __init__(self):
    self._params = Params()
    self.frame = -1
    self.green_light_state = E2EStates.INACTIVE
    self.prev_green_light_state = E2EStates.INACTIVE
    self.lead_depart_state = E2EStates.INACTIVE
    self.prev_lead_depart_state = E2EStates.INACTIVE

    self.green_light_alert = False
    self.green_light_alert_enabled = self._params.get_bool("GreenLightAlert")
    self.lead_depart_alert = False
    self.lead_depart_alert_enabled = self._params.get_bool("LeadDepartAlert")

    self.green_light_trigger_timer = 0
    self.lead_depart_trigger_timer = 0
    self.last_lead_distance = -1
    self.last_moving_frame = -1

    self.allowed = False
    self.has_lead = False
    self.lead_depart_allowed = False
    self.lead_dRel_filt = -1.0

    self.lead_depart_arm_timer = 0
    self.lead_depart_confirmed_lead = False
    self.lead_depart_armed = False

  def _read_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.green_light_alert_enabled = self._params.get_bool("GreenLightAlert")
      self.lead_depart_alert_enabled = self._params.get_bool("LeadDepartAlert")

  def update_alert_trigger(self, sm: messaging.SubMaster):
    CS = sm['carState']
    CC = sm['carControl']

    model_x = sm['modelV2'].position.x
    max_idx = len(model_x) - 1
    self.has_lead = sm['radarState'].leadOne.status
    lead_dRel = sm['radarState'].leadOne.dRel

    standstill = CS.standstill
    moving = not standstill and CS.vEgo > 0.1

    if moving:
      self.last_moving_frame = self.frame
    recent_moving = self.last_moving_frame == -1 or (self.frame - self.last_moving_frame) * DT_MDL < 2.0

    self.allowed = not moving and not CS.gasPressed and not CC.enabled and not recent_moving

    # Lead departure alert per spec: armed whenever the ego car is stopped (vEgo=0)
    # and the native radar tracks a lead — OP engagement does not gate it.
    self.lead_depart_allowed = not moving and not CS.gasPressed and not recent_moving

    # Green Light Alert
    green_light_trigger = False
    if self.green_light_state == E2EStates.ARMED:
      if model_x[max_idx] > GREEN_LIGHT_X_THRESHOLD:
        self.green_light_trigger_timer += 1
      else:
        self.green_light_trigger_timer = 0

      if self.green_light_trigger_timer * DT_MDL > TRIGGER_TIMER_THRESHOLD:
        green_light_trigger = True
    elif self.green_light_state != E2EStates.ARMED:
      self.green_light_trigger_timer = 0

    # Lead Departure Alert (native radar only)
    # Light EMA on the radar lead distance to reject detection jitter, then arm
    # while stopped behind a close lead. Once armed, the armed state is held as
    # long as the lead is still tracked nearby, so the departure trigger (dRel
    # rising more than LEAD_DEPART_DIST_THRESHOLD for >TRIGGER_TIMER_THRESHOLD)
    # can complete even though the lead leaves the close-arm window on departure.
    if not self.has_lead:
      self.lead_dRel_filt = -1.0
    elif self.lead_dRel_filt < 0.0:
      self.lead_dRel_filt = lead_dRel
    else:
      self.lead_dRel_filt += LEAD_DEPART_DIST_EMA * (lead_dRel - self.lead_dRel_filt)
    lead_dRel_smoothed = self.lead_dRel_filt if self.lead_dRel_filt >= 0.0 else lead_dRel

    close_lead_valid = self.has_lead and lead_dRel_smoothed < LEAD_DEPART_ARM_DIST
    lead_tracked = self.has_lead and lead_dRel_smoothed < LEAD_DEPART_HOLD_DIST
    if self.lead_depart_allowed and close_lead_valid:
      self.lead_depart_confirmed_lead = True
    elif not self.lead_depart_allowed:
      self.lead_depart_confirmed_lead = False

    if self.lead_depart_allowed and self.lead_depart_confirmed_lead and close_lead_valid:
      self.lead_depart_arm_timer += 1

      if self.lead_depart_arm_timer * DT_MDL >= 1.0:
        self.lead_depart_armed = True
    elif self.lead_depart_allowed and self.lead_depart_armed and lead_tracked:
      pass  # hold the armed state while the lead is still tracked nearby
    else:
      self.lead_depart_arm_timer = 0
      self.lead_depart_armed = False

    lead_depart_trigger = False
    if self.lead_depart_state == E2EStates.ARMED:
      if self.last_lead_distance == -1 or lead_dRel_smoothed < self.last_lead_distance:
        self.last_lead_distance = lead_dRel_smoothed

      # departure must exceed the anti-jitter range and persist
      if self.last_lead_distance != -1 and (lead_dRel_smoothed - self.last_lead_distance > LEAD_DEPART_DIST_THRESHOLD):
        self.lead_depart_trigger_timer += 1
      else:
        self.lead_depart_trigger_timer = 0

      if self.lead_depart_trigger_timer * DT_MDL > TRIGGER_TIMER_THRESHOLD:
        lead_depart_trigger = True
    elif self.lead_depart_state != E2EStates.ARMED:
      self.last_lead_distance = -1
      self.lead_depart_trigger_timer = 0

    return green_light_trigger, lead_depart_trigger

  @staticmethod
  def update_state_machine(state: int, enabled: bool, allowed: bool, triggered: bool) -> tuple[int, bool]:
    if state != E2EStates.INACTIVE:
      if not allowed or not enabled:
        state = E2EStates.INACTIVE

      else:
        if state == E2EStates.ARMED:
          if triggered:
            state = E2EStates.CONSUMED

        elif state == E2EStates.CONSUMED:
          pass

    elif state == E2EStates.INACTIVE:
      if allowed and enabled:
        state = E2EStates.ARMED

    return state, triggered

  def update(self, sm: messaging.SubMaster, events_sp: EventsSP) -> None:
    self._read_params()

    green_light_trigger, lead_depart_trigger = self.update_alert_trigger(sm)

    self.prev_green_light_state = self.green_light_state
    self.prev_lead_depart_state = self.lead_depart_state

    self.green_light_state, self.green_light_alert = self.update_state_machine(
      self.green_light_state,
      self.green_light_alert_enabled,
      self.allowed and not self.has_lead,
      green_light_trigger
    )

    self.lead_depart_state, self.lead_depart_alert = self.update_state_machine(
      self.lead_depart_state,
      self.lead_depart_alert_enabled,
      self.lead_depart_allowed and self.lead_depart_armed,
      lead_depart_trigger
    )

    if self.green_light_alert or self.lead_depart_alert:
      events_sp.add(custom.OnroadEventSP.EventName.e2eChime)

    self.frame += 1
