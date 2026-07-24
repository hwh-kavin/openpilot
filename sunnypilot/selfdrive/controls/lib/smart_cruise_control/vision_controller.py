"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

import cereal.messaging as messaging
from cereal import custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from opendbc.car import structs
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  apply_lat_capability_v_cap,
  build_plan_kappa_traj,
  cap_vel_plan_for_scc,
  compute_actual_lat_accel,
  get_scc_accel_scale,
  kappa_from_steer_angle,
  plan_curve_speed_from_kappa_traj,
)

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState

ACTIVE_STATES = (VisionState.entering, VisionState.turning, VisionState.leaving)
ENABLED_STATES = (VisionState.enabled, VisionState.overriding, *ACTIVE_STATES)

_TURNING_LAT_ACC_TH = 1.6
_LEAVING_LAT_ACC_TH = 1.3
_FINISH_LAT_ACC_TH = 1.1

_NO_OVERSHOOT_TIME_HORIZON = 3.5  # s

_A_TARGET_FILTER_RC = 0.45
_V_TARGET_FILTER_RC = 0.55
_V_TARGET_RISE_RC = 0.18

_LEAVING_ACC = 0.6

_ENTER_SPEED_MARGIN = 1.025
_EXIT_SPEED_MARGIN = 1.008

# Past-apex: peak κ within this horizon and falling ahead → leaving
_EXIT_PEAK_T_MAX = 1.5  # s


class SmartCruiseControlVision:
  v_target: float = 0
  a_target: float = 0.
  v_ego: float = 0.
  a_ego: float = 0.
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.

  def __init__(self, CP=None):
    self.params = Params()
    self.CP = CP if CP is not None else structs.CarParams()
    self.frame = -1
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.enabled = self.params.get_bool("SmartCruiseControlVision")
    self.v_cruise_setpoint = 0.

    self.state = VisionState.disabled
    self.current_lat_acc = 0.
    self.max_pred_lat_acc = 0.
    self.actual_lat_acc = 0.
    self.kappa_now = 0.
    self.peak_kappa = 0.
    self.peak_kappa_t = 0.
    self.kappa_ahead_falling = False
    self.has_curve_constraint = False
    self.planned_a_target = 0.
    self.v_passable = 0.
    self._a_target_filter = FirstOrderFilter(0.0, _A_TARGET_FILTER_RC, DT_MDL, initialized=False)
    self._v_target_filter = FirstOrderFilter(0.0, _V_TARGET_FILTER_RC, DT_MDL, initialized=False)

  @staticmethod
  def _lateral_saturated(controls_state) -> bool:
    try:
      which = controls_state.lateralControlState.which()
      lac = getattr(controls_state.lateralControlState, which)
      return bool(getattr(lac, 'saturated', False))
    except Exception:
      return False

  def get_a_target_from_control(self) -> float:
    return self.a_target

  def get_v_target_from_control(self) -> float:
    if self.is_active:
      v_turn = max(self.v_target, self.v_passable, MIN_V)
      return v_turn + self.a_target * _NO_OVERSHOOT_TIME_HORIZON

    return V_CRUISE_UNSET

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlVision")

  def _update_calculations(self, sm: messaging.SubMaster, personality) -> None:
    if not self.long_enabled:
      return

    vel_plan = cap_vel_plan_for_scc(np.array(sm['modelV2'].velocity.x), self.v_ego)
    pos_plan = np.array(sm['modelV2'].position.x)
    yaw_rate = np.array(sm['modelV2'].orientationRate.z)
    n = min(len(yaw_rate), len(vel_plan), len(pos_plan), len(ModelConstants.T_IDXS))
    t_idxs = np.array(ModelConstants.T_IDXS[:n])
    kappa_traj = build_plan_kappa_traj(yaw_rate[:n], vel_plan[:n])

    cs = sm['controlsState']
    kappa_meas = kappa_from_steer_angle(
      sm['carState'].steeringAngleDeg,
      sm['liveParameters'].angleOffsetDeg,
      self.CP.steerRatio,
      self.CP.wheelbase,
    )
    kappa_des = abs(float(cs.desiredCurvature))
    kappa_path = abs(float(cs.curvature))
    self.kappa_now = max(kappa_des, kappa_meas, kappa_path)

    self.current_lat_acc = compute_actual_lat_accel(self.v_ego, cs.curvature)
    self.actual_lat_acc = max(
      self.current_lat_acc,
      self.v_ego ** 2 * kappa_meas,
    )

    v_plan, a_plan, peak_kappa, peak_idx, has_constraint = plan_curve_speed_from_kappa_traj(
      self.v_ego, self.a_ego, kappa_traj, pos_plan[:n], t_idxs,
      self.kappa_now, personality, MIN_V,
    )
    self.peak_kappa = peak_kappa
    self.peak_kappa_t = float(t_idxs[peak_idx]) if n > 0 else 0.0
    self.has_curve_constraint = has_constraint
    # Cereal field: predicted lateral accel ≈ peak κ · v²
    self.max_pred_lat_acc = float(peak_kappa * (self.v_ego ** 2))

    # Past apex: peak early and κ ahead of peak is below peak (falling demand)
    if n > 0 and peak_idx < n - 1:
      ahead = kappa_traj[peak_idx + 1:n]
      self.kappa_ahead_falling = (
        self.peak_kappa_t <= _EXIT_PEAK_T_MAX and
        len(ahead) > 0 and
        float(np.max(ahead)) < peak_kappa * 0.85
      )
    else:
      self.kappa_ahead_falling = False

    self.v_target = apply_lat_capability_v_cap(
      v_plan, self.v_ego, cs.desiredCurvature, cs.curvature,
      self._lateral_saturated(cs), personality, MIN_V)
    self.v_passable = self.v_target
    self.planned_a_target = a_plan

  def _update_state_machine(self, personality) -> tuple[bool, bool]:
    del personality  # thresholds are κ/speed based now

    if self.state != VisionState.disabled:
      if not self.long_enabled or not self.enabled:
        self.state = VisionState.disabled
      elif self.long_override:
        self.state = VisionState.overriding

      else:
        need_slow = (
          self.has_curve_constraint and
          self.v_ego > MIN_V and
          self.v_ego > self.v_passable * _ENTER_SPEED_MARGIN
        )

        if self.state == VisionState.enabled:
          if need_slow:
            self.state = VisionState.entering

        elif self.state == VisionState.overriding:
          if not self.long_override:
            self.state = VisionState.enabled

        elif self.state == VisionState.entering:
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH or self.actual_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          elif self.kappa_ahead_falling and self.current_lat_acc >= _LEAVING_LAT_ACC_TH:
            self.state = VisionState.leaving
          elif not need_slow:
            self.state = VisionState.enabled

        elif self.state == VisionState.turning:
          if self.current_lat_acc <= _LEAVING_LAT_ACC_TH and self.kappa_ahead_falling:
            self.state = VisionState.leaving
          elif self.current_lat_acc <= _LEAVING_LAT_ACC_TH and not need_slow:
            self.state = VisionState.enabled

        elif self.state == VisionState.leaving:
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          elif self.current_lat_acc < _FINISH_LAT_ACC_TH or not need_slow:
            self.state = VisionState.enabled

        if self.state in ACTIVE_STATES and self.v_ego <= self.v_passable * _EXIT_SPEED_MARGIN:
          self.state = VisionState.enabled

    elif self.state == VisionState.disabled:
      if self.long_enabled and self.enabled:
        if self.long_override:
          self.state = VisionState.overriding
        else:
          self.state = VisionState.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    return enabled, active

  def _update_solution(self, personality) -> float:
    if self.state not in ACTIVE_STATES:
      return float(self.a_ego)
    if self.state == VisionState.leaving:
      return float(_LEAVING_ACC * get_scc_accel_scale(personality))
    if self.v_ego <= self.v_passable:
      return float(max(0.0, self.a_ego))
    # planned_a_target already includes personality brake scaling
    return float(self.planned_a_target)

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise_setpoint: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise_setpoint

    personality = sm['selfdriveState'].personality

    self._update_params()
    self._update_calculations(sm, personality)

    self.is_enabled, self.is_active = self._update_state_machine(personality)
    raw_a_target = self._update_solution(personality)
    self.a_target = self._a_target_filter.update(raw_a_target)

    if self.is_active:
      v_turn = max(self.v_target, self.v_passable, MIN_V)
      if v_turn > self._v_target_filter.x:
        self._v_target_filter.update_alpha(_V_TARGET_RISE_RC)
      else:
        self._v_target_filter.update_alpha(_V_TARGET_FILTER_RC)
      self._v_target_filter.update(v_turn)
      self.v_target = self._v_target_filter.x
    elif not self._v_target_filter.initialized:
      self._v_target_filter.update(max(self.v_ego, MIN_V))

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
