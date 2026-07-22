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
  apply_model_uncertainty_v_cap,
  cap_vel_plan_for_scc,
  combine_scc_model_actual_lat_acc,
  compute_actual_lat_accel,
  compute_curve_lat_acc_uncertainty,
  compute_scc_curve_v_target,
  compute_scc_passable_speed,
  compute_steer_angle_lat_accel,
  get_scc_abort_enter_lat_acc_th,
  get_scc_accel_scale,
  get_scc_early_abort_lat_acc_th,
  get_scc_early_enter_lat_acc_th,
  get_scc_enter_lat_acc_th,
  get_scc_lat_accel_max,
  inflate_lat_acc_with_uncertainty,
  inflate_pred_lat_accels_with_uncertainty,
)

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState

ACTIVE_STATES = (VisionState.entering, VisionState.turning, VisionState.leaving)
ENABLED_STATES = (VisionState.enabled, VisionState.overriding, *ACTIVE_STATES)

# Near-term path triggers strong response; far-term path enables early prediction.
_NEAR_LOOKAHEAD_T_S = 5.0
_FAR_LOOKAHEAD_T_S = 8.0
_ENTER_PRED_PERCENTILE = 95
_FAR_PRED_PERCENTILE = 92
_V_TARGET_PRED_PERCENTILE = 97

_TURNING_LAT_ACC_TH = 1.6  # Lat Acc threshold to trigger turning state.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger the end of the turn cycle.

_NO_OVERSHOOT_TIME_HORIZON = 3.5  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.05, -0.50]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.2, 3.]  # absolute value of lat acc ahead

_A_TARGET_FILTER_RC = 0.45  # s, smooth accel target across turn state transitions
_V_TARGET_FILTER_RC = 0.55  # s, smooth curve speed target fed to MPC (decel)
_V_TARGET_RISE_RC = 0.18  # s, faster recovery when curve speed target rises
_PRED_ENTER_FILTER_RC = 0.50  # s, smooth predicted lat acc for state transitions (rise)
_PRED_DECAY_RC = 0.22  # s, faster decay when path straightens

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.4, 0.05, -0.25]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.]  # absolute value of current lat acc

_LEAVING_ACC = 0.6  # Conformable acceleration to regain speed while leaving a turn.

# Hysteresis: only brake when clearly above passable speed; release as soon as speed is OK.
_ENTER_SPEED_MARGIN = 1.025
_EXIT_SPEED_MARGIN = 1.008


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
    self.max_pred_lat_acc_enter = 0.
    self.max_pred_lat_acc_far = 0.
    self.actual_lat_acc = 0.
    self.lat_acc_for_v = 0.
    self.curve_lat_acc_unc = 0.
    self.v_passable = 0.
    self._a_target_filter = FirstOrderFilter(0.0, _A_TARGET_FILTER_RC, DT_MDL, initialized=False)
    self._v_target_filter = FirstOrderFilter(0.0, _V_TARGET_FILTER_RC, DT_MDL, initialized=False)
    self._pred_enter_filter = FirstOrderFilter(0.0, _PRED_ENTER_FILTER_RC, DT_MDL, initialized=False)
    self._pred_far_filter = FirstOrderFilter(0.0, _PRED_ENTER_FILTER_RC, DT_MDL, initialized=False)

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

  def _update_pred_filter(self, filt: FirstOrderFilter, raw: float) -> float:
    if raw < filt.x:
      filt.update_alpha(_PRED_DECAY_RC)
    else:
      filt.update_alpha(_PRED_ENTER_FILTER_RC)
    return filt.update(raw)

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlVision")

  def _update_calculations(self, sm: messaging.SubMaster, personality) -> None:
    if not self.long_enabled:
      return

    rate_plan = np.array(np.abs(sm['modelV2'].orientationRate.z))
    vel_plan = cap_vel_plan_for_scc(np.array(sm['modelV2'].velocity.x), self.v_ego)
    pos_plan = np.array(sm['modelV2'].position.x)

    cs = sm['controlsState']
    a_path = compute_actual_lat_accel(self.v_ego, cs.curvature)
    a_steer = compute_steer_angle_lat_accel(
      self.v_ego,
      sm['carState'].steeringAngleDeg,
      sm['liveParameters'].angleOffsetDeg,
      self.CP.steerRatio,
      self.CP.wheelbase,
    )
    # Path curvature for state transitions; fuse path + steer angle for speed target.
    self.current_lat_acc = a_path
    self.actual_lat_acc = max(a_path, a_steer)

    predicted_lat_accels = rate_plan * vel_plan
    t_idxs = np.array(ModelConstants.T_IDXS[:len(predicted_lat_accels)])

    self.max_pred_lat_acc = float(np.percentile(predicted_lat_accels, _V_TARGET_PRED_PERCENTILE))

    near_mask = t_idxs <= _NEAR_LOOKAHEAD_T_S
    far_mask = t_idxs <= _FAR_LOOKAHEAD_T_S
    near_pred = predicted_lat_accels[near_mask] if np.any(near_mask) else predicted_lat_accels
    far_pred = predicted_lat_accels[far_mask] if np.any(far_mask) else predicted_lat_accels

    # Adaptive percentile: use model uncertainty (yaw_rate std) to boost percentile.
    # Higher relative uncertainty → higher percentile → more conservative deceleration.
    z_std = np.asarray(sm['modelV2'].orientationRate.zStd, dtype=float)
    raw_near_pred = float(np.percentile(near_pred, _ENTER_PRED_PERCENTILE))
    raw_far_pred = float(np.percentile(far_pred, _FAR_PRED_PERCENTILE))
    near_unc = 0.0
    if len(z_std) > 0:
      near_std = z_std[near_mask[:len(z_std)]] if np.any(near_mask) else z_std
      far_std = z_std[far_mask[:len(z_std)]] if np.any(far_mask) else z_std
      # Coefficient of variation: cv = σ / |μ|.  Cap at zero yaw to avoid division issues.
      near_mean = float(np.mean(np.abs(rate_plan[near_mask]))) if np.any(near_mask) else 0.0
      far_mean = float(np.mean(np.abs(rate_plan[far_mask]))) if np.any(far_mask) else 0.0
      near_cv = float(np.mean(near_std)) / max(near_mean, 1e-6)
      far_cv = float(np.mean(far_std)) / max(far_mean, 1e-6)
      # Map cv to percentile boost: cv=0 → 0, cv≥1 → +5pp
      near_boost = float(np.clip(near_cv * 5.0, 0.0, 5.0))
      far_boost = float(np.clip(far_cv * 5.0, 0.0, 5.0))
      near_pct = int(np.clip(_ENTER_PRED_PERCENTILE + near_boost, 90, 99))
      far_pct = int(np.clip(_FAR_PRED_PERCENTILE + far_boost, 90, 99))
      raw_near_pred = float(np.percentile(near_pred, near_pct))
      raw_far_pred = float(np.percentile(far_pred, far_pct))
      near_unc = compute_curve_lat_acc_uncertainty(z_std, vel_plan, near_mask)
    self.curve_lat_acc_unc = near_unc
    if not self._pred_enter_filter.initialized:
      self._pred_enter_filter.update(raw_near_pred)
    if not self._pred_far_filter.initialized:
      self._pred_far_filter.update(raw_far_pred)
    self.max_pred_lat_acc_enter = self._update_pred_filter(self._pred_enter_filter, raw_near_pred)
    self.max_pred_lat_acc_far = self._update_pred_filter(self._pred_far_filter, raw_far_pred)

    # Inflate predicted lat accel with absolute uncertainty so unclear curves slow earlier.
    model_lat_acc = max(self.max_pred_lat_acc_enter, self.max_pred_lat_acc)
    model_lat_acc = inflate_lat_acc_with_uncertainty(model_lat_acc, near_unc)
    predicted_for_v = inflate_pred_lat_accels_with_uncertainty(predicted_lat_accels, z_std, vel_plan)

    self.lat_acc_for_v = combine_scc_model_actual_lat_acc(model_lat_acc, self.actual_lat_acc, personality)
    self.v_target = compute_scc_curve_v_target(
      self.v_ego, self.lat_acc_for_v, personality, MIN_V, pos_plan, predicted_for_v, vel_plan)
    self.v_target = apply_lat_capability_v_cap(
      self.v_target, self.v_ego, cs.desiredCurvature, cs.curvature,
      self._lateral_saturated(cs), personality, MIN_V)
    # Extra speed cut only in curve context (avoid highway noise false decel).
    curve_ctx = (
      self.lat_acc_for_v > get_scc_abort_enter_lat_acc_th(personality) or
      self.max_pred_lat_acc_enter > get_scc_early_enter_lat_acc_th(personality) or
      self.actual_lat_acc > get_scc_abort_enter_lat_acc_th(personality)
    )
    if curve_ctx and near_unc > 0.0:
      self.v_target = apply_model_uncertainty_v_cap(self.v_target, near_unc, MIN_V)
    self.v_passable = self.v_target

  def _update_state_machine(self, personality) -> tuple[bool, bool]:
    abort_th = get_scc_abort_enter_lat_acc_th(personality)
    early_abort = get_scc_early_abort_lat_acc_th(personality)

    # ENABLED, ENTERING, TURNING, LEAVING, OVERRIDING
    if self.state != VisionState.disabled:
      # longitudinal and feature disable always have priority in a non-disabled state
      if not self.long_enabled or not self.enabled:
        self.state = VisionState.disabled
      elif self.long_override:
        self.state = VisionState.overriding

      else:
        # ENABLED
        if self.state == VisionState.enabled:
          if self.v_ego <= MIN_V:
            pass
          # Only slow down when current speed exceeds the curve passable speed.
          elif (self.v_ego > self.v_passable * _ENTER_SPEED_MARGIN and
                (self.max_pred_lat_acc_enter > abort_th or self.actual_lat_acc > abort_th)):
            self.state = VisionState.entering

        # OVERRIDING
        elif self.state == VisionState.overriding:
          if not self.long_override:
            self.state = VisionState.enabled

        # ENTERING
        elif self.state == VisionState.entering:
          # Transition to Turning if current lateral acceleration is over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Abort if the predicted lateral acceleration drops
          elif (self.max_pred_lat_acc_enter < abort_th and self.max_pred_lat_acc_far < early_abort and
                self.actual_lat_acc < abort_th):
            self.state = VisionState.enabled

        # TURNING
        elif self.state == VisionState.turning:
          # Transition to Leaving if current lateral acceleration drops below a threshold.
          if self.current_lat_acc <= _LEAVING_LAT_ACC_TH:
            self.state = VisionState.leaving

        # LEAVING
        elif self.state == VisionState.leaving:
          # Transition back to Turning if current lateral acceleration goes back over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Finish if current lateral acceleration goes below a threshold.
          elif self.current_lat_acc < _FINISH_LAT_ACC_TH:
            self.state = VisionState.enabled

        if self.state in ACTIVE_STATES and self.v_ego <= self.v_passable * _EXIT_SPEED_MARGIN:
          self.state = VisionState.enabled

    # DISABLED
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
    enter_th = get_scc_enter_lat_acc_th(personality)
    a_lat = get_scc_lat_accel_max(personality)
    decel_bp = [enter_th, max(a_lat + 0.5, enter_th + 0.5)]

    # DISABLED, ENABLED, OVERRIDING
    if self.state not in ACTIVE_STATES:
      a_target = self.a_ego
    # ENTERING
    elif self.state == VisionState.entering:
      if self.v_ego <= self.v_passable:
        a_target = max(0.0, self.a_ego)
      else:
        lat_acc_for_decel = self.lat_acc_for_v
        a_target = np.interp(lat_acc_for_decel, decel_bp, _ENTERING_SMOOTH_DECEL_V)
    # TURNING
    elif self.state == VisionState.turning:
      if self.v_ego <= self.v_passable:
        a_target = max(0.0, self.a_ego)
      else:
        a_target = np.interp(max(self.current_lat_acc, self.lat_acc_for_v),
                             _TURNING_ACC_BP, _TURNING_ACC_V)
    # LEAVING
    elif self.state == VisionState.leaving:
      a_target = _LEAVING_ACC
    else:
      raise NotImplementedError(f"SCC-V state not supported: {self.state}")

    return float(a_target * get_scc_accel_scale(personality))

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise_setpoint: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise_setpoint

    personality = sm['selfdriveState'].personality

    self._update_params()
    if not long_enabled:
      self._pred_enter_filter.initialized = False
      self._pred_far_filter.initialized = False

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
