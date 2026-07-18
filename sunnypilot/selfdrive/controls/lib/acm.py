import time
import numpy as np
from cereal import log, custom

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  PITCH_STOP_DISTANCE_MAX,
  get_coast_accel,
  get_safe_obstacle_distance,
  get_stopped_equivalence_factor,
  get_T_FOLLOW,
)

AcmComfortState = custom.LongitudinalPlanSP.AcmComfortState

# =========================================================
# 參數設定區
# =========================================================

TRAJECTORY_HORIZON = 6
INTENT_LOOKAHEAD = 3
INTENT_V_LOW = 0.0
INTENT_V_HIGH = 22.22
INTENT_FRAMES_LOW = 1
INTENT_FRAMES_HIGH = 20

SPEED_OFFSET_MIN_KPH = 1.0
SPEED_OFFSET_MAX_FLAT_KPH = 15.0
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0

PITCH_UPHILL_THRESHOLD = 0.050
PITCH_DOWNHILL_THRESHOLD = -0.030

SOFT_HOLD_PITCH_START = 0.050
SOFT_HOLD_PITCH_MAX = 0.080

EMERGENCY_TTC = 2.0
EMERGENCY_RELATIVE_SPEED = 10.0
EMERGENCY_DECEL_THRESHOLD = -1.5

LEAD_COOLDOWN_TIME = 0.5
# Coast is for overshoot only (set → set+offset). Tiny hysteresis avoids chatter at set speed;
# must stay far smaller than the old ~10 km/h band that let speed sag below cruise.
CRUISE_SPEED_ENTRY_TOLERANCE = 0.3  # m/s (~1 km/h) below v_cruise
FOLLOW_COAST_COOLDOWN_TIME = 0.5
FOLLOW_COAST_SPEED_MAX = 5.5
FOLLOW_COAST_MARGIN_BASE = 0.8
FOLLOW_COAST_MARGIN_STRENGTH = 1.5
FOLLOW_COAST_MARGIN_PITCH = 0.7
FOLLOW_COAST_LEAD_BRAKE = -0.5

SPEED_BP = [0., 10., 20., 30.]
MIN_DIST_V = [5., 10., 15., 20.]

SOFT_HOLD_RANGE_MIN = 0.70
SOFT_HOLD_RANGE_MAX = 0.99
SOFT_HOLD_TTC_THRESHOLD = 2.5
COAST_TTC_FULL = 6.0
VREL_DEBOUNCE_TIME = 0.6

SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V = [1.1, 0.90, 0.70, 0.50, 0.30, 0.10]

RATIO_ENTER_THRESHOLD = 1.08
RATIO_EXIT_THRESHOLD = 1.05
TARGET_FACTOR_FILTER_ALPHA = 0.3
SOFT_HOLD_HYSTERESIS_TIME = 1.0


class ComfortState:
  OFF = 0
  CRUISE_COAST = 1
  FOLLOW_COAST = 2
  MPC_FOLLOW = 3


COMFORT_STATE_TO_CAPNP = {
  ComfortState.OFF: AcmComfortState.off,
  ComfortState.CRUISE_COAST: AcmComfortState.cruiseCoast,
  ComfortState.FOLLOW_COAST: AcmComfortState.followCoast,
  ComfortState.MPC_FOLLOW: AcmComfortState.mpcFollow,
}


def _lead_closing_speed(v_ego: float, v_lead: float) -> float:
  return max(v_ego - v_lead, 0.1)


def compute_lead_ttc(lead, v_ego: float) -> float | None:
  if not lead or not lead.status:
    return None
  return lead.dRel / _lead_closing_speed(v_ego, lead.vLead)


def compute_safe_distance_A(v_ego: float, t_follow: float, pitch: float | None = None) -> float:
  return get_safe_obstacle_distance(v_ego, t_follow, pitch)


def compute_lead_ratio(lead, v_ego: float, t_follow: float, pitch: float | None = None) -> float | None:
  if not lead or not lead.status:
    return None
  desired_dist = compute_safe_distance_A(v_ego, t_follow, pitch)
  lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)
  if desired_dist < 0.1:
    return 10.0
  return lead_obstacle_dist / desired_dist


def compute_coast_strength(lead, v_ego: float, t_follow: float, pitch: float | None = None) -> float:
  if not lead or not lead.status:
    return 1.0

  ttc = compute_lead_ttc(lead, v_ego)
  ratio = compute_lead_ratio(lead, v_ego, t_follow, pitch)
  if ttc is None or ratio is None:
    return 1.0

  ttc_span = max(COAST_TTC_FULL - SOFT_HOLD_TTC_THRESHOLD, 0.1)
  ttc_factor = float(np.clip((ttc - SOFT_HOLD_TTC_THRESHOLD) / ttc_span, 0.0, 1.0))

  ratio_span = max(RATIO_ENTER_THRESHOLD - SOFT_HOLD_RANGE_MIN, 0.01)
  ratio_factor = float(np.clip((ratio - SOFT_HOLD_RANGE_MIN) / ratio_span, 0.0, 1.0))

  return min(ttc_factor, ratio_factor)


def compute_follow_margin(strength: float, pitch: float) -> float:
  base = FOLLOW_COAST_MARGIN_BASE + strength * FOLLOW_COAST_MARGIN_STRENGTH
  pitch_bonus = float(np.clip(abs(pitch) / PITCH_STOP_DISTANCE_MAX, 0.0, 1.0)) * FOLLOW_COAST_MARGIN_PITCH
  return base + pitch_bonus


# =========================================================
# L3 comfort: cruise coast (no lead, above set speed)
# =========================================================
class CoastingLogic:
  def __init__(self):
    self.active = False
    self.coast_strength = 1.0
    self.current_max_offset = 0.0
    self.coast_v_cruise = 0.0
    self.coast_v_upper = 0.0
    self._last_lead_time = 0.0

  def check_emergency(self, lead, v_ego, current_time):
    if not lead or not lead.status:
      return False

    lead_ttc = compute_lead_ttc(lead, v_ego)
    if lead_ttc is None:
      return False

    relative_speed = v_ego - lead.vLead
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    if (lead_ttc < EMERGENCY_TTC) or \
       (relative_speed > EMERGENCY_RELATIVE_SPEED) or \
       (lead.dRel < min_dist_for_speed and relative_speed > 0):
      self._last_lead_time = current_time
      return True
    return False

  def update_states(self, enabled, user_ctrl_lon, v_ego, v_cruise, current_pitch, dtsc_is_active,
                    current_time, lead, t_follow, pitch):
    if not enabled:
      self.active = False
      self.coast_strength = 0.0
      return

    has_lead = lead is not None and lead.status
    self.coast_strength = compute_coast_strength(lead, v_ego, t_follow, pitch) if has_lead else 1.0

    if current_pitch < PITCH_DOWNHILL_THRESHOLD:
      self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
      self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH

    self.coast_v_cruise = v_cruise
    self.coast_v_upper = v_cruise + (self.current_max_offset / 3.6)
    lower_bound = v_cruise - CRUISE_SPEED_ENTRY_TOLERANCE
    is_in_coast_window = (v_ego >= lower_bound and v_ego < self.coast_v_upper)
    in_cooldown = (current_time - self._last_lead_time) < LEAD_COOLDOWN_TIME

    self.active = (not has_lead and
                   not dtsc_is_active and
                   current_pitch <= PITCH_UPHILL_THRESHOLD and
                   not user_ctrl_lon and
                   not in_cooldown and
                   is_in_coast_window and
                   self.coast_strength > 0.0)

  def process_trajectory(self, a_desired_trajectory, pitch):
    if not self.active or pitch is None:
      return a_desired_trajectory

    traj = np.copy(a_desired_trajectory)
    if np.min(traj) < EMERGENCY_DECEL_THRESHOLD:
      self.active = False
      return a_desired_trajectory

    a_floor = get_coast_accel(pitch)
    return np.maximum(traj, a_floor)

  def process_v_trajectory(self, v_desired_trajectory, v_ego):
    if not self.active:
      return v_desired_trajectory

    v_min_target = max(self.coast_v_cruise, v_ego)
    return np.clip(np.maximum(v_desired_trajectory, v_min_target), self.coast_v_cruise, self.coast_v_upper)


# =========================================================
# L3 comfort: follow coast (low speed, A < dRel < A + margin)
# =========================================================
class FollowCoastLogic:
  def __init__(self):
    self.active = False
    self._last_exit_time = 0.0

  def update_states(self, enabled, has_lead, lead, v_ego, t_follow, pitch, strength,
                    dtsc_is_active, user_ctrl_lon, current_time):
    if not enabled or not has_lead or dtsc_is_active or user_ctrl_lon or pitch is None:
      if self.active:
        self._last_exit_time = current_time
      self.active = False
      return

    if (current_time - self._last_exit_time) < FOLLOW_COAST_COOLDOWN_TIME:
      self.active = False
      return

    safe_a = compute_safe_distance_A(v_ego, t_follow, pitch)
    margin = compute_follow_margin(strength, pitch)
    lead_obstacle = lead.dRel + get_stopped_equivalence_factor(lead.vLead)
    ttc = compute_lead_ttc(lead, v_ego)
    is_lead_braking = lead.aLeadK < FOLLOW_COAST_LEAD_BRAKE

    if self.active:
      should_exit = (
        lead_obstacle <= safe_a or
        (ttc is not None and ttc < SOFT_HOLD_TTC_THRESHOLD) or
        is_lead_braking or
        v_ego >= FOLLOW_COAST_SPEED_MAX or
        strength <= 0.0
      )
      if should_exit:
        self.active = False
        self._last_exit_time = current_time
    else:
      in_band = safe_a < lead_obstacle < safe_a + margin
      should_enter = (
        in_band and
        v_ego < FOLLOW_COAST_SPEED_MAX and
        strength > 0.0 and
        (ttc is None or ttc >= SOFT_HOLD_TTC_THRESHOLD) and
        not is_lead_braking
      )
      if should_enter:
        self.active = True

  def process_trajectory(self, a_desired_trajectory, pitch):
    if not self.active or pitch is None:
      return a_desired_trajectory

    traj = np.copy(a_desired_trajectory)
    if np.min(traj) < EMERGENCY_DECEL_THRESHOLD:
      self.active = False
      return a_desired_trajectory

    a_floor = get_coast_accel(pitch)
    return np.maximum(traj, a_floor)


# =========================================================
# L3 comfort: SoftHold (MPC_FOLLOW only)
# =========================================================
class SoftHoldLogic:
  def __init__(self):
    self._soft_hold_factor = 1.0
    self._vrel_high_start_time = 0.0
    self._vrel_high_active = False
    self._last_lead_time = 0.0
    self._last_target_factor = 1.0
    self._last_soft_hold_accel = 0.0
    self.accel_intent_counter = 0
    self.intent_accelerating = False
    self._accel_intent_strength = 0.0
    self._ratio_hysteresis_state = False
    self._cancel_filter = 0.0
    self._target_factor_smooth = 1.0
    self._last_stable_cancel_state = False
    self._state_change_time = 0.0

  def process_trajectory(self, a_desired_trajectory, v_ego, lead, current_pitch, t_follow):
    should_cancel_soft_hold = False
    current_time = time.monotonic()

    recent_trajectory = a_desired_trajectory[:TRAJECTORY_HORIZON]
    has_valid_lead = lead is not None and lead.status

    v_ratio = max(0.0, min((v_ego - INTENT_V_LOW) / (INTENT_V_HIGH - INTENT_V_LOW), 1.0))
    dynamic_intent_frames = int(round(INTENT_FRAMES_LOW + v_ratio * (INTENT_FRAMES_HIGH - INTENT_FRAMES_LOW)))

    moment_accel = sum(1 for a in recent_trajectory if a > 0.05) >= INTENT_LOOKAHEAD and (lead.vRel > 0.05 if has_valid_lead else True)

    target_factor = 1.0
    v_ego_kph = v_ego * 3.6
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)
    is_lead_braking_strict = False
    skip_state_2 = False

    if not has_valid_lead:
      self._vrel_high_active = False
      decrement = 1.0 / max(dynamic_intent_frames, 1)
      self._accel_intent_strength = max(0.0, self._accel_intent_strength - decrement)
      if self._accel_intent_strength < 0.1:
        self.intent_accelerating = False
        self.accel_intent_counter = 0

      if (current_time - self._last_lead_time) < 0.5:
        if self._last_soft_hold_accel >= 0.0:
          target_factor = self._last_target_factor
          current_soft_hold_accel = self._last_soft_hold_accel
        else:
          target_factor = 0.0
          current_soft_hold_accel = 0.0
        should_cancel_soft_hold = False
        skip_state_2 = True
      else:
        should_cancel_soft_hold = True
        skip_state_2 = True
    else:
      self._last_lead_time = current_time

      if moment_accel:
        increment = 1.0 / max(dynamic_intent_frames, 1)
        self._accel_intent_strength = min(1.0, self._accel_intent_strength + increment)
        self.accel_intent_counter += 1
      else:
        decrement = 1.0 / max(dynamic_intent_frames, 1)
        self._accel_intent_strength = max(0.0, self._accel_intent_strength - decrement)
        self.accel_intent_counter = 0

      if self._accel_intent_strength > 0.7:
        self.intent_accelerating = True
      elif self._accel_intent_strength < 0.3:
        self.intent_accelerating = False

      if self.intent_accelerating:
        cancel_probability = min(1.0, self._accel_intent_strength * 1.5)
        self._cancel_filter = 0.8 * self._cancel_filter + 0.2 * cancel_probability
        if self._cancel_filter > 0.5:
          should_cancel_soft_hold = True

      if lead.vRel > 1.0:
        if not self._vrel_high_active:
          self._vrel_high_active = True
          self._vrel_high_start_time = current_time
        elif (current_time - self._vrel_high_start_time) > VREL_DEBOUNCE_TIME:
          should_cancel_soft_hold = True
      else:
        self._vrel_high_active = False

      if current_pitch > SOFT_HOLD_PITCH_MAX:
        should_cancel_soft_hold = True

    if not skip_state_2:
      is_lead_stopped = (lead.vLead < 1.0) and (lead.vRel <= 0.3)

      if v_ego_kph <= 10.0:
        is_lead_braking_strict = (lead.aLeadK < -0.1 or is_lead_stopped) and (lead.vRel < 0.5)
      elif v_ego_kph <= 30.0:
        is_lead_braking_strict = (lead.aLeadK < -0.5 or is_lead_stopped) and (lead.vRel < 0.5)
      elif v_ego_kph <= 40.0:
        is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
      else:
        is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

      current_ttc = compute_lead_ttc(lead, v_ego)
      ratio = compute_lead_ratio(lead, v_ego, t_follow, current_pitch)

      if not should_cancel_soft_hold:
        if ratio > RATIO_ENTER_THRESHOLD:
          self._ratio_hysteresis_state = True
        elif ratio < RATIO_EXIT_THRESHOLD:
          self._ratio_hysteresis_state = False

        if self._ratio_hysteresis_state:
          should_cancel_soft_hold = True

    if should_cancel_soft_hold != self._last_stable_cancel_state:
      if self._state_change_time == 0.0:
        self._state_change_time = current_time
      elif (current_time - self._state_change_time) > (SOFT_HOLD_HYSTERESIS_TIME / 2):
        self._last_stable_cancel_state = should_cancel_soft_hold
        self._state_change_time = 0.0
    else:
      self._state_change_time = 0.0

    should_cancel_soft_hold = self._last_stable_cancel_state

    if should_cancel_soft_hold:
      target_factor = 1.0
      alpha = 0.60 if self.intent_accelerating else 0.30

    elif not skip_state_2:
      distance_factor = 1.0
      if current_pitch <= SOFT_HOLD_PITCH_MAX:
        if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
          distance_factor = 0.0

      v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
      target_factor = max(distance_factor, v_rel_factor)

      if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and is_lead_braking_strict:
        if current_pitch > SOFT_HOLD_PITCH_START:
          smooth_factor = float(np.interp(current_pitch, [SOFT_HOLD_PITCH_START, SOFT_HOLD_PITCH_MAX], [0.0, 1.0]))
          target_factor = smooth_factor
          current_soft_hold_accel = current_soft_hold_accel * smooth_factor
        else:
          if is_lead_stopped:
            current_soft_hold_accel = float(np.interp(v_ego_kph, [0.0, 150.0], [0.0, -0.30]))
          elif v_ego_kph >= 50.0:
            if lead.vRel < -0.1 and lead.aLeadK <= -1.5:
              dynamic_brake = lead.aLeadK * 0.30
              current_soft_hold_accel = np.clip(dynamic_brake, -1.0, 0.0)
            else:
              current_soft_hold_accel = 0.0
          else:
            current_soft_hold_accel = 0.0
          target_factor = 0.0

      alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20
    else:
      alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20

    self._last_target_factor = target_factor
    self._last_soft_hold_accel = current_soft_hold_accel

    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor
    self._target_factor_smooth = (1.0 - TARGET_FACTOR_FILTER_ALPHA) * self._target_factor_smooth + TARGET_FACTOR_FILTER_ALPHA * self._soft_hold_factor

    traj = np.copy(a_desired_trajectory)
    if self._target_factor_smooth < 0.99:
      hold_strength = 1.0 - self._target_factor_smooth
      dynamic_limit = np.maximum(traj, 0.0) * self._target_factor_smooth + current_soft_hold_accel * hold_strength
      blend_factor = 0.5
      exceeds_mask = traj > dynamic_limit
      traj = np.where(
        exceeds_mask,
        dynamic_limit * blend_factor + traj * (1.0 - blend_factor),
        traj
      )

    return traj


# =========================================================
# L3 comfort facade
# =========================================================
class ACM:
  def __init__(self):
    self.enabled = False
    self.current_pitch = 0.0
    self.comfort_state = ComfortState.OFF
    self.personality = log.LongitudinalPersonality.standard
    self._dtsc_is_active = False
    self._scc_is_active = False
    self._mode = 'acc'

    self.coasting = CoastingLogic()
    self.follow_coast = FollowCoastLogic()
    self.soft_hold = SoftHoldLogic()

  @property
  def active(self):
    return self.comfort_state in (ComfortState.CRUISE_COAST, ComfortState.FOLLOW_COAST)

  @property
  def comfort_state_capnp(self):
    return COMFORT_STATE_TO_CAPNP.get(self.comfort_state, AcmComfortState.off)

  def _blocked(self) -> bool:
    # SCC curve slowing always wins; ACM works in both classic ACC and Experimental/E2E.
    return (not self.enabled) or self._scc_is_active

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard,
                    dtsc_is_active=False, scc_active=False, road_pitch: float | None = None):
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active
    self._scc_is_active = scc_active
    self._mode = mode

    if not self.enabled or road_pitch is None or scc_active:
      self.comfort_state = ComfortState.OFF
      self.coasting.active = False
      self.coasting.coast_strength = 0.0
      self.follow_coast.active = False
      if road_pitch is not None:
        self.current_pitch = road_pitch
      return

    self.current_pitch = road_pitch
    current_time = time.monotonic()
    lead = rs.leadOne
    has_lead = lead is not None and lead.status
    t_follow = get_T_FOLLOW(personality)
    strength = compute_coast_strength(lead, v_ego, t_follow, road_pitch) if has_lead else 1.0

    if self.coasting.check_emergency(lead, v_ego, current_time):
      self.comfort_state = ComfortState.MPC_FOLLOW
      self.coasting.active = False
      self.coasting.coast_strength = 0.0
      self.follow_coast.active = False
      return

    self.coasting.update_states(
      self.enabled, user_ctrl_lon, v_ego, v_cruise, self.current_pitch, dtsc_is_active,
      current_time, lead, t_follow, road_pitch)

    self.follow_coast.update_states(
      self.enabled, has_lead, lead, v_ego, t_follow, road_pitch, strength,
      dtsc_is_active, user_ctrl_lon, current_time)

    if self.follow_coast.active:
      self.comfort_state = ComfortState.FOLLOW_COAST
    elif self.coasting.active:
      self.comfort_state = ComfortState.CRUISE_COAST
    else:
      self.comfort_state = ComfortState.MPC_FOLLOW

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None, t_follow=None,
                                    road_pitch: float | None = None):
    if self._blocked():
      return a_desired_trajectory

    if t_follow is None:
      t_follow = get_T_FOLLOW(self.personality)

    pitch = road_pitch if road_pitch is not None else self.current_pitch

    if self.comfort_state == ComfortState.CRUISE_COAST:
      return self.coasting.process_trajectory(a_desired_trajectory, pitch)

    if self.comfort_state == ComfortState.FOLLOW_COAST:
      return self.follow_coast.process_trajectory(a_desired_trajectory, pitch)

    return self.soft_hold.process_trajectory(a_desired_trajectory, v_ego, lead, pitch, t_follow)

  def update_v_desired_trajectory(self, v_desired_trajectory, v_ego=0.0):
    if self._blocked():
      return v_desired_trajectory

    if self.comfort_state == ComfortState.CRUISE_COAST:
      return self.coasting.process_v_trajectory(v_desired_trajectory, v_ego)

    return v_desired_trajectory

  def apply_to_accel(self, a_target: float, v_ego: float = 0.0, lead=None, t_follow: float | None = None,
                     road_pitch: float | None = None) -> float:
    """
    Post-blend ACM: apply coast / soft-hold to the final accel command.

    Needed for Experimental/E2E where output is min(e2e, mpc) — modifying only the MPC
    trajectory is not enough when the model accel wins the min().
    """
    if self._blocked() or self.comfort_state == ComfortState.OFF:
      return float(a_target)

    if t_follow is None:
      t_follow = get_T_FOLLOW(self.personality)

    pitch = road_pitch if road_pitch is not None else self.current_pitch
    a_target = float(a_target)

    # Never fight a hard brake
    if a_target < EMERGENCY_DECEL_THRESHOLD:
      return a_target

    if self.comfort_state in (ComfortState.CRUISE_COAST, ComfortState.FOLLOW_COAST):
      return float(max(a_target, get_coast_accel(pitch)))

    # Soft hold / MPC follow comfort on a single-sample trajectory
    traj = self.soft_hold.process_trajectory(np.array([a_target], dtype=float), v_ego, lead, pitch, t_follow)
    return float(traj[0])
