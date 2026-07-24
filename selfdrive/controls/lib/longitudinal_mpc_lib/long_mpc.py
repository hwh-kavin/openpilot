#!/usr/bin/env python3
import os
import time
import numpy as np
from cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

if __name__ == '__main__':  # generating code
  from openpilot.third_party.acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1, LongitudinalPlanSource.cruise)

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0  # legacy default; flat stop gap uses STOP_DISTANCE_FLAT
STOP_DISTANCE_FLAT = 4.0
STANDSTILL_HEADWAY_SPEED = 0.3  # below this, use stop buffer only (no time-headway inflation)
PITCH_SMOOTH_ALPHA_UP = 0.30
PITCH_SMOOTH_ALPHA_DOWN = 0.05
CRUISE_MIN_ACCEL = -1.2
CRUISE_MAX_ACCEL = 1.6
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
MIN_X_LEAD_FACTOR = 0.5

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.5
  elif personality == log.LongitudinalPersonality.standard:
    return 1.0
  elif personality == log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")


# Ford + FordStockAccFusion: speed-based follow time gap (no gap-button / personality distance)
_FORD_AUTO_T_FOLLOW_V_KPH = [0., 20., 40., 60., 80., 100.]
_FORD_AUTO_T_FOLLOW = [1.20, 1.30, 1.40, 1.50, 1.60, 1.70]
_FORD_FOLLOW_BARS_HOLD_S = 1.0


def is_ford_auto_follow_gap(params, CP) -> bool:
  """True when Ford stock fusion should use speed-based t_follow instead of personality."""
  if not getattr(CP, 'openpilotLongitudinalControl', False):
    return False
  if 'FORD' not in CP.carFingerprint:
    return False
  try:
    return bool(params.get_bool("FordStockAccFusion"))
  except Exception:
    return False


def get_t_follow_auto(v_ego: float, standstill: bool = False) -> float:
  """Continuous t_follow from speed (m/s). Standstill uses minimum gap."""
  from opendbc.car.common.conversions import Conversions as CV
  if standstill:
    return float(_FORD_AUTO_T_FOLLOW[0])
  v_kph = max(0.0, v_ego * CV.MS_TO_KPH)
  return float(np.interp(v_kph, _FORD_AUTO_T_FOLLOW_V_KPH, _FORD_AUTO_T_FOLLOW))


def resolve_t_follow(v_ego: float, standstill: bool, personality, params, CP) -> float:
  if is_ford_auto_follow_gap(params, CP):
    return get_t_follow_auto(v_ego, standstill)
  return get_T_FOLLOW(personality)


def _t_follow_to_bars_target(t_follow: float) -> int:
  if t_follow <= 1.28:
    return 1
  if t_follow <= 1.42:
    return 2
  if t_follow <= 1.58:
    return 3
  return 4


class FordFollowBarsDisplay:
  """Smooth Ford IPC 1–4 bar display for auto t_follow."""

  def __init__(self):
    self.bars = 3
    self._last_change = 0.0

  def update(self, t_follow: float, standstill: bool) -> int:
    target = 1 if standstill else _t_follow_to_bars_target(t_follow)
    now = time.monotonic()
    if target != self.bars:
      if (now - self._last_change) >= _FORD_FOLLOW_BARS_HOLD_S or abs(target - self.bars) > 1:
        self.bars = target
        self._last_change = now
    return self.bars


def _personality_max_accel_vals(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return [1.0, 0.85, 0.65, 0.50]
  if personality == log.LongitudinalPersonality.aggressive:
    return [1.9, 1.45, 1.05, 0.75]
  return [1.6, 1.2, 0.8, 0.6]


def get_max_accel(v_ego, personality=log.LongitudinalPersonality.standard):
  return float(np.interp(v_ego, A_CRUISE_MAX_BP, _personality_max_accel_vals(personality)))


def get_cruise_max_accel(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.0
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.9
  return CRUISE_MAX_ACCEL


def get_cruise_min_accel(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return -1.0
  if personality == log.LongitudinalPersonality.aggressive:
    return -1.4
  return CRUISE_MIN_ACCEL


def get_accel_slew_rate(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.03
  if personality == log.LongitudinalPersonality.aggressive:
    return 0.08
  return 0.05


def get_start_accel(personality, base_start_accel: float) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    factor = 0.55
  elif personality == log.LongitudinalPersonality.aggressive:
    factor = 1.25
  else:
    factor = 1.0
  return float(base_start_accel * factor)


def get_scc_accel_scale(personality=log.LongitudinalPersonality.standard):
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.75
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.15
  return 1.0


# Home SUV curve speed limits (m/s²). v_corner = sqrt(a_lat / curvature).
def get_scc_lat_accel_max(personality=log.LongitudinalPersonality.standard) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.35
  if personality == log.LongitudinalPersonality.aggressive:
    return 2.05
  return 1.65


# Passable speed vs pure physics ( <1 = slightly deeper curve decel ).
_SCC_PASSABLE_SPEED_FACTOR = 1.00


def get_scc_enter_lat_acc_th(personality=log.LongitudinalPersonality.standard) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    return 1.05
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.40
  return 1.25


def get_scc_abort_enter_lat_acc_th(personality=log.LongitudinalPersonality.standard) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.90
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.22
  return 1.08


def get_scc_early_enter_lat_acc_th(personality=log.LongitudinalPersonality.standard) -> float:
  """Lower threshold on the far lookahead window to start slowing earlier."""
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.95
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.25
  return 1.10


def get_scc_early_abort_lat_acc_th(personality=log.LongitudinalPersonality.standard) -> float:
  if personality == log.LongitudinalPersonality.relaxed:
    return 0.82
  if personality == log.LongitudinalPersonality.aggressive:
    return 1.10
  return 0.98


def cap_vel_plan_for_scc(vel_plan: np.ndarray, v_ego: float) -> np.ndarray:
  """Cap model velocity so predicted lateral accel is not inflated above road speed."""
  v_ego = max(float(v_ego), 0.1)
  v_cap = v_ego * 1.08 + 1.5
  return np.minimum(np.asarray(vel_plan, dtype=np.float64), v_cap)


def compute_actual_lat_accel(v_ego: float, curvature: float) -> float:
  """Lateral acceleration from measured path curvature (steering / yaw). a = v²κ."""
  v_ego = max(float(v_ego), 0.1)
  return v_ego ** 2 * abs(float(curvature))


def compute_steer_angle_lat_accel(v_ego: float, steer_angle_deg: float, angle_offset_deg: float,
                                  steer_ratio: float, wheelbase: float) -> float:
  """Lateral accel from bicycle-model steering angle. Same form as limit_accel_in_turns."""
  v_ego = max(float(v_ego), 0.1)
  return v_ego ** 2 * kappa_from_steer_angle(steer_angle_deg, angle_offset_deg, steer_ratio, wheelbase)


_SCC_KAPPA_VEL_MIN = 1.0  # m/s, floor when converting yaw-rate → κ
_SCC_KAPPA_EPS = 1e-4
_SCC_CURVE_TARGET_ACCEL = -1.2  # m/s² comfort brake (scaled by personality)
_SCC_CURVE_DIST_OFFSET_S = 1.0  # s * v_corner extra distance margin
_SCC_CURVE_A_MIN = -1.2
_SCC_CURVE_A_MAX = 0.6
_SCC_CURVE_LOOKAHEAD_T = 10.0  # s, full model plan horizon


def kappa_from_steer_angle(steer_angle_deg: float, angle_offset_deg: float,
                           steer_ratio: float, wheelbase: float) -> float:
  """Bicycle-model curvature from measured steering wheel angle. κ = |δ| / (SR · WB)."""
  from openpilot.common.constants import CV
  steer_ratio = max(float(steer_ratio), 1e-3)
  wheelbase = max(float(wheelbase), 1e-3)
  angle_rad = (float(steer_angle_deg) - float(angle_offset_deg)) * CV.DEG_TO_RAD
  return abs(angle_rad) / (steer_ratio * wheelbase)


def build_plan_kappa_traj(yaw_rate_z, vel_plan, min_v: float = _SCC_KAPPA_VEL_MIN) -> np.ndarray:
  """Plan curvature trajectory. κ[i] = |yaw_rate[i]| / max(v[i], min_v) — same as curvature_lead."""
  yaw = np.abs(np.asarray(yaw_rate_z, dtype=np.float64).reshape(-1))
  vel = np.asarray(vel_plan, dtype=np.float64).reshape(-1)
  n = min(len(yaw), len(vel))
  if n == 0:
    return np.zeros(0, dtype=np.float64)
  return yaw[:n] / np.maximum(np.abs(vel[:n]), float(min_v))


def _scc_curve_brake_accel(personality=log.LongitudinalPersonality.standard) -> float:
  return float(_SCC_CURVE_TARGET_ACCEL * get_scc_accel_scale(personality))


def _scc_decel_reach_distance(v_ego: float, v_corner: float, a_brake: float) -> float:
  """Distance within which comfort braking must start to hit v_corner (plus time offset)."""
  v_ego = max(float(v_ego), 0.1)
  v_corner = max(float(v_corner), 0.0)
  if v_ego <= v_corner:
    return 0.0
  a = min(float(a_brake), -0.1)
  return (v_ego ** 2 - v_corner ** 2) / (2.0 * abs(a)) + v_corner * _SCC_CURVE_DIST_OFFSET_S


def plan_curve_speed_from_kappa_traj(
    v_ego: float,
    a_ego: float,
    kappa_traj,
    position_x,
    t_idxs,
    kappa_now: float,
    personality=log.LongitudinalPersonality.standard,
    min_v: float = 0.0,
    max_lookahead_t: float = _SCC_CURVE_LOOKAHEAD_T,
) -> tuple[float, float, float, int, bool]:
  """Plan curve speed from current + future curvature (κ-native, no steer-angle traj).

  Returns (v_target, a_target, peak_kappa, peak_idx, has_constraint).
  """
  a_lat = get_scc_lat_accel_max(personality)
  a_brake = _scc_curve_brake_accel(personality)
  a_min = float(_SCC_CURVE_A_MIN * get_scc_accel_scale(personality))
  v_ego = max(float(v_ego), 0.1)
  kappa_th = get_scc_early_enter_lat_acc_th(personality) / (v_ego ** 2)

  kappa = np.asarray(kappa_traj, dtype=np.float64).reshape(-1)
  pos = np.asarray(position_x, dtype=np.float64).reshape(-1)
  t = np.asarray(t_idxs, dtype=np.float64).reshape(-1)
  n = int(min(len(kappa), len(pos), len(t)))

  v_limit = v_ego
  peak_kappa = 0.0
  peak_idx = 0
  has_constraint = False
  binding_d = 0.0

  kappa_now = max(float(kappa_now), 0.0)
  if kappa_now > kappa_th:
    v_now = max(float(min_v), float(np.sqrt(a_lat / max(kappa_now, _SCC_KAPPA_EPS))))
    if v_now < v_limit:
      v_limit = v_now
      has_constraint = True
      binding_d = 0.0

  for i in range(n):
    if float(t[i]) > float(max_lookahead_t):
      break
    k = float(kappa[i])
    if k > peak_kappa:
      peak_kappa = k
      peak_idx = i
    if k <= kappa_th:
      continue
    v_corner = max(float(min_v), float(np.sqrt(a_lat / max(k, _SCC_KAPPA_EPS))))
    d = max(float(pos[i]), 0.0)
    if v_ego > v_corner and d <= _scc_decel_reach_distance(v_ego, v_corner, a_brake):
      if v_corner < v_limit:
        v_limit = v_corner
        has_constraint = True
        binding_d = d

  if not has_constraint:
    return v_ego, max(0.0, float(a_ego)), peak_kappa, peak_idx, False

  v_target = max(float(min_v), min(v_ego, v_limit))
  if v_ego <= v_target:
    a_cmd = min(float(_SCC_CURVE_A_MAX), max(0.0, float(a_ego)))
  elif binding_d < 1.0:
    a_cmd = a_brake
  else:
    a_cmd = (v_target ** 2 - v_ego ** 2) / (2.0 * max(binding_d, 1.0))
  a_cmd = float(np.clip(a_cmd, a_min, float(_SCC_CURVE_A_MAX)))
  return v_target, a_cmd, peak_kappa, peak_idx, True


_LAT_CAPABILITY_TRACKING_FACTOR = 0.95
_LAT_CAPABILITY_KAPPA_EPS = 1e-4

# Model curve uncertainty (orientationRate.zStd → lat-accel σ ≈ zStd * v).
_CURVE_UNCERTAINTY_GAIN = 1.0          # add 1σ to effective predicted lat accel
_CURVE_UNCERTAINTY_V_SCALE_MAX = 0.18  # up to 18% extra speed cut
_CURVE_UNCERTAINTY_REF = 1.0          # m/s² σ that maps to full scale-down


def compute_curve_lat_acc_uncertainty(z_std, vel_plan, mask=None) -> float:
  """Mean predicted lateral-accel std from yaw-rate std and planned speed."""
  z_std = np.asarray(z_std, dtype=np.float64).reshape(-1)
  vel = np.asarray(vel_plan, dtype=np.float64).reshape(-1)
  n = min(len(z_std), len(vel))
  if n == 0:
    return 0.0
  z_std = np.abs(z_std[:n])
  vel = np.abs(vel[:n])
  if mask is not None:
    m = np.asarray(mask, dtype=bool).reshape(-1)[:n]
    if np.any(m):
      z_std = z_std[m]
      vel = vel[m]
    else:
      return 0.0
  return float(np.mean(z_std * vel))


def inflate_lat_acc_with_uncertainty(lat_acc: float, lat_acc_unc: float,
                                     gain: float = _CURVE_UNCERTAINTY_GAIN) -> float:
  """Raise effective lat accel by a multiple of prediction uncertainty (safer v_corner)."""
  return max(0.0, float(lat_acc)) + float(gain) * max(0.0, float(lat_acc_unc))


def inflate_pred_lat_accels_with_uncertainty(predicted_lat_accels: np.ndarray, z_std, vel_plan,
                                             gain: float = _CURVE_UNCERTAINTY_GAIN) -> np.ndarray:
  """Per-horizon: a_eff = a_pred + gain * zStd * v."""
  pred = np.asarray(predicted_lat_accels, dtype=np.float64).copy()
  z_std = np.asarray(z_std, dtype=np.float64).reshape(-1)
  vel = np.asarray(vel_plan, dtype=np.float64).reshape(-1)
  n = min(len(pred), len(z_std), len(vel))
  if n == 0:
    return pred
  pred[:n] = pred[:n] + float(gain) * np.abs(z_std[:n]) * np.abs(vel[:n])
  return pred


def apply_model_uncertainty_v_cap(v_target: float, lat_acc_unc: float, min_v: float = 0.,
                                  ref_unc: float = _CURVE_UNCERTAINTY_REF,
                                  max_scale: float = _CURVE_UNCERTAINTY_V_SCALE_MAX) -> float:
  """Scale down curve speed as lat-accel prediction uncertainty rises."""
  u = float(np.clip(float(lat_acc_unc) / max(float(ref_unc), 1e-3), 0.0, 1.0))
  factor = 1.0 - float(max_scale) * u
  return max(float(min_v), float(v_target) * factor)


def apply_lat_capability_v_cap(v_target: float, v_ego: float, desired_curvature: float, curvature: float,
                               saturated: bool, personality=log.LongitudinalPersonality.standard,
                               min_v: float = 0.) -> float:
  """Lower curve speed when lateral demand exceeds OP capability / comfort a_lat_max."""
  v_target = max(float(v_target), 0.0)
  v_ego = max(float(v_ego), 0.1)
  a_lat_max = get_scc_lat_accel_max(personality)
  kappa_des = abs(float(desired_curvature))
  kappa_act = abs(float(curvature))
  a_des = v_ego ** 2 * kappa_des

  v_cap = v_target
  if saturated or a_des > a_lat_max:
    v_cap = float(np.sqrt(a_lat_max / max(kappa_des, _LAT_CAPABILITY_KAPPA_EPS)))
    if kappa_des > kappa_act * 1.15 and kappa_des > _LAT_CAPABILITY_KAPPA_EPS:
      v_cap *= _LAT_CAPABILITY_TRACKING_FACTOR

  return max(float(min_v), min(v_target, v_cap))


# While still mostly straight, trust most of the vision lat-accel prediction so curve
# speed can drop before measured steering builds. Actual still blends in to reject noise.
_SCC_MODEL_TRUST_STRAIGHT = 0.70


def combine_scc_model_actual_lat_acc(model_lat_acc: float, actual_lat_acc: float,
                                     personality=log.LongitudinalPersonality.standard) -> float:
  """Fuse model and steering-based lateral accel for SCC passable speed.

  When the vehicle is turning, actual steering can exceed a lagging model estimate.
  When the vehicle is still mostly straight, soft-blend model with actual (instead of a
  hard abort-threshold cap) so upcoming curves can slow earlier without fully trusting
  single-frame model spikes.
  """
  actual_th = get_scc_abort_enter_lat_acc_th(personality)
  model_lat_acc = max(float(model_lat_acc), 0.0)
  actual_lat_acc = max(float(actual_lat_acc), 0.0)
  if actual_lat_acc > actual_th:
    return max(model_lat_acc, actual_lat_acc)
  return (_SCC_MODEL_TRUST_STRAIGHT * model_lat_acc +
          (1.0 - _SCC_MODEL_TRUST_STRAIGHT) * actual_lat_acc)


def compute_scc_passable_speed(v_ego: float, max_pred_lat_acc: float,
                               personality=log.LongitudinalPersonality.standard,
                               min_v: float = 0.) -> float:
  """Max speed that can navigate a curve with predicted lateral accel max_pred_lat_acc."""
  a_lat = get_scc_lat_accel_max(personality)
  max_pred = max(float(max_pred_lat_acc), 1e-3)
  v_ref = max(float(v_ego), 0.1)
  return max(min_v, v_ref * float(np.sqrt(a_lat / max_pred)))


def compute_scc_curve_v_target(v_ego: float, max_pred_lat_acc: float,
                             personality=log.LongitudinalPersonality.standard,
                             min_v: float = 0.,
                             position_x: np.ndarray | None = None,
                             predicted_lat_accels: np.ndarray | None = None,
                             vel_plan: np.ndarray | None = None) -> float:
  """Return a speed cap for the curve. If v_ego is already at or below the passable speed, no decel."""
  v_ego = max(float(v_ego), 0.1)
  a_lat = get_scc_lat_accel_max(personality)
  v_limit = compute_scc_passable_speed(v_ego, max_pred_lat_acc, personality, min_v)

  curve_th = get_scc_early_enter_lat_acc_th(personality)
  if predicted_lat_accels is not None:
    pred = np.asarray(predicted_lat_accels, dtype=np.float64)
    if vel_plan is None:
      vel = np.full_like(pred, v_ego)
    else:
      vel = np.asarray(vel_plan[:len(pred)], dtype=np.float64)

    for a_pred, v_plan_pt in zip(pred, vel):
      if a_pred <= curve_th:
        continue
      v_pass = max(min_v, float(v_plan_pt) * np.sqrt(a_lat / a_pred))
      v_limit = min(v_limit, v_pass)

  v_limit *= _SCC_PASSABLE_SPEED_FACTOR

  if v_ego <= v_limit:
    return v_ego
  return max(min_v, min(v_ego, v_limit))


def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)


def get_stop_distance_for_pitch(pitch: float) -> float:
  """Return fixed stop distance regardless of pitch."""
  return STOP_DISTANCE_FLAT


class RoadPitchFilter:
  def __init__(self):
    self.pitch = 0.0
    self.initialized = False

  def reset(self):
    self.pitch = 0.0
    self.initialized = False

  def update(self, orientation_ned) -> float | None:
    if orientation_ned is None or len(orientation_ned) != 3:
      return None
    new_pitch = orientation_ned[1]
    if not self.initialized:
      self.pitch = new_pitch
      self.initialized = True
    else:
      alpha = PITCH_SMOOTH_ALPHA_UP if new_pitch > self.pitch else PITCH_SMOOTH_ALPHA_DOWN
      self.pitch = alpha * new_pitch + (1.0 - alpha) * self.pitch
    return self.pitch


def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def get_safe_obstacle_distance(v_ego, t_follow):
  stop_dist = STOP_DISTANCE_FLAT
  kinetic = (v_ego**2) / (2 * COMFORT_BRAKE)
  headway = t_follow * v_ego
  moving_dist = kinetic + headway + stop_dist
  try:
    from casadi import MX, SX, if_else
    if isinstance(v_ego, (SX, MX)):
      return if_else(v_ego >= STANDSTILL_HEADWAY_SPEED, moving_dist, stop_dist)
  except ImportError:
    pass
  v_arr = np.asarray(v_ego, dtype=float)
  return np.where(v_arr >= STANDSTILL_HEADWAY_SPEED, moving_dist, stop_dist)

def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # states
  x_ego, v_ego, a_ego = SX.sym('x_ego'), SX.sym('v_ego'), SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  a_prev = SX.sym('a_prev')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, a_prev, lead_t_follow, lead_danger_factor)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model

def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  a_prev = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - a_prev,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    self.solver.reset()

    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    self.yref = np.zeros((N+1, COST_DIM))

    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))

    self.last_cloudlog_t = 0
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    jerk_factor = get_jerk_factor(personality)
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
    cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
    constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    # 只有前车已停止时才强制最小停车距离
    if v_lead < 0.5:  # 前车速度<0.5m/s(约2km/h)视为停止
      min_x_lead = max(min_x_lead, STOP_DISTANCE_FLAT)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def update(self, radarstate, v_cruise, personality=log.LongitudinalPersonality.standard,
             pitch: float | None = None, t_follow: float | None = None):
    if t_follow is None:
      t_follow = get_T_FOLLOW(personality)
    v_ego = self.x0[1]

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
    # when the leads are no factor.
    cruise_min = get_cruise_min_accel(personality)
    cruise_max = get_cruise_max_accel(personality)
    v_lower = v_ego + (T_IDXS * cruise_min * 1.05)
    v_upper = v_ego + (T_IDXS * cruise_max * 1.05)
    v_cruise_clipped = np.clip(v_cruise * np.ones(N+1), v_lower, v_upper)
    cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(
      v_cruise_clipped, t_follow)

    x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
    self.source = MPC_SOURCES[np.argmin(x_obstacles[0])]

    self.yref[:,:] = 0.0
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR

    self.run()
    # FCW gate: high vision confidence, or a *moving* radar lead.
    # Stationary radar-only tracks (common Ford MRR clutter) previously satisfied
    # `or radar` and spammed FCW / stock IPC collision warnings.
    lead = radarstate.leadOne
    fcw_confident = lead.modelProb > 0.9 or (lead.radar and lead.vLead > 2.0)
    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            fcw_confident):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def run(self):
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
