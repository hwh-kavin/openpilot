"""Adaptive lateral curvature lead from the model plan.

Larger predicted future curvature → larger sample lead time, so the
curvature command starts climbing earlier when a hard curve is ahead.
Mild curves get ~zero lead. No user toggle.
"""
from __future__ import annotations

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_curvature_from_plan
from openpilot.selfdrive.modeld.constants import ModelConstants

# Horizon used to measure upcoming curve severity
_LOOKAHEAD_T_MIN = 0.25  # s
_LOOKAHEAD_T_MAX = 2.0   # s

# Match modeld: lat_action_t = lat_delay + frame_delay + action_delay
_MODEL_ACTION_EXTRA = DT_MDL + DT_MDL / 2  # ~0.075 s

# Lead time vs predicted peak |κ| (1/m). Mild curves → 0; sharp → up to max.
# Absolute κ breakpoints keep lead proportional to predicted curvature size.
_KAPPA_LEAD_BP = [0.0010, 0.0025, 0.0050]  # 1/m
_LEAD_T_V = [0.0, 0.18, 0.35]              # s

# Also scale by how hard the peak is vs the car's usable κ limit at speed,
# so low-speed large-κ and high-speed a_lat-limited cases both lead more.
_RATIO_LEAD_BP = [0.70, 1.00, 1.40]  # κ_peak / κ_lim
_RATIO_LEAD_V = [0.0, 0.15, 0.35]    # s

_MAX_LEAD_T = 0.40  # s hard cap
_MAX_EXIT_LEAD_T = 0.32  # s hard cap for post-apex unwind
_MIN_SPEED = 1.0    # m/s

# Detect plan κ falling after an early peak (past apex)
_EXIT_T_PEAK_MAX = 0.55   # s — peak must be this early to count as exiting
_EXIT_KAPPA_MIN = 0.0015    # 1/m — ignore very mild bends
_EXIT_KAPPA_DROP_MIN = 0.0008  # 1/m — plan must show meaningful κ reduction ahead
_EXIT_LEAD_BP = [0.0025, 0.0050, 0.0090]  # peak |κ| (1/m)
_EXIT_LEAD_T_V = [0.08, 0.15, 0.28]       # s extra sample lead


def _plan_peak_curvature(orientation_rate_z, velocity_x, v_ego: float) -> float:
  # Convert first: pycapnp DynamicList does not support Python slicing.
  yaw_rate = np.asarray(orientation_rate_z, dtype=float)
  n = min(len(yaw_rate), len(ModelConstants.T_IDXS))
  if n < 2:
    return 0.0

  t = np.asarray(ModelConstants.T_IDXS[:n], dtype=float)
  yaw_rate = yaw_rate[:n]
  if len(velocity_x) >= n:
    speed = np.maximum(np.asarray(velocity_x, dtype=float)[:n], _MIN_SPEED)
  else:
    speed = np.full(n, max(v_ego, _MIN_SPEED), dtype=float)

  kappa = np.abs(yaw_rate) / speed
  mask = (t >= _LOOKAHEAD_T_MIN) & (t <= _LOOKAHEAD_T_MAX)
  if not np.any(mask):
    return float(np.max(kappa))
  return float(np.max(kappa[mask]))


def _kappa_limit(v_ego: float, max_lat_accel: float, max_curvature: float) -> float:
  v = max(v_ego, _MIN_SPEED)
  return float(min(max_curvature, max_lat_accel / (v * v)))


def curvature_lead_time(kappa_peak: float, v_ego: float,
                        max_lat_accel: float = 2.4,
                        max_curvature: float = 0.02) -> float:
  """Lead seconds from predicted peak curvature (and saturation ratio)."""
  if kappa_peak <= 0.0:
    return 0.0

  lead_from_kappa = float(np.interp(kappa_peak, _KAPPA_LEAD_BP, _LEAD_T_V))

  kappa_lim = _kappa_limit(v_ego, max_lat_accel, max_curvature)
  ratio = kappa_peak / max(kappa_lim, 1e-6)
  lead_from_ratio = float(np.interp(ratio, _RATIO_LEAD_BP, _RATIO_LEAD_V))

  # Take the more aggressive of the two so "larger κ → more lead" holds
  # both in absolute curvature and relative to the speed-dependent limit.
  return float(min(_MAX_LEAD_T, max(lead_from_kappa, lead_from_ratio)))


def apply_curvature_lead(model_v2, v_ego: float, base_curvature: float, lat_delay: float,
                         max_lat_accel: float = 2.4, max_curvature: float = 0.02) -> float:
  """Resample plan curvature with lead; fall back to base_curvature if plan missing."""
  # Convert first: pycapnp DynamicList does not support Python slicing.
  yaws = np.asarray(model_v2.orientation.z, dtype=float)
  yaw_rates = np.asarray(model_v2.orientationRate.z, dtype=float)
  if len(yaws) < 2 or len(yaw_rates) < 2:
    return float(base_curvature)

  kappa_peak = _plan_peak_curvature(yaw_rates, model_v2.velocity.x, v_ego)
  lead_t = curvature_lead_time(kappa_peak, v_ego, max_lat_accel, max_curvature)
  if lead_t <= 1e-4:
    return float(base_curvature)

  # Sample further ahead than modeld's lat_action_t by lead_t only.
  action_t = max(lat_delay + _MODEL_ACTION_EXTRA + lead_t, 1e-3)
  n = min(len(yaws), len(yaw_rates), len(ModelConstants.T_IDXS))
  return float(get_curvature_from_plan(
    yaws[:n],
    yaw_rates[:n],
    ModelConstants.T_IDXS[:n],
    v_ego,
    action_t,
  ))


def apply_curvature_exit_lead(model_v2, v_ego: float, base_curvature: float, lat_delay: float,
                              max_lat_accel: float = 2.4, max_curvature: float = 0.02) -> float:
  """After apex, sample plan further ahead where |κ| is smaller to unwind earlier."""
  yaw_rates = np.asarray(model_v2.orientationRate.z, dtype=float)
  if len(yaw_rates) < 2:
    return float(base_curvature)

  n = min(len(yaw_rates), len(ModelConstants.T_IDXS))
  t = np.asarray(ModelConstants.T_IDXS[:n], dtype=float)
  yaw_rate = yaw_rates[:n]
  if len(model_v2.velocity.x) >= n:
    speed = np.maximum(np.asarray(model_v2.velocity.x, dtype=float)[:n], _MIN_SPEED)
  else:
    speed = np.full(n, max(v_ego, _MIN_SPEED), dtype=float)

  kappa = np.abs(yaw_rate) / speed
  mask = (t >= _LOOKAHEAD_T_MIN) & (t <= _LOOKAHEAD_T_MAX)
  if not np.any(mask):
    return float(base_curvature)

  kappa_w = kappa[mask]
  t_w = t[mask]
  kappa_peak = float(np.max(kappa_w))
  if kappa_peak < _EXIT_KAPPA_MIN:
    return float(base_curvature)

  t_peak = float(t_w[int(np.argmax(kappa_w))])
  if t_peak > _EXIT_T_PEAK_MAX:
    return float(base_curvature)

  i_now = int(np.argmin(np.abs(t - 0.12)))
  kappa_now = float(kappa[i_now])
  t_future = min(t_peak + 0.35, float(t[n - 1]))
  i_future = int(np.argmin(np.abs(t - t_future)))
  if kappa_now - float(kappa[i_future]) < _EXIT_KAPPA_DROP_MIN:
    return float(base_curvature)

  exit_lead_t = float(np.interp(kappa_peak, _EXIT_LEAD_BP, _EXIT_LEAD_T_V))
  exit_lead_t = min(exit_lead_t, _MAX_EXIT_LEAD_T)
  if exit_lead_t <= 1e-4:
    return float(base_curvature)

  yaws = np.asarray(model_v2.orientation.z, dtype=float)
  if len(yaws) < 2:
    return float(base_curvature)

  action_t = max(lat_delay + _MODEL_ACTION_EXTRA + exit_lead_t, 1e-3)
  n = min(len(yaws), len(yaw_rates), len(ModelConstants.T_IDXS))
  led_curvature = float(get_curvature_from_plan(
    yaws[:n],
    yaw_rates[:n],
    ModelConstants.T_IDXS[:n],
    v_ego,
    action_t,
  ))

  # Only apply when exit lead reduces |κ| vs the entry-adjusted command
  if abs(led_curvature) + 1e-6 < abs(base_curvature):
    return led_curvature
  return float(base_curvature)
