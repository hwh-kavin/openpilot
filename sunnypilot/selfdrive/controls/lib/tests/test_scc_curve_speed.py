import numpy as np
import pytest

from cereal import log
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  apply_lat_capability_v_cap,
  apply_model_uncertainty_v_cap,
  combine_scc_model_actual_lat_acc,
  compute_actual_lat_accel,
  compute_curve_lat_acc_uncertainty,
  compute_scc_curve_v_target,
  compute_scc_passable_speed,
  compute_steer_angle_lat_accel,
  get_scc_abort_enter_lat_acc_th,
  get_scc_enter_lat_acc_th,
  get_scc_lat_accel_max,
  inflate_lat_acc_with_uncertainty,
  inflate_pred_lat_accels_with_uncertainty,
)


@pytest.mark.parametrize("personality, a_lat, enter_th", [
  (log.LongitudinalPersonality.relaxed, 1.35, 1.05),
  (log.LongitudinalPersonality.standard, 1.65, 1.25),
  (log.LongitudinalPersonality.aggressive, 2.05, 1.40),
])
def test_scc_personality_limits(personality, a_lat, enter_th):
  assert get_scc_lat_accel_max(personality) == pytest.approx(a_lat)
  assert get_scc_enter_lat_acc_th(personality) == pytest.approx(enter_th)


def test_scc_passable_speed_formula():
  v_ego = 27.8
  max_pred = 2.5
  a_lat = get_scc_lat_accel_max(log.LongitudinalPersonality.standard)
  expected = v_ego * (a_lat / max_pred) ** 0.5
  assert compute_scc_passable_speed(v_ego, max_pred, log.LongitudinalPersonality.standard) == pytest.approx(expected)


def test_scc_no_decel_when_below_passable_speed():
  v_ego = 25.0
  for max_pred in (0.5, 0.95, 1.0):
    v_out = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard)
    assert v_out == pytest.approx(v_ego)


def test_scc_curve_v_target_sharp_turn():
  v_ego = 27.8  # 100 km/h
  max_pred = 2.5
  v_relaxed = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.relaxed)
  v_standard = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard)
  v_aggressive = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.aggressive)

  assert v_relaxed < v_standard < v_aggressive < v_ego
  v_pass = compute_scc_passable_speed(v_ego, max_pred, log.LongitudinalPersonality.standard)
  assert v_standard == pytest.approx(min(v_ego, v_pass * 1.00), rel=1e-3)


def test_scc_path_scan_limits_for_upcoming_curve():
  v_ego = 27.8
  max_pred = 1.0  # current path is mild; v_ego is below passable speed for this segment
  n = 20
  pred = np.full(n, 0.5, dtype=np.float64)
  pred[5] = 2.8
  vel = np.full(n, v_ego, dtype=np.float64)
  x = np.linspace(0.0, 200.0, n)

  v_now = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard, min_v=0.0)
  v_path = compute_scc_curve_v_target(
    v_ego, max_pred, log.LongitudinalPersonality.standard, min_v=0.0,
    position_x=x, predicted_lat_accels=pred, vel_plan=vel)

  assert v_now == pytest.approx(v_ego)
  assert v_path < v_ego
  assert v_path == pytest.approx(min(v_ego, v_ego * (1.65 / 2.8) ** 0.5 * 1.00), rel=1e-3)


def test_combine_lat_acc_uses_steering_in_turn():
  personality = log.LongitudinalPersonality.standard
  combined = combine_scc_model_actual_lat_acc(1.0, 2.2, personality)
  assert combined == pytest.approx(2.2)


def test_combine_lat_acc_vetoes_model_when_straight():
  personality = log.LongitudinalPersonality.standard
  abort_th = get_scc_abort_enter_lat_acc_th(personality)
  combined = combine_scc_model_actual_lat_acc(2.5, 0.4, personality)
  assert combined == pytest.approx(abort_th)


def test_actual_steering_can_trigger_decel_when_model_lags():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  model_lat = 0.8
  actual_lat = compute_actual_lat_accel(v_ego, 0.05)
  lat_acc = combine_scc_model_actual_lat_acc(model_lat, actual_lat, personality)
  v_model_only = compute_scc_curve_v_target(v_ego, model_lat, personality)
  v_combined = compute_scc_curve_v_target(v_ego, lat_acc, personality)

  assert v_model_only == pytest.approx(v_ego)
  assert v_combined < v_ego


def test_min_v_floor_on_sharp_curve():
  v_ego = 27.8
  max_pred = 20.0  # extremely sharp predicted lat accel
  v_out = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard, min_v=MIN_V)
  assert v_out >= MIN_V
  assert MIN_V == pytest.approx(30.0 * CV.KPH_TO_MS)


def test_steer_angle_lat_accel_exceeds_path_when_steering_deeper():
  v_ego = 20.0
  steer_ratio = 14.8
  wheelbase = 2.7
  a_path = compute_actual_lat_accel(v_ego, 0.01)
  a_steer = compute_steer_angle_lat_accel(v_ego, 90.0, 0.0, steer_ratio, wheelbase)
  assert a_steer > a_path
  fused = max(a_path, a_steer)
  assert fused == pytest.approx(a_steer)


def test_lat_capability_cap_when_saturated():
  personality = log.LongitudinalPersonality.standard
  v_ego = 27.8
  v_target = 25.0
  desired_kappa = 0.05
  v_capped = apply_lat_capability_v_cap(
    v_target, v_ego, desired_kappa, 0.02, saturated=True, personality=personality, min_v=MIN_V)
  a_lat = get_scc_lat_accel_max(personality)
  expected = max(MIN_V, min(v_target, (a_lat / desired_kappa) ** 0.5 * 0.95))
  assert v_capped == pytest.approx(expected)
  assert v_capped < v_target
  assert v_capped >= MIN_V


def test_lat_capability_cap_when_desired_exceeds_a_lat_max():
  personality = log.LongitudinalPersonality.standard
  v_ego = 27.8
  a_lat = get_scc_lat_accel_max(personality)
  desired_kappa = (a_lat * 1.5) / (v_ego ** 2)
  v_target = v_ego
  v_capped = apply_lat_capability_v_cap(
    v_target, v_ego, desired_kappa, desired_kappa, saturated=False, personality=personality, min_v=MIN_V)
  expected = max(MIN_V, min(v_target, (a_lat / desired_kappa) ** 0.5))
  assert v_capped == pytest.approx(expected)
  assert v_capped < v_target


def test_lat_capability_no_cap_when_within_limits():
  personality = log.LongitudinalPersonality.standard
  v_ego = 20.0
  v_target = 18.0
  desired_kappa = 0.002
  v_capped = apply_lat_capability_v_cap(
    v_target, v_ego, desired_kappa, desired_kappa, saturated=False, personality=personality, min_v=MIN_V)
  assert v_capped == pytest.approx(v_target)

def test_curve_lat_acc_uncertainty_from_yaw_std():
  z_std = np.array([0.05, 0.05, 0.10])
  vel = np.array([20.0, 20.0, 20.0])
  unc = compute_curve_lat_acc_uncertainty(z_std, vel)
  assert unc == pytest.approx(np.mean([1.0, 1.0, 2.0]))


def test_inflate_lat_acc_with_uncertainty_raises_effective_pred():
  assert inflate_lat_acc_with_uncertainty(1.5, 0.5) == pytest.approx(2.0)


def test_inflate_pred_lat_accels_with_uncertainty():
  pred = np.array([1.0, 1.0, 1.0])
  z_std = np.array([0.1, 0.0, 0.2])
  vel = np.array([10.0, 10.0, 10.0])
  out = inflate_pred_lat_accels_with_uncertainty(pred, z_std, vel)
  assert out[0] == pytest.approx(2.0)
  assert out[1] == pytest.approx(1.0)
  assert out[2] == pytest.approx(3.0)


def test_model_uncertainty_lowers_v_target():
  v_target = 25.0
  v_low_unc = apply_model_uncertainty_v_cap(v_target, 0.0, min_v=MIN_V)
  v_high_unc = apply_model_uncertainty_v_cap(v_target, 1.0, min_v=MIN_V)
  assert v_low_unc == pytest.approx(v_target)
  assert v_high_unc == pytest.approx(v_target * 0.82)
  assert v_high_unc >= MIN_V


def test_higher_uncertainty_yields_lower_curve_speed():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  base_pred = 2.0
  v_certain = compute_scc_curve_v_target(v_ego, base_pred, personality, min_v=MIN_V)
  v_uncertain = compute_scc_curve_v_target(
    v_ego, inflate_lat_acc_with_uncertainty(base_pred, 0.8), personality, min_v=MIN_V)
  assert v_uncertain < v_certain
  assert v_uncertain >= MIN_V
