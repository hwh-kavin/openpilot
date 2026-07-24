import numpy as np
import pytest

from cereal import log
from openpilot.common.constants import CV
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  apply_lat_capability_v_cap,
  apply_model_uncertainty_v_cap,
  build_plan_kappa_traj,
  compute_actual_lat_accel,
  compute_curve_lat_acc_uncertainty,
  compute_scc_curve_v_target,
  compute_scc_passable_speed,
  compute_steer_angle_lat_accel,
  get_scc_enter_lat_acc_th,
  get_scc_lat_accel_max,
  kappa_from_steer_angle,
  plan_curve_speed_from_kappa_traj,
)


@pytest.mark.parametrize("personality, a_lat, enter_th", [
  (log.LongitudinalPersonality.relaxed, 1.35, 1.05),
  (log.LongitudinalPersonality.standard, 1.65, 1.25),
  (log.LongitudinalPersonality.aggressive, 2.05, 1.40),
])
def test_scc_personality_limits(personality, a_lat, enter_th):
  assert get_scc_lat_accel_max(personality) == pytest.approx(a_lat)
  assert get_scc_enter_lat_acc_th(personality) == pytest.approx(enter_th)


def test_build_plan_kappa_traj():
  yaw = np.array([0.5, 1.0, 1.5])
  vel = np.array([25.0, 25.0, 25.0])
  kappa = build_plan_kappa_traj(yaw, vel)
  assert kappa == pytest.approx(yaw / 25.0)


def test_kappa_from_steer_angle():
  # δ_rad / (SR * WB)
  kappa = kappa_from_steer_angle(14.8 * 2.7 * (180.0 / np.pi), 0.0, 14.8, 2.7)
  assert kappa == pytest.approx(1.0, rel=1e-3)


def test_plan_sharp_curve_ahead_lowers_v():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  a_lat = get_scc_lat_accel_max(personality)
  # Sharp κ ahead within braking distance
  kappa = np.array([0.0005, 0.0005, 0.008, 0.008])
  pos = np.array([0.0, 40.0, 80.0, 120.0])
  t = np.array([0.0, 1.5, 3.0, 4.5])
  v_t, a_t, peak_k, peak_i, has_c = plan_curve_speed_from_kappa_traj(
    v_ego, 0.0, kappa, pos, t, kappa_now=0.0, personality=personality, min_v=MIN_V)
  assert has_c
  assert peak_k == pytest.approx(0.008)
  assert v_t < v_ego
  assert v_t == pytest.approx(max(MIN_V, (a_lat / 0.008) ** 0.5), rel=1e-3)
  assert a_t < 0.0


def test_plan_mild_curve_no_constraint():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  # Very mild κ → a_y = κ v² ≈ 0.39 < early enter th
  kappa = np.full(10, 0.0005)
  pos = np.linspace(0.0, 200.0, 10)
  t = np.linspace(0.0, 8.0, 10)
  v_t, a_t, peak_k, _, has_c = plan_curve_speed_from_kappa_traj(
    v_ego, 0.0, kappa, pos, t, kappa_now=0.0, personality=personality, min_v=MIN_V)
  assert not has_c
  assert v_t == pytest.approx(v_ego)
  assert peak_k == pytest.approx(0.0005)


def test_plan_current_steer_kappa_constrains():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  a_lat = get_scc_lat_accel_max(personality)
  kappa_now = 0.01  # already steering hard
  kappa = np.full(5, 0.0002)
  pos = np.linspace(0.0, 100.0, 5)
  t = np.linspace(0.0, 4.0, 5)
  v_t, a_t, _, _, has_c = plan_curve_speed_from_kappa_traj(
    v_ego, 0.0, kappa, pos, t, kappa_now=kappa_now, personality=personality, min_v=MIN_V)
  assert has_c
  assert v_t == pytest.approx(max(MIN_V, (a_lat / kappa_now) ** 0.5), rel=1e-3)
  assert a_t < 0.0


def test_plan_far_curve_outside_brake_distance_ignored():
  v_ego = 20.0
  personality = log.LongitudinalPersonality.standard
  # Sharp but very far — beyond comfort brake reach at this speed
  kappa = np.array([0.0002, 0.01])
  pos = np.array([0.0, 500.0])
  t = np.array([0.0, 9.0])
  v_t, _, _, _, has_c = plan_curve_speed_from_kappa_traj(
    v_ego, 0.0, kappa, pos, t, kappa_now=0.0, personality=personality, min_v=MIN_V)
  assert not has_c
  assert v_t == pytest.approx(v_ego)


def test_scc_passable_speed_formula():
  v_ego = 27.8
  max_pred = 2.5
  a_lat = get_scc_lat_accel_max(log.LongitudinalPersonality.standard)
  expected = v_ego * (a_lat / max_pred) ** 0.5
  assert compute_scc_passable_speed(v_ego, max_pred, log.LongitudinalPersonality.standard) == pytest.approx(expected)


def test_scc_curve_v_target_sharp_turn():
  v_ego = 27.8
  max_pred = 2.5
  v_relaxed = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.relaxed)
  v_standard = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard)
  v_aggressive = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.aggressive)
  assert v_relaxed < v_standard < v_aggressive < v_ego


def test_min_v_floor_on_sharp_kappa_plan():
  v_ego = 27.8
  personality = log.LongitudinalPersonality.standard
  kappa = np.array([0.5, 0.5])  # extreme
  pos = np.array([10.0, 20.0])
  t = np.array([0.5, 1.0])
  v_t, _, _, _, has_c = plan_curve_speed_from_kappa_traj(
    v_ego, 0.0, kappa, pos, t, kappa_now=0.0, personality=personality, min_v=MIN_V)
  assert has_c
  assert v_t >= MIN_V
  assert MIN_V == pytest.approx(30.0 * CV.KPH_TO_MS)


def test_steer_angle_lat_accel_exceeds_path_when_steering_deeper():
  v_ego = 20.0
  steer_ratio = 14.8
  wheelbase = 2.7
  a_path = compute_actual_lat_accel(v_ego, 0.01)
  a_steer = compute_steer_angle_lat_accel(v_ego, 90.0, 0.0, steer_ratio, wheelbase)
  assert a_steer > a_path


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


def test_model_uncertainty_lowers_v_target():
  v_target = 25.0
  v_low_unc = apply_model_uncertainty_v_cap(v_target, 0.0, min_v=MIN_V)
  v_high_unc = apply_model_uncertainty_v_cap(v_target, 1.0, min_v=MIN_V)
  assert v_low_unc == pytest.approx(v_target)
  assert v_high_unc == pytest.approx(v_target * 0.82)
