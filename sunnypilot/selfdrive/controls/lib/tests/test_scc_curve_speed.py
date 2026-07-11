import numpy as np
import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  combine_scc_model_actual_lat_acc,
  compute_actual_lat_accel,
  compute_scc_curve_v_target,
  compute_scc_passable_speed,
  get_scc_abort_enter_lat_acc_th,
  get_scc_enter_lat_acc_th,
  get_scc_lat_accel_max,
)


@pytest.mark.parametrize("personality, a_lat, enter_th", [
  (log.LongitudinalPersonality.relaxed, 1.42, 1.05),
  (log.LongitudinalPersonality.standard, 1.75, 1.25),
  (log.LongitudinalPersonality.aggressive, 2.18, 1.40),
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
  assert v_standard == pytest.approx(min(v_ego, v_pass * 1.035), rel=1e-3)


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
  assert v_path == pytest.approx(min(v_ego, v_ego * (1.75 / 2.8) ** 0.5 * 1.035), rel=1e-3)


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
