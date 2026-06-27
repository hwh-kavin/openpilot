import numpy as np
import pytest

from cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  apply_scc_lookahead_lead,
  compute_scc_curve_v_target,
  get_scc_enter_lat_acc_th,
  get_scc_lat_accel_max,
)


@pytest.mark.parametrize("personality, a_lat, enter_th", [
  (log.LongitudinalPersonality.relaxed, 1.35, 0.85),
  (log.LongitudinalPersonality.standard, 1.65, 1.00),
  (log.LongitudinalPersonality.aggressive, 2.05, 1.20),
])
def test_scc_personality_limits(personality, a_lat, enter_th):
  assert get_scc_lat_accel_max(personality) == pytest.approx(a_lat)
  assert get_scc_enter_lat_acc_th(personality) == pytest.approx(enter_th)


def test_scc_curve_v_target_sharp_turn():
  v_ego = 27.8  # 100 km/h
  max_pred = 2.5
  v_relaxed = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.relaxed)
  v_standard = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard)
  v_aggressive = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.aggressive)

  assert v_relaxed < v_standard < v_aggressive < v_ego
  assert v_relaxed == pytest.approx(v_ego * (1.35 / 2.5) ** 0.5, rel=1e-3)


def test_scc_curve_v_target_gentle_turn():
  v_ego = 25.0
  # Relaxed enters earlier and applies more gentle-curve reduction.
  v_relaxed = compute_scc_curve_v_target(v_ego, 0.95, log.LongitudinalPersonality.relaxed)
  v_standard = compute_scc_curve_v_target(v_ego, 0.95, log.LongitudinalPersonality.standard)

  assert v_relaxed < v_ego
  assert v_standard == pytest.approx(v_ego)
  assert v_relaxed < v_standard


def test_scc_curve_v_target_below_enter_threshold():
  v_ego = 25.0
  v_out = compute_scc_curve_v_target(v_ego, 0.5, log.LongitudinalPersonality.standard)
  assert v_out == pytest.approx(v_ego)


def test_scc_lookahead_lead_reduces_speed_before_curve():
  v_ego = 27.8
  max_pred = 2.5
  v_base = compute_scc_curve_v_target(v_ego, max_pred, log.LongitudinalPersonality.standard, min_v=0.0)
  n = 20
  pred = np.full(n, 0.5, dtype=np.float64)
  pred[5] = 1.5
  x = np.linspace(0.0, 200.0, n)
  v_lead = compute_scc_curve_v_target(
    v_ego, max_pred, log.LongitudinalPersonality.standard, min_v=0.0, position_x=x, predicted_lat_accels=pred)

  assert v_lead < v_base
  assert v_lead > 0.0
