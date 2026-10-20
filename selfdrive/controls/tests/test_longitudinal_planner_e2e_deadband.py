import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  E2E_DEADBAND_MIN_LEAD_GAP,
  E2E_DEADBAND_MIN_SPEED,
  E2E_LIGHT_DECEL_DEADBAND,
  LongitudinalPlanner,
)


def make_planner(lead_d_rel=float('inf'), allow_throttle=True):
  # 只测试死区逻辑，不启动 MPC/acados
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.fcw = False
  planner.allow_throttle = allow_throttle
  planner._e2e_deadband_active = False
  planner._mpc_lead_d_rel = lead_d_rel
  return planner


def test_clamps_light_negative_accel():
  planner = make_planner()
  assert planner._apply_e2e_deadband(-E2E_LIGHT_DECEL_DEADBAND / 2, 20.0, False) == 0.0


def test_keeps_large_decel():
  planner = make_planner()
  accel = -(E2E_LIGHT_DECEL_DEADBAND * 2)
  assert planner._apply_e2e_deadband(accel, 20.0, False) == accel


def test_keeps_positive_accel():
  planner = make_planner()
  assert planner._apply_e2e_deadband(0.2, 20.0, False) == 0.2


def test_disabled_when_stopping():
  planner = make_planner()
  accel = -E2E_LIGHT_DECEL_DEADBAND / 2
  assert planner._apply_e2e_deadband(accel, 20.0, True) == accel


def test_disabled_below_min_speed():
  planner = make_planner()
  accel = -E2E_LIGHT_DECEL_DEADBAND / 2
  assert planner._apply_e2e_deadband(accel, E2E_DEADBAND_MIN_SPEED - 0.1, False) == accel


def test_disabled_with_close_lead():
  planner = make_planner(lead_d_rel=20.0 * (E2E_DEADBAND_MIN_LEAD_GAP - 0.5))
  accel = -E2E_LIGHT_DECEL_DEADBAND / 2
  assert planner._apply_e2e_deadband(accel, 20.0, False) == accel


def test_enabled_with_distant_lead():
  planner = make_planner(lead_d_rel=20.0 * (E2E_DEADBAND_MIN_LEAD_GAP + 1.0))
  assert planner._apply_e2e_deadband(-E2E_LIGHT_DECEL_DEADBAND / 2, 20.0, False) == 0.0


def test_disabled_when_throttle_not_allowed():
  planner = make_planner(allow_throttle=False)
  accel = -E2E_LIGHT_DECEL_DEADBAND / 2
  assert planner._apply_e2e_deadband(accel, 20.0, False) == accel


def test_hysteresis():
  planner = make_planner()
  assert planner._apply_e2e_deadband(-E2E_LIGHT_DECEL_DEADBAND / 2, 20.0, False) == 0.0
  assert planner._e2e_deadband_active

  # 已进入死区后允许更大的轻微减速，仍保持为 0
  accel = -E2E_LIGHT_DECEL_DEADBAND * 1.4
  assert planner._apply_e2e_deadband(accel, 20.0, False) == 0.0
  assert planner._e2e_deadband_active

  # 超过退出阈值则恢复真实减速度，并退出死区
  accel = -E2E_LIGHT_DECEL_DEADBAND * 2
  assert planner._apply_e2e_deadband(accel, 20.0, False) == accel
  assert not planner._e2e_deadband_active
