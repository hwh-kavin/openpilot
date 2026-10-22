from opendbc.car.ford.carcontroller import (
  LOW_SPEED_GENTLE_BRAKE_JERK,
  LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_MAX_S,
  _ZIGZAG_IN_ACCEL,
  _ZIGZAG_WAIT_ACCEL,
  _ZIGZAG_WAIT_DECEL,
  update_low_speed_gentle_brake,
)
from opendbc.car import DT_CTRL
from opendbc.car.ford.values import CarControllerParams


def test_low_speed_decel_accel_decel_under_six_seconds_is_rate_limited():
  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=-0.2, requested_accel=-0.2, previous_accel=0.0, v_ego=5.0,
    lead_distance=10.0, lead_v_rel=0.0, active=False, zigzag_phase=_ZIGZAG_WAIT_DECEL,
    accel_duration=0.0, fcw=False, long_active=True, standstill=False,
  )
  assert not active
  assert phase == _ZIGZAG_WAIT_ACCEL

  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=0.2, requested_accel=0.2, previous_accel=-0.2, v_ego=3.0,
    lead_distance=10.0, lead_v_rel=0.0, active=active, zigzag_phase=phase,
    accel_duration=duration, fcw=False, long_active=True, standstill=False,
  )
  assert phase == _ZIGZAG_IN_ACCEL
  assert duration == DT_CTRL

  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=0.2, requested_accel=0.2, previous_accel=0.2, v_ego=3.0,
    lead_distance=10.0, lead_v_rel=0.0, active=active, zigzag_phase=phase,
    accel_duration=duration, fcw=False, long_active=True, standstill=False,
  )

  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=-0.5, requested_accel=-0.5, previous_accel=0.2, v_ego=3.0,
    lead_distance=10.0, lead_v_rel=0.0, active=active, zigzag_phase=phase,
    accel_duration=duration, fcw=False, long_active=True, standstill=False,
  )
  assert active
  assert accel == 0.2 - LOW_SPEED_GENTLE_BRAKE_JERK * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL

  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=-0.5, requested_accel=-0.5, previous_accel=accel, v_ego=4.3,
    lead_distance=10.0, lead_v_rel=0.0, active=active, zigzag_phase=phase,
    accel_duration=duration, fcw=False, long_active=True, standstill=False,
  )
  assert active
  assert accel > -0.5


def test_low_speed_zigzag_does_not_trigger_when_accel_phase_reaches_six_seconds():
  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=-0.5, requested_accel=-0.5, previous_accel=0.2, v_ego=3.0,
    lead_distance=10.0, lead_v_rel=0.0, active=False, zigzag_phase=_ZIGZAG_IN_ACCEL,
    accel_duration=LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_MAX_S,
    fcw=False, long_active=True, standstill=False,
  )
  assert not active
  assert phase == _ZIGZAG_WAIT_ACCEL
  assert accel == -0.5


def test_low_speed_zigzag_candidate_resets_at_fifteen_kph():
  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=0.2, requested_accel=0.2, previous_accel=0.2, v_ego=15.0 / 3.6,
    lead_distance=10.0, lead_v_rel=0.0, active=False, zigzag_phase=_ZIGZAG_IN_ACCEL,
    accel_duration=1.0, fcw=False, long_active=True, standstill=False,
  )
  assert not active
  assert phase == _ZIGZAG_WAIT_DECEL
  assert duration == 0.0


def test_low_speed_gentle_brake_bypasses_at_two_meters():
  active, phase, duration, accel = update_low_speed_gentle_brake(
    accel=-0.5, requested_accel=-0.5, previous_accel=0.2, v_ego=3.0,
    lead_distance=2.0, lead_v_rel=0.0, active=True, zigzag_phase=_ZIGZAG_WAIT_DECEL,
    accel_duration=0.0, fcw=False, long_active=True, standstill=False,
  )
  assert not active
  assert accel == -0.5


def test_low_speed_gentle_brake_bypasses_for_ttc_fcw_and_hard_braking():
  risk_cases = [
    (10.0, -4.0, False, -0.5, -0.5),
    (10.0, 0.0, True, -0.5, -0.5),
    (10.0, 0.0, False, -2.5, -2.5),
  ]
  for lead_distance, lead_v_rel, fcw, accel, requested_accel in risk_cases:
    active, phase, duration, accel = update_low_speed_gentle_brake(
      accel=accel, requested_accel=requested_accel, previous_accel=0.2, v_ego=3.0,
      lead_distance=lead_distance, lead_v_rel=lead_v_rel, active=True, zigzag_phase=_ZIGZAG_WAIT_DECEL,
      accel_duration=0.0, fcw=fcw, long_active=True, standstill=False,
    )
    assert not active
    assert accel == requested_accel