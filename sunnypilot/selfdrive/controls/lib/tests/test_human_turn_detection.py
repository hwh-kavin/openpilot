from openpilot.sunnypilot.selfdrive.controls.lib.human_turn_detection import (
  HTDState,
  HumanTurnDetection,
  PauseReason,
)
import time


def _htd() -> HumanTurnDetection:
  h = HumanTurnDetection()
  h._read_params = lambda: None  # type: ignore[method-assign]
  h._enabled = True
  h._curve_exit_enabled = True
  h._angle_threshold_deg = 45.0
  h._resume_angle_diff_deg = 10.0
  h._resume_delay_sec = 0.0
  h._curve_exit_model_deg = 6.0
  h._curve_latch_deg = 12.0
  h._curve_exit_error_deg = 10.0
  h._curve_exit_resume_error_deg = 5.0
  h._curve_exit_resume_delay_sec = 0.0
  return h


def test_curve_exit_triggers_on_angle_error():
  """Q3-scale: steer ~18°, model ~3° → error 15° ≥ 10° triggers (no absolute 25°)."""
  h = _htd()
  allowed, state = h.update(True, 15.0, 20.0, False, 14.0)  # latch (≥12°)
  assert allowed and state == HTDState.INACTIVE
  assert h._curve_latched

  allowed, state = h.update(True, 18.0, 20.0, False, 3.0)  # error=15, model small
  assert not allowed
  assert state == HTDState.PAUSED
  assert h._pause_reason == PauseReason.CURVE_EXIT


def test_curve_exit_requires_prior_curve():
  h = _htd()
  # Never latched — should not pause even with large error
  allowed, state = h.update(True, 18.0, 20.0, False, 3.0)
  assert allowed and state == HTDState.INACTIVE


def test_curve_exit_no_trigger_when_error_small():
  h = _htd()
  h.update(True, 15.0, 20.0, False, 14.0)  # latch
  # model small but error only 4° < 10°
  allowed, state = h.update(True, 7.0, 20.0, False, 3.0)
  assert allowed and state == HTDState.INACTIVE


def test_curve_exit_immediate_resume_when_near_desired():
  h = _htd()
  h.update(True, 15.0, 20.0, False, 14.0)
  h.update(True, 18.0, 20.0, False, 3.0)  # pause

  # Still misaligned
  allowed, state = h.update(True, 18.0, 20.0, False, 3.0)
  assert not allowed and state == HTDState.PAUSED

  # Near desired + delay 0 → takeback (desired κ snap happens in controlsd)
  allowed, state = h.update(True, 6.0, 20.0, False, 3.0)  # error=3
  assert allowed and state == HTDState.INACTIVE


def test_curve_exit_resume_uses_delay_when_set():
  h = _htd()
  h._curve_exit_resume_delay_sec = 0.05
  h.update(True, 15.0, 20.0, False, 14.0)
  h.update(True, 18.0, 20.0, False, 3.0)

  # Aligned but delay not elapsed
  allowed, state = h.update(True, 6.0, 20.0, False, 3.0)
  assert not allowed and state == HTDState.PAUSED

  time.sleep(0.06)
  allowed, state = h.update(True, 6.0, 20.0, False, 3.0)
  assert allowed and state == HTDState.INACTIVE


def test_curve_exit_aborts_when_model_wants_turn():
  h = _htd()
  h.update(True, 15.0, 20.0, False, 14.0)
  h.update(True, 18.0, 20.0, False, 3.0)

  # model 14° ≥ latch 12 → re-engage
  allowed, state = h.update(True, 18.0, 20.0, False, 14.0)
  assert allowed and state == HTDState.INACTIVE


def test_human_turn_still_works():
  h = _htd()
  allowed, state = h.update(True, 50.0, 20.0, True, 10.0)
  assert not allowed and state == HTDState.PAUSED
  assert h._pause_reason == PauseReason.HUMAN_TURN

  allowed, state = h.update(True, 50.0, 20.0, False, 10.0)
  assert allowed and state == HTDState.INACTIVE


def test_no_curve_exit_while_lane_changing():
  h = _htd()
  h.update(True, 15.0, 20.0, False, 14.0)
  allowed, state = h.update(True, 18.0, 20.0, False, 3.0, lane_changing=True)
  assert allowed and state == HTDState.INACTIVE
