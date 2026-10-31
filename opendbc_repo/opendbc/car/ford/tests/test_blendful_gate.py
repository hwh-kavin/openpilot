from types import SimpleNamespace

from opendbc.sunnypilot.car.ford.lateral_angle_ext import LateralAngleExt


def _cs(left_blinker=False, right_blinker=False):
  return SimpleNamespace(out=SimpleNamespace(leftBlinker=left_blinker, rightBlinker=right_blinker))


def _ext(curvature=0.002, trust=0.8, enabled=True, lane_change=False):
  ext = LateralAngleExt()
  ext.blendful_curvature = curvature
  ext.blendful_trust = trust
  ext.blendful_enabled = enabled
  ext.lane_change = lane_change
  return ext


def test_applies_nudge_when_enabled_and_trusted():
  assert _ext().blendful_delta(_cs()) == 0.002


def test_zero_when_toggle_off():
  assert _ext(enabled=False).blendful_delta(_cs()) == 0.0


def test_zero_when_trust_low():
  assert _ext(trust=0.05).blendful_delta(_cs()) == 0.0
  assert _ext(trust=0.0).blendful_delta(_cs()) == 0.0


def test_zero_while_model_changes_lanes():
  assert _ext(lane_change=True).blendful_delta(_cs()) == 0.0


def test_zero_while_signalling():
  assert _ext().blendful_delta(_cs(left_blinker=True)) == 0.0
  assert _ext().blendful_delta(_cs(right_blinker=True)) == 0.0
