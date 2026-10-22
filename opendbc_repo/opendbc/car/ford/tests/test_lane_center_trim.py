import unittest

import numpy as np

from opendbc.sunnypilot.car.ford.lane_center_trim import LaneCenterTrim, _CORRECTION_ROC_PER_TICK


class _XY:
  def __init__(self, x, y):
    self.x = x
    self.y = y


class _Position:
  def __init__(self, x, y):
    self.x = x
    self.y = y


class _Model:
  def __init__(self, lane_center_y=0.0, model_y=0.0, probs=(0.9, 0.9, 0.9, 0.9),
               stds=(0.1, 0.1, 0.1, 0.1), width=3.7):
    xs = list(range(0, 60, 2))
    half_width = width / 2.0
    self.laneLines = [
      _XY(xs, [lane_center_y - half_width - 3.7] * len(xs)),
      _XY(xs, [lane_center_y - half_width] * len(xs)),
      _XY(xs, [lane_center_y + half_width] * len(xs)),
      _XY(xs, [lane_center_y + half_width + 3.7] * len(xs)),
    ]
    self.laneLineProbs = list(probs)
    self.laneLineStds = list(stds)
    self.position = _Position(xs, [model_y] * len(xs))


class TestLaneCenterTrim(unittest.TestCase):
  def setUp(self):
    self.trim = LaneCenterTrim()

  def _run(self, model, offset=0.0, gain=1.0, speed=15.0, enabled=True,
           lat_active=True, lane_change=False, iterations=1, kappa_cmd=0.0):
    result = kappa_cmd
    for _ in range(iterations):
      result = self.trim.update(kappa_cmd, model, speed, enabled, offset, gain,
                                lat_active, lane_change)
    return result

  def test_lane_lines_pull_toward_center(self):
    model = _Model(lane_center_y=2.0)
    self._run(model, iterations=500)
    self.assertGreater(self.trim.correction, 0.0)

  def test_missing_lane_lines_keep_user_offset(self):
    model = _Model(probs=(0.9, 0.0, 0.0, 0.9))
    self._run(model, offset=0.5, iterations=500)
    self.assertGreater(self.trim.correction, 0.0)

  def test_low_confidence_lines_do_not_pull_toward_phantom_center(self):
    model = _Model(lane_center_y=2.0, probs=(0.9, 0.1, 0.1, 0.9))
    self._run(model, iterations=100)
    self.assertAlmostEqual(self.trim.correction, 0.0, places=6)

  def test_strength_scales_correction(self):
    model = _Model()
    self._run(model, offset=0.5, gain=1.0, iterations=500)
    full_strength = self.trim.correction
    self.trim.reset()
    self._run(model, offset=0.5, gain=0.5, iterations=500)
    self.assertAlmostEqual(self.trim.correction, full_strength * 0.5, places=6)

  def test_disabled_lane_change_and_low_speed_reset(self):
    model = _Model()
    self._run(model, offset=0.5, iterations=30)
    self.assertNotEqual(self.trim.correction, 0.0)
    self._run(model, offset=0.5, enabled=False)
    self.assertEqual(self.trim.correction, 0.0)
    self._run(model, offset=0.5, lane_change=True)
    self.assertEqual(self.trim.correction, 0.0)
    self._run(model, offset=0.5, speed=5.0)
    self.assertEqual(self.trim.correction, 0.0)

  def test_correction_is_bounded_and_rate_limited(self):
    model = _Model()
    previous = self._run(model, offset=0.5, iterations=1)
    current = self._run(model, offset=-0.5, iterations=1)
    self.assertLessEqual(abs(current - previous), _CORRECTION_ROC_PER_TICK + 1e-9)
    self._run(model, offset=0.5, iterations=1000)
    self.assertLessEqual(abs(self.trim.correction), 0.004)

  def test_nonfinite_tuning_is_inert(self):
    result = self._run(_Model(), offset=float("nan"), kappa_cmd=0.01)
    self.assertTrue(np.isfinite(result))
    self.assertEqual(self.trim.correction, 0.0)


if __name__ == "__main__":
  unittest.main()