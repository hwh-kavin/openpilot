"""Angle-mode lane-centering trim in the curvature domain.

Adds a small curvature correction to the planner/model command before the existing
measured-curvature deviation clip, so the trim inherits every downstream angle limiter.
The target is the lane-line center when both ego lines are trustworthy, otherwise the model's
own path (plus the user's offset). openpilot's lateral coordinate is left-positive, so a
positive ``offset`` shifts the target left; the Ford CAN layer negates path_angle on the wire.
"""

import numpy as np
from numpy import interp

from opendbc.car import DT_CTRL
from opendbc.car.ford.values import CarControllerParams


_WIDTH_TOLERANCE_BP = (2.4, 2.8, 3.75, 4.25)
_WIDTH_TOLERANCE_V = (0.0, 0.81, 0.81, 0.59)
_STD_TOLERANCE_BP = (0.3, 0.5)
_STD_TOLERANCE_V = (0.81, 0.0)
_CONFIDENCE_BP = (0.6, 0.8)
_CONFIDENCE_V = (0.0, 1.0)

_SPEED_RAMP_BP = (0.0, 9.0, 15.0)
_SPEED_RAMP_V = (0.0, 0.0, 1.0)
_LOOKAHEAD_MIN_M = 8.0
_LOOKAHEAD_MAX_M = 35.0
_MAX_RAW_CORRECTION = 0.004
_SMOOTH_TAU_S = 0.4
# Per-second correction slew, converted to the actual lateral tick below. At the current
# STEER_STEP=5 (20Hz) this is 0.00015 (1/m)/tick; keeping it cadence-derived avoids the
# stale-scaling class of bug seen when this branch changed STEER_STEP from 1 to 5.
_CORRECTION_ROC_PER_S = 0.003
_STEER_DT = CarControllerParams.STEER_STEP * DT_CTRL
_CORRECTION_ROC_PER_TICK = _CORRECTION_ROC_PER_S * _STEER_DT


class LaneCenterTrim:
  def __init__(self):
    self._correction = 0.0

  def reset(self) -> None:
    self._correction = 0.0

  def update(self, kappa_cmd: float, model, v_ego: float, enabled: bool, offset: float,
             gain: float, lat_active: bool, lane_change: bool) -> float:
    if not enabled or not lat_active or lane_change or model is None:
      self.reset()
      return kappa_cmd

    speed_factor = float(interp(v_ego, _SPEED_RAMP_BP, _SPEED_RAMP_V))
    if (not np.isfinite(v_ego) or speed_factor <= 0.0 or
        not (np.isfinite(offset) and np.isfinite(gain))):
      self.reset()
      return kappa_cmd

    valid, raw = self._raw_correction(model, v_ego, offset)
    if not valid:
      self.reset()
      return kappa_cmd

    target = float(np.clip(raw, -_MAX_RAW_CORRECTION, _MAX_RAW_CORRECTION))
    target *= float(np.clip(gain, 0.0, 1.0)) * speed_factor
    alpha = 1.0 - np.exp(-_STEER_DT / _SMOOTH_TAU_S)
    filtered = float(alpha * target + (1.0 - alpha) * self._correction)
    self._correction = float(np.clip(filtered,
                                     self._correction - _CORRECTION_ROC_PER_TICK,
                                     self._correction + _CORRECTION_ROC_PER_TICK))
    return kappa_cmd + self._correction

  @property
  def correction(self) -> float:
    return self._correction

  def _raw_correction(self, model, v_ego: float, offset: float) -> tuple[bool, float]:
    try:
      pos_x = np.asarray(model.position.x, dtype=float)
      pos_y = np.asarray(model.position.y, dtype=float)
      if pos_x.size < 2 or pos_x.size != pos_y.size:
        return False, 0.0
      if not (np.isfinite(pos_x).all() and np.isfinite(pos_y).all() and np.all(np.diff(pos_x) > 0)):
        return False, 0.0

      lookahead = float(np.clip(v_ego, _LOOKAHEAD_MIN_M, _LOOKAHEAD_MAX_M))
      model_y = float(np.interp(lookahead, pos_x, pos_y))
    except (AttributeError, IndexError, TypeError, ValueError):
      return False, 0.0

    scale, laneline_center_y = self._laneline_blend(model, lookahead)
    target_y = model_y * (1.0 - scale) + laneline_center_y * scale
    error = (target_y + offset) - model_y
    return True, float(2.0 * error / (lookahead ** 2))

  def _laneline_blend(self, model, lookahead: float) -> tuple[float, float]:
    try:
      lane_lines = model.laneLines
      probs = model.laneLineProbs
      stds = model.laneLineStds
      if len(lane_lines) < 3 or len(probs) < 3 or len(stds) < 3:
        return 0.0, 0.0

      left_x = np.asarray(lane_lines[1].x, dtype=float)
      left_y = np.asarray(lane_lines[1].y, dtype=float)
      right_x = np.asarray(lane_lines[2].x, dtype=float)
      right_y = np.asarray(lane_lines[2].y, dtype=float)
      if (left_x.size < 2 or left_x.size != left_y.size or
          right_x.size < 2 or right_x.size != right_y.size):
        return 0.0, 0.0
      if not (np.isfinite(left_x).all() and np.isfinite(left_y).all() and
              np.isfinite(right_x).all() and np.isfinite(right_y).all()):
        return 0.0, 0.0
      if not (np.all(np.diff(left_x) > 0) and np.all(np.diff(right_x) > 0)):
        return 0.0, 0.0
      if not all(np.isfinite(value) for value in (probs[1], probs[2], stds[1], stds[2])):
        return 0.0, 0.0

      left = float(np.interp(lookahead, left_x, left_y))
      right = float(np.interp(lookahead, right_x, right_y))
      width = right - left
      width_tolerance = float(interp(width, _WIDTH_TOLERANCE_BP, _WIDTH_TOLERANCE_V))
      std_tolerance = float(interp(max(float(stds[1]), float(stds[2])),
                                   _STD_TOLERANCE_BP, _STD_TOLERANCE_V))
      confidence = min(float(probs[1]), float(probs[2]), width_tolerance, std_tolerance)
      scale = float(np.clip(interp(confidence, _CONFIDENCE_BP, _CONFIDENCE_V), 0.0, 1.0))
      return scale, 0.5 * (left + right)
    except (AttributeError, IndexError, TypeError, ValueError):
      return 0.0, 0.0