"""
BluePilot: turn the adjacent-lane large vehicle detection into a gentle in-lane lateral offset.

The detector (adjacent_truck.py) produces an offset target in metres; this class turns it into a
curvature delta for controlsd, following the Blendful approach:

  * the commanded offset is ramped at OFFSET_RATE instead of stepping,
  * the offset is converted to curvature at a lookahead distance (parabola: y = dk * x^2 / 2),
  * the curvature delta itself is rate limited with Blendful's build/release bound, so the wheel
    can never be moved abruptly,
  * everything falls back to zero: disabled, not lat-active, lane change, low speed, or any
    failure in the detector.

It runs in controlsd because that is where lateral control actually happens, and works with both
modeld implementations (the daemon's Param output is only for logging/state).
"""
import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib.adjacent_truck import (
  PHASE_HOLD, PHASE_APPROACH, AdjacentTruckDetector, Detection,
)

MAX_OFFSET = 0.40          # m, same cap as the detector; a lane is ~3.6 m wide
OFFSET_RATE = 0.30         # m/s, how fast the commanded offset moves (a full offset in ~1.3 s)
LOOKAHEAD_TIME = 2.0       # s, distance over which the offset is achieved
MIN_LOOKAHEAD = 12.0       # m
MAX_DK = 0.0025            # 1/m, hard cap on the curvature delta (~2.2 m/s^2 at 30 m/s)
BUILD = 0.5                # Blendful build bound: jerk-like limit, dk rate = BUILD / v^2 per s
RELEASE = 2.0              # Blendful release bound, lets go faster than it builds
MIN_SPEED = 5.0            # m/s, no offset games at low speed
ACTIVE_PHASES = (PHASE_APPROACH, PHASE_HOLD)


class AdjacentTruckAvoid:
  """Detection + offset ramp + curvature delta. Call update() at ~20 Hz, curvature() at 100 Hz."""

  def __init__(self):
    self.detector = AdjacentTruckDetector()
    self.target = 0.0   # m, desired offset, positive is left
    self.offset = 0.0   # m, ramped commanded offset
    self.dk = 0.0       # 1/m, curvature delta added to the model's desired curvature
    self.last = Detection()
    self.last_phase = "idle"

  def reset(self) -> None:
    self.detector.reset()
    self.target = 0.0
    self.offset = 0.0
    self.dk = 0.0
    self.last = Detection()
    self.last_phase = "idle"

  def disengage(self) -> None:
    """Feature disabled / lane change / not engaged: drop the detector state and ramp back."""
    self.detector.reset()
    self.target = 0.0

  def update(self, dt: float, v_ego: float, points, lane_lines, lane_probs,
             engaged: bool, lane_change: bool) -> float:
    """20 Hz entry point: run the detector and return the offset target (m)."""
    if lane_change or not engaged or v_ego < MIN_SPEED:
      self.disengage()
      return self.target

    self.last = self.detector.update(dt, v_ego, points, lane_lines, lane_probs, engaged, True)
    self.last_phase = self.last.phase
    if self.last.phase in ACTIVE_PHASES:
      self.target = float(np.clip(self.last.offset_target, -MAX_OFFSET, MAX_OFFSET))
    else:
      self.target = 0.0
    return self.target

  def curvature(self, dt: float, v_ego: float, lat_active: bool) -> float:
    """100 Hz entry point: ramp the offset and return the curvature delta (1/m)."""
    target = self.target if lat_active else 0.0

    # ramp the offset command (never step the wheel)
    step = OFFSET_RATE * dt
    self.offset += float(np.clip(target - self.offset, -step, step))

    # offset -> curvature at the lookahead distance; y = dk * x^2 / 2
    x = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    dk_target = float(np.clip(2.0 * self.offset / (x * x), -MAX_DK, MAX_DK))

    # Blendful-style build/release bound on the curvature delta
    v2 = max(v_ego, 1.0) ** 2
    if dk_target > self.dk:
      self.dk = min(self.dk + BUILD / v2 * dt, dk_target)
    else:
      self.dk = max(self.dk - RELEASE / v2 * dt, dk_target)

    # fully settled: drop the residue so we never add a constant bias
    if target == 0.0 and abs(self.offset) < 1e-3:
      self.offset = 0.0
      if abs(self.dk) < 1e-6:
        self.dk = 0.0

    return float(self.dk)
