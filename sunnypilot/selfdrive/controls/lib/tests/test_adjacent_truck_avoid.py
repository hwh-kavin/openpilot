"""Adjacent-truck lateral offset: rate-limited curvature nudge on top of the detector."""
import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.adjacent_truck_avoid import (
  MAX_DK, MAX_OFFSET, OFFSET_RATE, AdjacentTruckAvoid,
)

DT20 = 0.05
DT100 = 0.01
V_EGO = 25.0


def lanes(width: float = 3.6, n: int = 21, x_max: float = 100.0):
  x = np.linspace(0.0, x_max, n)
  return np.vstack([x, np.full(n, width / 2), np.full(n, -width / 2)]), [1.0, 1.0, 1.0, 1.0]


def track(d_rel, y_rel, v_rel, rcs=30.0, track_id=7):
  return (track_id, d_rel, y_rel, v_rel, rcs)


def chain(d_front, y_rel, v_rel, rcs=30.0, n=4, spacing=4.0, track_id=100):
  return [track(d_front - spacing * i, y_rel, v_rel, rcs, track_id + i) for i in range(n)]


def feed(av, frames, *, d_rel=40.0, y=3.6, v_rel=-3.0, engaged=True, lane_change=False):
  out = 0.0
  for _ in range(frames):
    out = av.update(DT20, V_EGO, chain(d_rel, y, v_rel), *lanes(),
                    engaged=engaged, lane_change=lane_change)
    d_rel += v_rel * DT20
  return out


def test_lane_change_and_disengaged_do_nothing():
  av = AdjacentTruckAvoid()
  assert feed(av, 20, lane_change=True) == 0.0
  assert av.curvature(DT100, V_EGO, True) == 0.0
  av.reset()
  assert feed(av, 20, engaged=False) == 0.0
  assert av.target == 0.0


def test_left_truck_pushes_right_with_bounded_curvature():
  av = AdjacentTruckAvoid()
  assert feed(av, 20) == pytest.approx(-MAX_OFFSET)

  worst = 0.0
  for _ in range(400):  # 4 s at 100 Hz
    dk = av.curvature(DT100, V_EGO, True)
    worst = max(worst, abs(dk))

  assert -MAX_OFFSET <= av.offset < -0.35
  assert dk < 0.0
  assert worst <= MAX_DK + 1e-9


def test_right_truck_pushes_left():
  av = AdjacentTruckAvoid()
  assert feed(av, 20, y=-3.6) == pytest.approx(MAX_OFFSET)
  for _ in range(400):
    dk = av.curvature(DT100, V_EGO, True)
  assert dk > 0.0


def test_offset_ramps_instead_of_stepping():
  av = AdjacentTruckAvoid()
  feed(av, 20)
  last = av.offset
  for _ in range(200):
    av.curvature(DT100, V_EGO, True)
    assert abs(av.offset - last) <= OFFSET_RATE * DT100 + 1e-9
    last = av.offset


def test_release_returns_to_zero():
  av = AdjacentTruckAvoid()
  feed(av, 20)
  for _ in range(400):
    av.curvature(DT100, V_EGO, True)
  assert av.offset < -0.35

  # truck gone: target drops to zero and both the offset and the curvature delta release
  for _ in range(400):
    av.update(DT20, V_EGO, [], *lanes(), engaged=True, lane_change=False)
    dk = av.curvature(DT100, V_EGO, True)

  assert av.offset == 0.0
  assert abs(dk) < 1e-6


def test_not_lat_active_releases():
  av = AdjacentTruckAvoid()
  feed(av, 20)
  for _ in range(400):
    dk = av.curvature(DT100, V_EGO, False)
  assert abs(dk) < 1e-6
  assert av.offset == 0.0
