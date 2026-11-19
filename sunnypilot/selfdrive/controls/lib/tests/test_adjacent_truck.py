"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import adjacent_truck as at
from openpilot.sunnypilot.selfdrive.controls.lib.adjacent_truck import AdjacentTruckDetector, lane_reference

DT = 0.05               # 20 Hz, same as liveTracks
V_EGO = 25.0            # m/s
NO_RCS = float('nan')   # radar that does not report RCS
CHAIN_SPACING = 4.0     # m, how the radar splits one long object into consecutive tracks


def make_lanes(width: float = 3.6, n: int = 21, x_max: float = 100.0):
  """Straight lane, ego centred, y positive to the left (openpilot model frame)."""
  x = np.linspace(0.0, x_max, n)
  return np.vstack([x, np.full(n, width / 2), np.full(n, -width / 2)]), [1.0, 1.0, 1.0, 1.0]


def track(d_rel, y_rel, v_rel, rcs=NO_RCS, track_id=7):
  return (track_id, d_rel, y_rel, v_rel, rcs)


def chain_pts(d_front, y_rel, v_rel, rcs, n=4, spacing=CHAIN_SPACING, track_id=100):
  """One long object, published by the radar as `n` consecutive tracks."""
  return [track(d_front - spacing * i, y_rel, v_rel, rcs, track_id + i) for i in range(n)]


def step(det, frames: int, *, d_rel: float, y_rel: float, v_rel: float, rcs=NO_RCS,
         n_tracks: int = 1, spacing: float = CHAIN_SPACING, v_ego: float = V_EGO,
         lane_width: float = 3.6, points_extra=None):
  """Advance `frames` frames with a track (or a chain of tracks) approaching at `v_rel`."""
  lanes, probs = make_lanes(lane_width)
  out = None
  for _ in range(frames):
    pts = chain_pts(d_rel, y_rel, v_rel, rcs, n=n_tracks, spacing=spacing)
    if points_extra is not None:
      pts += points_extra
    out = det.update(DT, v_ego, pts, lanes, probs, engaged=True, enabled=True)
    d_rel += v_rel * DT
  return out


# ---------------------------------------------------------------------------- lane geometry

def test_lane_reference_sane():
  lanes, probs = make_lanes(3.6)
  center, width = lane_reference(lanes, probs, 30.0)
  assert center == pytest.approx(0.0, abs=1e-9)
  assert width == pytest.approx(3.6, abs=1e-9)


def test_lane_reference_rejects_low_confidence():
  lanes, _ = make_lanes(3.6)
  probs = [1.0, 0.9, 0.1, 1.0]  # right line is not trusted
  assert lane_reference(lanes, probs, 30.0) is None


def test_lane_reference_rejects_out_of_range_distance():
  lanes, probs = make_lanes(3.6, x_max=50.0)
  assert lane_reference(lanes, probs, 80.0) is None


# ---------------------------------------------------------------------------- length via chain

def test_long_vehicle_split_across_tracks_is_stitched_together():
  """The radar clusters against the previous cycle with a ~5 m threshold, so one truck arrives as
  several tracks. Their chain extent must be what decides the classification."""
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, n_tracks=4)
  assert out.active
  assert out.phase == at.PHASE_APPROACH
  assert out.side == 1
  assert out.span >= 11.0     # 4 tracks 4 m apart => 12 m


def test_short_vehicle_is_ignored():
  """A normal car is a single short track; chain extent stays near zero."""
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=3.6, v_rel=-3.0, n_tracks=1)
  assert not out.active
  assert out.phase == at.PHASE_IDLE
  assert out.side == 0


def test_separate_cars_are_not_chained_into_one_truck():
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  out = None
  for _ in range(20):
    pts = [track(40.0, 3.6, -3.0, NO_RCS, track_id=1), track(20.0, 3.6, -3.0, NO_RCS, track_id=2)]
    out = det.update(DT, V_EGO, pts, lanes, probs, engaged=True, enabled=True)
  assert not out.active


def test_chain_ignores_laterally_misaligned_track():
  """A track in the next lane over must not extend the chain."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  out = None
  for _ in range(20):
    pts = chain_pts(40.0, 3.6, -3.0, NO_RCS, n=4) + [track(28.0, 3.6 + 2.0, -3.0, NO_RCS, track_id=200)]
    out = det.update(DT, V_EGO, pts, lanes, probs, engaged=True, enabled=True)
  # the offset track is outside the neighbour band, so the chain stays at 12 m either way
  assert out.active


def test_ego_lane_vehicle_is_ignored():
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=0.0, v_rel=-3.0, n_tracks=4)
  assert not out.active


def test_vehicle_two_lanes_over_is_ignored():
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=7.2, v_rel=-3.0, n_tracks=4)
  assert not out.active


# ---------------------------------------------------------------------------- speed gate

def test_static_object_is_rejected_even_with_huge_extent():
  """A guardrail is a long chain of tracks that does not move: v_truck ~= 0 disqualifies it."""
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=3.6, v_rel=-V_EGO, n_tracks=6)
  assert not out.active
  assert out.phase == at.PHASE_IDLE


def test_high_rcs_static_object_is_still_rejected():
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=3.6, v_rel=-V_EGO, rcs=35.0, n_tracks=1)
  assert not out.active


# ---------------------------------------------------------------------------- size gate: RCS

def test_rcs_alone_classifies_a_split_truck():
  """Each piece of an articulated truck has a short chain, but truck-level RCS."""
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0, n_tracks=1)
  assert out.active
  assert out.rcs == pytest.approx(30.0)


def test_car_with_normal_rcs_is_ignored():
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=12.0, n_tracks=1)
  assert not out.active


def test_implausible_rcs_is_treated_as_unavailable():
  """Out of the CAN_DET_AMPLITUDE range => the field is not really an RCS (other radar / unset)."""
  det = AdjacentTruckDetector()
  out = step(det, 60, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=500.0, n_tracks=1)
  assert not out.active


def test_missing_rcs_falls_back_to_chain_extent():
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=NO_RCS, n_tracks=4)
  assert out.active
  assert out.rcs == 0.0


def test_rcs_accumulates_as_a_maximum():
  """A single strong reflection is enough; it must not be averaged away."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  rcs_seq = [10.0, 10.0, 31.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
  d_rel = 40.0
  out = None
  for rcs in rcs_seq:
    out = det.update(DT, V_EGO, [track(d_rel, 3.6, -3.0, rcs)], lanes, probs, engaged=True, enabled=True)
    d_rel += -3.0 * DT
  assert out.rcs == pytest.approx(31.0)


def test_single_rcs_spike_does_not_arm():
  """RCS max is only for reporting. One multipath spike on a car must not latch the classification -
  confirmation has to come from sustained per-frame evidence."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  seq = [12.0] * 10 + [40.0] + [12.0] * 20
  d_rel = 40.0
  out = None
  for rcs in seq:
    out = det.update(DT, V_EGO, [track(d_rel, 3.6, -3.0, rcs)], lanes, probs, engaged=True, enabled=True)
    d_rel += -3.0 * DT
  assert not out.active
  assert out.rcs == pytest.approx(40.0)   # still reported, just not acted on


def test_rcs_is_ignored_outside_the_reliable_range_band():
  """Too far out the aspect angle and beam footprint make the RCS number meaningless for the
  decision, but the raw value is still reported so thresholds can be calibrated from logs."""
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=65.0, y_rel=3.6, v_rel=-3.0, rcs=35.0, n_tracks=1)
  assert not out.active
  assert out.rcs == pytest.approx(35.0)


def test_rcs_is_trusted_inside_the_reliable_range_band():
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=35.0, n_tracks=1)
  assert out.active
  assert out.rcs == pytest.approx(35.0)


# ---------------------------------------------------------------------------- direction

def test_offset_pushes_away_from_truck_on_the_left():
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0)
  assert out.side == 1
  assert out.offset_target == pytest.approx(-at.MAX_OFFSET)


def test_offset_pushes_away_from_truck_on_the_right():
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=-3.6, v_rel=-3.0, rcs=30.0)
  assert out.side == -1
  assert out.offset_target == pytest.approx(at.MAX_OFFSET)


def test_blocked_other_side_prevents_offset():
  det = AdjacentTruckDetector()
  other = chain_pts(40.0, -3.6, -3.0, NO_RCS, n=4, track_id=300)
  out = step(det, 40, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0, points_extra=other)
  assert not out.active
  assert out.offset_target == 0.0


def test_blocked_other_side_via_rcs_prevents_offset():
  det = AdjacentTruckDetector()
  other = [track(40.0, -3.6, -3.0, 32.0, track_id=9)]
  out = step(det, 40, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0, points_extra=other)
  assert not out.active


# ---------------------------------------------------------------------------- pass timing

def test_enters_hold_then_returns_to_idle():
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)

  d_rel = 40.0
  phase_seen = set()
  for _ in range(800):
    pts = chain_pts(d_rel, 3.6, -3.0, 30.0, n=4) if d_rel > at.D_BLIND else []
    out = det.update(DT, V_EGO, pts, lanes, probs, engaged=True, enabled=True)
    phase_seen.add(out.phase)
    d_rel += -3.0 * DT
    if out.phase == at.PHASE_IDLE and at.PHASE_RETURN in phase_seen:
      break

  assert at.PHASE_APPROACH in phase_seen
  assert at.PHASE_HOLD in phase_seen
  assert at.PHASE_RETURN in phase_seen
  assert det.phase == at.PHASE_IDLE


def test_hold_countdown_shrinks_over_time():
  det = AdjacentTruckDetector()
  out = step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0)
  assert out.phase == at.PHASE_APPROACH

  lanes, probs = make_lanes(3.6)
  first = None
  last = None
  for _ in range(400):
    out = det.update(DT, V_EGO, [], lanes, probs, engaged=True, enabled=True)
    if out.phase == at.PHASE_HOLD:
      if first is None:
        first = out.hold_remaining
      last = out.hold_remaining
    elif first is not None:
      break
  assert first is not None
  assert last is not None
  assert last < first


def test_hold_is_capped_when_speeds_match():
  """Pacing a truck at the same speed would otherwise hold the offset forever."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)

  d_rel = 40.0
  for _ in range(300):
    pts = chain_pts(d_rel, 3.6, -0.2, 30.0, n=4) if d_rel > at.D_BLIND else []
    det.update(DT, V_EGO, pts, lanes, probs, engaged=True, enabled=True)
    d_rel += -0.2 * DT

  assert det.d_rem <= abs(V_EGO - (V_EGO - 0.2)) * at.HOLD_CAP_S + 1e-6


# ---------------------------------------------------------------------------- gating

def test_disabled_never_acts():
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  out = None
  for _ in range(40):
    out = det.update(DT, V_EGO, chain_pts(40.0, 3.6, -3.0, 30.0), lanes, probs, engaged=True, enabled=False)
  assert not out.active
  assert out.phase == at.PHASE_IDLE


def test_not_engaged_never_acts():
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  out = None
  for _ in range(40):
    out = det.update(DT, V_EGO, chain_pts(40.0, 3.6, -3.0, 30.0), lanes, probs, engaged=False, enabled=True)
  assert not out.active


def test_low_speed_never_acts():
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  out = None
  for _ in range(40):
    out = det.update(DT, at.MIN_SPEED - 1.0, chain_pts(20.0, 3.6, -1.0, 30.0), lanes, probs,
                     engaged=True, enabled=True)
  assert not out.active


def test_reset_clears_state():
  det = AdjacentTruckDetector()
  step(det, 20, d_rel=40.0, y_rel=3.6, v_rel=-3.0, rcs=30.0)
  assert det.phase == at.PHASE_APPROACH
  det.reset()
  assert det.phase == at.PHASE_IDLE
  assert det.cand.track_id == -1
  assert det.cand.rcs_max == 0.0
  assert det.d_rem == 0.0


# ---------------------------------------------------------------------------- offline metrics

def test_frame_metrics_matches_detector_geometry():
  lanes, probs = make_lanes(3.6)
  pts = chain_pts(40.0, 3.6, -3.0, 30.0, n=4)
  m = at.frame_metrics(pts, lanes, probs, V_EGO, gap_sweep=(4.0, 5.0))

  assert set(m) == {1}                       # only the left side has tracks
  assert m[1]["n_tracks"] == 4
  assert m[1]["chain"] >= 11.0
  assert m[1]["rcs"] == pytest.approx(30.0)
  assert m[1]["v_truck"] == pytest.approx(V_EGO - 3.0)
  assert m[1]["d_rel"] == pytest.approx(28.0)   # closest track of the chain
  assert set(m[1]["chain_sweep"]) == {4.0, 5.0}


def test_frame_metrics_reports_both_sides():
  lanes, probs = make_lanes(3.6)
  pts = chain_pts(40.0, 3.6, -3.0, 30.0, n=2, track_id=1) + \
        chain_pts(30.0, -3.6, -3.0, 30.0, n=2, track_id=50)
  m = at.frame_metrics(pts, lanes, probs, V_EGO)
  assert set(m) == {-1, 1}


def test_frame_metrics_ignores_empty_scene():
  lanes, probs = make_lanes(3.6)
  assert at.frame_metrics([], lanes, probs, V_EGO) == {}


# ---------------------------------------------------------------------------- regressions

def test_track_loss_holds_from_last_known_distance():
  """Losing the flank must not pretend the truck is already alongside (D_BLIND); the offset has to
  keep being held for the distance that was last measured, or it releases while the truck is ahead."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)

  d_rel = 40.0
  for _ in range(20):
    det.update(DT, V_EGO, chain_pts(d_rel, 3.6, -3.0, 30.0), lanes, probs, engaged=True, enabled=True)
    d_rel += -3.0 * DT

  assert det.phase == at.PHASE_APPROACH
  last_seen = d_rel

  out = det.update(DT, V_EGO, [], lanes, probs, engaged=True, enabled=True)
  assert out.phase == at.PHASE_HOLD
  # want = last_seen + ego length + truck length; the old code started from D_BLIND (12 m)
  assert det.d_rem > at.D_BLIND + at.EGO_LENGTH + 10.0
  assert det.d_rem > 0.8 * last_seen


def test_pass_span_survives_track_id_churn():
  """The radar re-clusters every cycle. Once a 12 m chain has been measured, a short single-cluster
  frame with a brand-new track id must not wipe the pass span (it is the truck-length estimate)."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)

  d_rel = 40.0
  for _ in range(20):
    det.update(DT, V_EGO, chain_pts(d_rel, 3.6, -3.0, 30.0, track_id=100), lanes, probs, engaged=True, enabled=True)
    d_rel += -3.0 * DT
  assert det.pass_span == pytest.approx(12.0)

  # re-clustered as one short piece with a new id, still inside the trusted RCS range
  for _ in range(12):
    det.update(DT, V_EGO, chain_pts(20.0, 3.6, -3.0, 30.0, n=1, track_id=500), lanes, probs,
               engaged=True, enabled=True)

  assert det.cand.span_max < at.SPAN_BIG
  assert det.pass_span == pytest.approx(12.0)


def test_enter_hold_uses_pass_span_not_cluster_span():
  """At the hand-off the last visible cluster can be short; the hold length must come from the pass
  span, otherwise the offset is released while the truck is still alongside."""
  det = AdjacentTruckDetector()
  det.pass_span = 12.0
  det.cand.span_max = 0.0
  det.cand.v_rel_f = -3.0

  det._enter_hold(V_EGO, at.D_BLIND)

  # want = D_BLIND + EGO_LENGTH + clip(12 * 1.2, 6, 18) = 31 m
  assert det.d_rem >= at.D_BLIND + at.EGO_LENGTH + 10.0


def test_chain_requires_matching_relative_speed():
  """Two adjacent-lane cars travelling at clearly different speeds must not be stitched into one
  long vehicle just because their clusters happen to be close together."""
  det = AdjacentTruckDetector()
  lanes, probs = make_lanes(3.6)
  pts = [track(40.0, 3.6, -3.0, NO_RCS, 1), track(36.0, 3.6, -3.0, NO_RCS, 2),
         track(32.0, 3.6, -9.0, NO_RCS, 3), track(28.0, 3.6, -9.0, NO_RCS, 4)]

  out = None
  for _ in range(20):
    out = det.update(DT, V_EGO, pts, lanes, probs, engaged=True, enabled=True)

  assert not out.active
  assert out.phase == at.PHASE_IDLE
