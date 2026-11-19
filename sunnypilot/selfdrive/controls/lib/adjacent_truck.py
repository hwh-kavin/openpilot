"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

BluePilot: adjacent-lane large vehicle (truck) detection for in-lane avoidance.

This module is intentionally free of any cereal/messaging dependency so it can be unit tested in
isolation and, more importantly, so that nothing here ever runs on the controls realtime path.
Its consumers are the standalone `adjacent_truckd` process (state/logging) and controlsd through
`adjacent_truck_avoid` (the rate-limited lateral offset).

What the sensors actually give us
---------------------------------
`liveTracks` (Car.RadarData) publishes every clustered radar track at 20 Hz, including the ones
that are *not* in our lane: (trackId, dRel, yRel, vRel). That is enough to say "there is something
in the adjacent lane"; it is not enough to say how big it is. RadarPoint carries no size field,
the cluster `dRel` is the nearest point (the rear face), and a big object does not stay in view
any longer than a small one: visibility time is (d_enter - d_exit) / delta_v, which is
independent of length. Track lifetime is therefore *not* a size proxy - it is a speed proxy.

Size comes from two orthogonal signals:

  * longitudinal extent - the radar clusters against the *previous* cycle's centroids with a ~5 m
    threshold, so one long vehicle is published as several adjacent tracks and no single cluster is
    ever longer than that threshold. The usable length therefore has to be stitched back together
    from a chain of contiguous tracks (see CHAIN_GAP), using only the standard dRel field.
  * RCS - the Ford DELPHI_MRR interface repurposes the otherwise unused `yvRel` field to carry the
    cluster's peak CAN_DET_AMPLITUDE (dBsm). The radar has a single, non-resolved elevation beam, so
    height and width are integrated into this one scalar: it is the only way "how big is this thing"
    enters the measurement at all. It is unaffected by the clustering split, but it drifts with
    range and aspect angle.

Neither signal is exact for an adjacent-lane vehicle, and the code is written defensively because of it:

  * the neighbour is seen at a grazing angle, not broadside, so its flank mostly specular-reflects
    away and only a few corner/wheel scatterers come back. The chain extent therefore under-reads and
    is treated as necessary-but-not-sufficient. (It is not compressed by the oblique geometry, though:
    dRel is already the longitudinal component, cos(azimuth) * dist, and `lateral` is measured from the
    lane centre so a curve does not break the chain.)
  * RCS depends on aspect angle - a truck presents a large flat tail while being approached (the phase
    where the decision has to be made) and much less when alongside - and near the sensor the elevation
    beam footprint is only a couple of metres tall, so a car and a truck both overfill it. RCS is
    therefore only trusted inside RCS_RANGE, and only this frame's value may drive the confirmation;
    a single multipath spike must never latch the classification.

Either one is enough to classify. When RCS is unavailable (other radar types) the chain extent still
works; when no lane-relative tracks line up, nothing is ever classified as a large vehicle, which is
the safe failure mode: no avoidance.

Why a static object must never qualify
--------------------------------------
Guardrails, tunnel walls, jersey barriers and cones sit exactly where an adjacent-lane truck
would and they produce huge spans. Speed rejects them trivially: a moving vehicle has
    v_truck = v_ego + v_rel    and |v_truck| is large,
while a static object always has v_rel ~= -v_ego, i.e. v_truck ~= 0.
"""

from dataclasses import dataclass

import numpy as np

# --- gating ---------------------------------------------------------------------------------
MIN_SPEED = 5.0            # m/s, no lateral offset games below this

# --- lane geometry --------------------------------------------------------------------------
LANE_WIDTH_DEFAULT = 3.6   # m
LANE_WIDTH_CLIP = (2.6, 4.8)
LANE_PROB_MIN = 0.35       # both lane lines must be at least this confident

# a track counts as "adjacent lane" when its offset from lane centre, normalised by lane width,
# falls in this band. 0.55..1.45 keeps us clear of the ego lane edge and of the lane after next.
NEIGHBOR_LO = 0.55
NEIGHBOR_HI = 1.45

# --- track window ---------------------------------------------------------------------------
D_NEAR = 10.0              # m, radar cannot see the whole flank closer than this
D_FAR = 70.0               # m, detections get too sparse to trust the span
V_REL_TAU = 0.4            # s, EMA on the dRel slope, which is far cleaner than the cluster vRel

# --- large vehicle classification -----------------------------------------------------------
SPAN_BIG = 7.0             # m, longitudinal extent above which we call it a large vehicle
SPAN_HYST = 1.5            # m, hysteresis once confirmed
RCS_BIG = 22.0             # dBsm, cluster peak RCS above which we call it a large vehicle (needs field calibration)
RCS_HYST = 4.0             # dBsm, hysteresis once confirmed
RCS_VALID = (0.0, 63.0)    # physical range of CAN_DET_AMPLITUDE; anything else means "not available"
# RCS is only trusted in this range band. Closer in, the elevation beam footprint is barely a couple
# of metres tall so both a car and a truck overfill it (and a truck's RCS gets under-read); further
# out, detections get sparse and the aspect angle drifts. Inside the band the approach is mostly
# rear-facing, which is the geometry the threshold should be calibrated against.
RCS_RANGE = (15.0, 55.0)   # m
V_TRUCK_MIN = 5.0          # m/s, rejects static objects (guardrails, walls, cones)
CONFIRM_T = 0.4            # s, must hold before we act

# The radar clusters against the *previous* cycle's centroids with a ~5 m threshold, so one long
# vehicle is published as several adjacent tracks and a single cluster's own extent is capped at
# roughly the threshold. The usable length signal is therefore the extent of a *chain* of contiguous
# adjacent-lane tracks, which we stitch together here.
CHAIN_GAP = 4.5            # m, max dRel gap between consecutive tracks of the same object
CHAIN_LATERAL = 0.7        # m, max lateral offset mismatch inside a chain

# --- pass timing ----------------------------------------------------------------------------
ARM_D = 48.0               # m, start offsetting once the vehicle is this close
D_BLIND = 12.0             # m, radar loses the flank around here; freeze and dead-reckon
EGO_LENGTH = 4.6           # m
TRUCK_LEN_MIN = 6.0        # m
TRUCK_LEN_MAX = 18.0       # m
TRUCK_LEN_SCALE = 1.2      # span is a lower bound, so pad it a little
HOLD_CAP_S = 22.0          # s, never hold longer than this (pacing a truck at equal speed)

# --- output ---------------------------------------------------------------------------------
MAX_OFFSET = 0.40          # m, in-lane lateral offset away from the truck (positive is left)
RETURN_T = 2.5             # s, nominal time to slide back to lane centre

# lane line indices in modelV2.laneLines / modelV2.laneLineProbs
IDX_LEFT = 1
IDX_RIGHT = 2

PHASE_IDLE = "idle"
PHASE_APPROACH = "approach"
PHASE_HOLD = "hold"
PHASE_RETURN = "return"


@dataclass
class Detection:
  active: bool = False
  phase: str = PHASE_IDLE
  side: int = 0                # +1 vehicle on the left, -1 on the right, 0 unknown/none
  offset_target: float = 0.0   # m, positive is to the left
  track_id: int = -1
  d_rel: float = 0.0
  y_rel: float = 0.0
  lateral: float = 0.0         # m, offset of the track from lane centre
  span: float = 0.0            # m, best observed cluster span
  rcs: float = 0.0             # dBsm, best observed cluster RCS (0.0 when unavailable)
  v_truck: float = 0.0         # m/s
  hold_remaining: float = 0.0  # s, only meaningful while holding


@dataclass
class _Candidate:
  track_id: int = -1
  side: int = 0
  v_rel_f: float = 0.0
  span_max: float = 0.0
  rcs_max: float = 0.0
  confirm_t: float = 0.0


@dataclass
class _Hit:
  side: int
  track_id: int
  d_rel: float
  y_rel: float
  lateral: float
  rcs: float
  rcs_trusted: bool
  width: float
  v_rel: float


def lane_reference(lane_lines, lane_probs, distance: float):
  """Return (center, width) of the ego lane at `distance`, or None if it cannot be trusted."""
  if lane_lines is None or lane_probs is None:
    return None

  ll = np.asarray(lane_lines, dtype=float)
  if ll.ndim != 2 or ll.shape[0] <= max(IDX_LEFT, IDX_RIGHT) or ll.shape[1] < 2:
    return None

  x = ll[0]
  if not np.all(np.isfinite(x)):
    return None
  if distance < x[0] or distance > x[-1]:
    return None

  probs = np.asarray(lane_probs, dtype=float).reshape(-1)
  if probs.size <= max(IDX_LEFT, IDX_RIGHT):
    return None
  if probs[IDX_LEFT] < LANE_PROB_MIN or probs[IDX_RIGHT] < LANE_PROB_MIN:
    return None

  left = float(np.interp(distance, x, ll[IDX_LEFT]))
  right = float(np.interp(distance, x, ll[IDX_RIGHT]))
  if not (np.isfinite(left) and np.isfinite(right)):
    return None

  width = left - right
  if width <= 0.5:  # lines crossed or garbage
    return None

  return 0.5 * (left + right), float(np.clip(width, *LANE_WIDTH_CLIP))


def frame_metrics(points, lane_lines, lane_probs, v_ego: float, gap_sweep=None) -> dict[int, dict]:
  """Raw per-frame measurements per side, for offline threshold calibration.

  Same geometry as the detector but with no state and no confirmation logic, so a log can be
  reduced to the numbers the thresholds are calibrated against.

  Returns {side: {"chain", "rcs", "rcs_trusted", "d_rel", "lateral", "v_truck", "n_tracks"}} for the
  sides that have at least one adjacent-lane track. If `gap_sweep` is given, each entry also carries
  "chain_sweep": {gap: chain_span} so CHAIN_GAP itself can be calibrated.
  """
  det = AdjacentTruckDetector()
  hits = det._hits(points, lane_lines, lane_probs)

  out: dict[int, dict] = {}
  for side in (1, -1):
    side_hits = [h for h in hits if h.side == side]
    if not side_hits:
      continue

    closest = min(side_hits, key=lambda h: h.d_rel)
    entry = {
      "chain": det._chain_span(hits, side),
      "rcs": max(h.rcs for h in side_hits),
      "rcs_trusted": max((h.rcs for h in side_hits if h.rcs_trusted), default=0.0),
      "d_rel": closest.d_rel,
      "lateral": closest.lateral,
      "v_truck": v_ego + closest.v_rel,
      "n_tracks": len(side_hits),
    }
    if gap_sweep is not None:
      entry["chain_sweep"] = {g: det._chain_span(hits, side, gap=g) for g in gap_sweep}
    out[side] = entry
  return out


class AdjacentTruckDetector:
  """Stateful detector. Feed one frame at a time, read back a `Detection`."""

  def __init__(self):
    self.cand = _Candidate()
    self.phase = PHASE_IDLE
    self.side = 0
    self.v_truck_frozen = 0.0
    self.d_rem = 0.0
    self.return_t = 0.0
    self.prev_d_rel: float | None = None
    self.offset_target = 0.0
    self.last = Detection()

  def reset(self) -> None:
    self.cand = _Candidate()
    self.phase = PHASE_IDLE
    self.side = 0
    self.v_truck_frozen = 0.0
    self.d_rem = 0.0
    self.return_t = 0.0
    self.prev_d_rel = None
    self.offset_target = 0.0

  # -- candidate selection -------------------------------------------------------------------

  def _hits(self, points, lane_lines, lane_probs):
    out: list[_Hit] = []
    for track_id, d_rel, y_rel, v_rel, rcs in points:
      if not (np.isfinite(d_rel) and np.isfinite(y_rel)):
        continue
      if d_rel < D_NEAR or d_rel > D_FAR:
        continue

      ref = lane_reference(lane_lines, lane_probs, d_rel)
      if ref is None:
        continue
      center, width = ref

      lateral = y_rel - center
      if not NEIGHBOR_LO * width < abs(lateral) < NEIGHBOR_HI * width:
        continue

      # out-of-range means "this radar does not report RCS (or did not fill the field)"
      rcs_plausible = np.isfinite(rcs) and RCS_VALID[0] <= rcs <= RCS_VALID[1]
      rcs = float(rcs) if rcs_plausible else 0.0
      # ...and outside RCS_RANGE the number is not trustworthy (beam footprint / aspect angle), but it
      # is still reported so thresholds can be calibrated from logs
      rcs_trusted = rcs_plausible and RCS_RANGE[0] <= d_rel <= RCS_RANGE[1]

      out.append(_Hit(
        side=1 if lateral > 0.0 else -1,
        track_id=int(track_id),
        d_rel=float(d_rel),
        y_rel=float(y_rel),
        lateral=float(lateral),
        rcs=rcs,
        rcs_trusted=rcs_trusted,
        width=width,
        v_rel=float(v_rel) if np.isfinite(v_rel) else 0.0,
      ))
    return out

  def _chain_span(self, hits, side: int, gap: float | None = None,
                  lateral_max: float | None = None) -> float:
    """Longitudinal extent of the contiguous run of adjacent-lane tracks on one side.

    A single cluster's own extent is capped by the radar's ~5 m clustering threshold, so the length
    of a long vehicle only becomes visible by stitching consecutive tracks back together.
    """
    gap = CHAIN_GAP if gap is None else gap
    lateral_max = CHAIN_LATERAL if lateral_max is None else lateral_max

    hs = sorted((h for h in hits if h.side == side), key=lambda h: h.d_rel)
    if not hs:
      return 0.0

    best = 0.0
    start = prev = hs[0]
    for h in hs[1:]:
      if h.d_rel - prev.d_rel <= gap and abs(h.lateral - prev.lateral) <= lateral_max:
        prev = h
      else:
        best = max(best, prev.d_rel - start.d_rel)
        start = prev = h
    return max(best, prev.d_rel - start.d_rel)

  def _update_candidate(self, hit: _Hit, chain_span: float, dt: float) -> None:
    if hit.track_id != self.cand.track_id:
      # new object: restart accumulation
      self.cand = _Candidate(track_id=hit.track_id, side=hit.side, v_rel_f=hit.v_rel,
                             span_max=chain_span, rcs_max=hit.rcs)
      self.prev_d_rel = None
      return

    if self.prev_d_rel is not None and dt > 1e-3:
      slope = (hit.d_rel - self.prev_d_rel) / dt
      self.cand.v_rel_f += (slope - self.cand.v_rel_f) * dt / (V_REL_TAU + dt)
    self.cand.span_max = max(self.cand.span_max, chain_span)
    self.cand.rcs_max = max(self.cand.rcs_max, hit.rcs)

  def _confirmed_big(self, hit: _Hit, chain_span: float, v_ego: float, dt: float) -> bool:
    """Confirm on *this frame's* evidence, not on the running maximum.

    Using the accumulated max here would let a single multipath spike latch the classification
    forever; the max is only kept for reporting and for the hysteresis margin.
    """
    confirmed = self.cand.confirm_t > 0.0
    span_needed = SPAN_BIG - (SPAN_HYST if confirmed else 0.0)
    rcs_needed = RCS_BIG - (RCS_HYST if confirmed else 0.0)
    rcs_now = hit.rcs if hit.rcs_trusted else 0.0
    evidence = chain_span >= span_needed or rcs_now >= rcs_needed
    big = evidence and abs(v_ego + self.cand.v_rel_f) > V_TRUCK_MIN
    self.cand.confirm_t = max(0.0, self.cand.confirm_t + (dt if big else -2.0 * dt))
    return self.cand.confirm_t >= CONFIRM_T

  def _other_side_blocked(self, hits, v_ego: float, side: int) -> bool:
    for other in (1, -1):
      if other == side:
        continue
      chain_span = self._chain_span(hits, other)
      for h in hits:
        if h.side != other:
          continue
        v_truck = v_ego + h.v_rel
        rcs_now = h.rcs if h.rcs_trusted else 0.0
        if (chain_span >= SPAN_BIG or rcs_now >= RCS_BIG) and abs(v_truck) > V_TRUCK_MIN:
          return True
    return False

  # -- phase helpers -------------------------------------------------------------------------

  def _enter_hold(self, v_ego: float, d_now: float) -> None:
    self.v_truck_frozen = v_ego + self.cand.v_rel_f
    length = float(np.clip(self.cand.span_max * TRUCK_LEN_SCALE, TRUCK_LEN_MIN, TRUCK_LEN_MAX))
    # distance still to cover before the whole ego car is past the whole truck
    want = d_now + EGO_LENGTH + length
    cap = max(0.5, abs(v_ego - self.v_truck_frozen) * HOLD_CAP_S)
    self.d_rem = float(np.clip(want, 0.5, cap))
    self.phase = PHASE_HOLD
    self.side = self.cand.side
    self.offset_target = -self.side * MAX_OFFSET
    self.prev_d_rel = None

  def _advance_blind(self, dt: float, v_ego: float) -> Detection:
    if self.phase == PHASE_HOLD:
      self.d_rem -= abs(v_ego - self.v_truck_frozen) * dt
      self.offset_target = -self.side * MAX_OFFSET
      if self.d_rem <= 0.0:
        self.phase = PHASE_RETURN
        self.return_t = RETURN_T
        self.offset_target = 0.0
      rate = max(abs(v_ego - self.v_truck_frozen), 0.1)
      return Detection(
        active=True, phase=self.phase, side=self.side, offset_target=self.offset_target,
        track_id=self.cand.track_id, d_rel=max(self.d_rem, 0.0),
        span=self.cand.span_max, rcs=self.cand.rcs_max,
        v_truck=self.v_truck_frozen, hold_remaining=max(self.d_rem, 0.0) / rate,
      )

    # PHASE_RETURN
    self.return_t -= dt
    self.offset_target = 0.0
    if self.return_t <= 0.0:
      self.reset()
      return Detection()

    return Detection(
      active=True, phase=PHASE_RETURN, side=self.side, offset_target=0.0,
      track_id=self.cand.track_id, span=self.cand.span_max, rcs=self.cand.rcs_max,
      v_truck=self.v_truck_frozen,
    )

  # -- main ----------------------------------------------------------------------------------

  def update(self, dt: float, v_ego: float, points, lane_lines, lane_probs,
             engaged: bool, enabled: bool) -> Detection:
    if not enabled or not engaged or v_ego < MIN_SPEED:
      self.reset()
      self.last = Detection()
      return self.last

    # the flank is invisible while beside us: run on timers only
    if self.phase in (PHASE_HOLD, PHASE_RETURN):
      self.last = self._advance_blind(dt, v_ego)
      return self.last

    hits = self._hits(points, lane_lines, lane_probs)

    # prefer the tracked candidate, otherwise the strongest reflector in an adjacent lane
    hit = next((h for h in hits if h.track_id == self.cand.track_id), None)
    if hit is None and hits:
      hit = max(hits, key=lambda h: (h.rcs, -h.d_rel))

    if hit is None:
      # lost it. If we were already offsetting, the radar just went blind - start predicting.
      if self.phase == PHASE_APPROACH:
        self._enter_hold(v_ego, D_BLIND)
        self.last = self._advance_blind(0.0, v_ego)
        return self.last
      self.reset()
      self.last = Detection()
      return self.last

    chain_span = self._chain_span(hits, hit.side)
    self._update_candidate(hit, chain_span, dt)
    confirmed = self._confirmed_big(hit, chain_span, v_ego, dt)
    blocked = self._other_side_blocked(hits, v_ego, hit.side)

    if confirmed and not blocked and hit.d_rel <= D_BLIND:
      self._enter_hold(v_ego, D_BLIND)
      self.last = self._advance_blind(0.0, v_ego)
      return self.last

    if confirmed and not blocked and hit.d_rel <= ARM_D:
      self.phase = PHASE_APPROACH
      self.side = hit.side
      self.offset_target = -self.side * MAX_OFFSET
    else:
      self.phase = PHASE_IDLE
      self.side = hit.side if confirmed else 0
      self.offset_target = 0.0

    self.last = Detection(
      active=self.phase == PHASE_APPROACH,
      phase=self.phase,
      side=self.side if confirmed else 0,
      offset_target=self.offset_target,
      track_id=hit.track_id,
      d_rel=hit.d_rel,
      y_rel=hit.y_rel,
      lateral=hit.lateral,
      span=self.cand.span_max,
      rcs=self.cand.rcs_max,
      v_truck=v_ego + self.cand.v_rel_f,
    )
    return self.last
