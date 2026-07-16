import numpy as np
from typing import cast
from collections import defaultdict
from math import cos, sin
from dataclasses import dataclass
from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford.fordcan import CanBus
from opendbc.car.ford.values import DBC, RADAR
from opendbc.car.interfaces import RadarInterfaceBase

DELPHI_ESR_RADAR_MSGS = list(range(0x500, 0x540))

DELPHI_MRR_RADAR_START_ADDR = 0x120
DELPHI_MRR_RADAR_HEADER_ADDR = 0x174  # MRR_Header_SensorCoverage
DELPHI_MRR_RADAR_MSG_COUNT = 64

DELPHI_MRR_RADAR_RANGE_COVERAGE = {0: 42, 1: 164, 2: 45, 3: 175}  # scan index to detection range (m)
DELPHI_MRR_MIN_LONG_RANGE_DIST = 30  # meters
DELPHI_MRR_CLUSTER_THRESHOLD = 5  # meters, lateral distance and relative velocity are weighted
# Scan index should step 0→1→2→3→0. Treat stuck (R/P repeat) vs skip (missed header) separately.
DELPHI_MRR_STUCK_THRESHOLD = 15  # same index repeated (typical in reverse)
DELPHI_MRR_SKIP_THRESHOLD = 10   # non-sequential jumps while driving; ignore brief blips

# Temporal / kinematic clutter filter (one publish cycle ≈ full 0→3 scan rotation)
DELPHI_MRR_CONFIRM_CYCLES = 3          # new tracks must persist this many cycles before publish
DELPHI_MRR_MAX_DREL_JUMP = 10.0        # m — reject single-cycle range spikes
DELPHI_MRR_MAX_VREL_JUMP = 10.0        # m/s — reject Doppler spikes
DELPHI_MRR_MAX_YREL_JUMP = 3.0         # m — reject lateral teleport

MRR_FAULT_SIGNALS = (
  'CAN_RADAR_NOT_OP',
  'CAN_RADAR_OVERHEAT_ERROR',
  'CAN_RADAR_EXT_COND_NOK',
  'CAN_RADAR_ALIGN_OUT_RANGE',
)


@dataclass
class Cluster:
  dRel: float = 0.0
  yRel: float = 0.0
  vRel: float = 0.0
  trackId: int = 0


@dataclass
class _TrackFilter:
  """Per-track state for MRR publish gating (association still uses Cluster)."""
  dRel: float
  yRel: float
  vRel: float
  confirm_cnt: int = 0
  published: bool = False


def cluster_points(pts_l: list[list[float]], pts2_l: list[list[float]], max_dist: float) -> list[int]:
  """
  Clusters a collection of points based on another collection of points. This is useful for correlating clusters through time.
  Points in pts2 not close enough to any point in pts are assigned -1.
  Args:
    pts_l: List of points to base the new clusters on
    pts2_l: List of points to cluster using pts
    max_dist: Max distance from cluster center to candidate point

  Returns:
    List of cluster indices for pts2 that correspond to pts
  """

  if not len(pts2_l):
    return []

  if not len(pts_l):
    return [-1] * len(pts2_l)

  max_dist_sq = max_dist ** 2
  pts = np.array(pts_l)
  pts2 = np.array(pts2_l)

  # Compute squared norms
  pts_norm_sq = np.sum(pts ** 2, axis=1)
  pts2_norm_sq = np.sum(pts2 ** 2, axis=1)

  # Compute squared Euclidean distances using the identity
  # dist_sq[i, j] = ||pts2[i]||^2 + ||pts[j]||^2 - 2 * pts2[i] . pts[j]
  dist_sq = pts2_norm_sq[:, np.newaxis] + pts_norm_sq[np.newaxis, :] - 2 * np.dot(pts2, pts.T)
  dist_sq = np.maximum(dist_sq, 0.0)

  # Find the closest cluster for each point and assign its index
  closest_clusters = np.argmin(dist_sq, axis=1)
  closest_dist_sq = dist_sq[np.arange(len(pts2)), closest_clusters]
  cluster_idxs = np.where(closest_dist_sq < max_dist_sq, closest_clusters, -1)

  return cast(list[int], cluster_idxs.tolist())


def _create_delphi_esr_radar_can_parser(CP) -> CANParser:
  msg_n = len(DELPHI_ESR_RADAR_MSGS)
  messages = list(zip(DELPHI_ESR_RADAR_MSGS, [20] * msg_n, strict=True))

  return CANParser(RADAR.DELPHI_ESR, messages, CanBus(CP).radar)


def _create_delphi_mrr_radar_can_parser(CP) -> CANParser:
  messages = [
    ("MRR_Status_Radar", 30),
    ("MRR_Header_InformationDetections", 33),
    ("MRR_Header_SensorCoverage", 33),
  ]

  for i in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
    msg = f"MRR_Detection_{i:03d}"
    messages += [(msg, 33)]

  return CANParser(RADAR.DELPHI_MRR, messages, CanBus(CP).radar)


class RadarInterface(RadarInterfaceBase):
  def __init__(self, CP, CP_SP):
    super().__init__(CP, CP_SP)

    self.points: list[list[float]] = []
    self.clusters: list[Cluster] = []
    self._track_filters: dict[int, _TrackFilter] = {}

    self.updated_messages = set()
    self.track_id = 0
    self.radar = DBC[CP.carFingerprint].get(Bus.radar)
    self.scan_index_invalid_cnt = 0
    self.radar_stuck_cnt = 0
    self.radar_skip_cnt = 0
    self.prev_headerScanIndex = 0
    self._header_scan_initialized = False
    if CP.radarUnavailable:
      self.rcp = None
    elif self.radar == RADAR.DELPHI_ESR:
      self.rcp = _create_delphi_esr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_ESR_RADAR_MSGS[-1]
      self.valid_cnt = {key: 0 for key in DELPHI_ESR_RADAR_MSGS}
    elif self.radar == RADAR.DELPHI_MRR:
      self.rcp = _create_delphi_mrr_radar_can_parser(CP)
      self.trigger_msg = DELPHI_MRR_RADAR_HEADER_ADDR
    else:
      raise ValueError(f"Unsupported radar: {self.radar}")

  def _reset_mrr_tracks(self) -> None:
    self.pts.clear()
    self.points.clear()
    self.clusters.clear()
    self._track_filters.clear()

  @staticmethod
  def _mrr_kinematic_jump(prev: _TrackFilter, d_rel: float, y_rel: float, v_rel: float) -> bool:
    return (abs(d_rel - prev.dRel) > DELPHI_MRR_MAX_DREL_JUMP or
            abs(y_rel - prev.yRel) > DELPHI_MRR_MAX_YREL_JUMP or
            abs(v_rel - prev.vRel) > DELPHI_MRR_MAX_VREL_JUMP)

  def _build_ret(self) -> structs.RadarData:
    ret = structs.RadarData()
    if not self.rcp.can_valid:
      ret.errors.canError = True
    if self.radar == RADAR.DELPHI_MRR:
      self._check_mrr_faults(ret)
    ret.points = list(self.pts.values())
    return ret

  def _check_mrr_faults(self, ret: structs.RadarData) -> None:
    status = self.rcp.vl.get("MRR_Status_Radar")
    if status is None:
      return
    if any(status[sig] for sig in MRR_FAULT_SIGNALS):
      ret.errors.radarFault = True

  def update(self, can_strings):
    if self.rcp is None:
      return super().update(None)

    vls = self.rcp.update(can_strings)
    self.updated_messages.update(vls)

    if self.trigger_msg not in self.updated_messages:
      # Keep publishing the last cluster set between MRR scan cycles
      if self.radar == RADAR.DELPHI_MRR and self.pts:
        return self._build_ret()
      return None

    updated_messages = set(self.updated_messages)
    self.updated_messages.clear()

    ret = structs.RadarData()
    if not self.rcp.can_valid:
      ret.errors.canError = True

    if self.radar == RADAR.DELPHI_ESR:
      self._update_delphi_esr(updated_messages)
    elif self.radar == RADAR.DELPHI_MRR:
      self._check_mrr_faults(ret)
      _update = self._update_delphi_mrr(ret)
      if not _update:
        if self.pts:
          ret.points = list(self.pts.values())
          return ret
        return None

    ret.points = list(self.pts.values())
    return ret

  def _update_delphi_esr(self, updated_messages: set[int]):
    del updated_messages  # trigger frame: refresh all slots from latest CAN parse
    for ii in DELPHI_ESR_RADAR_MSGS:
      if ii not in self.rcp.vl:
        continue
      cpt = self.rcp.vl[ii]

      if cpt['X_Rel'] > 0.00001:
        self.valid_cnt[ii] = min(self.valid_cnt[ii] + 1, 10)
      else:
        self.valid_cnt[ii] = max(self.valid_cnt[ii] - 1, 0)

      if self.valid_cnt[ii] > 0:
        if ii not in self.pts:
          self.pts[ii] = structs.RadarData.RadarPoint()
          self.pts[ii].trackId = self.track_id
          self.track_id += 1
        self.pts[ii].dRel = cpt['X_Rel']  # from front of car
        self.pts[ii].yRel = cpt['X_Rel'] * cpt['Angle'] * CV.DEG_TO_RAD  # in car frame's y axis, left is positive
        self.pts[ii].vRel = cpt['V_Rel']
        self.pts[ii].aRel = float('nan')
        self.pts[ii].yvRel = float('nan')
        self.pts[ii].measured = True
      else:
        if ii in self.pts:
          del self.pts[ii]

  def _update_delphi_mrr(self, ret: structs.RadarData):
    headerScanIndex = int(self.rcp.vl["MRR_Header_InformationDetections"]['CAN_SCAN_INDEX']) & 0b11

    # Scan index should advance by 1 each header. Reverse often repeats the same index (stuck);
    # driving more often drops a header (skip). Require sustained faults before failing out.
    if not self._header_scan_initialized:
      self.prev_headerScanIndex = headerScanIndex
      self._header_scan_initialized = True
    else:
      expected = (self.prev_headerScanIndex + 1) % 4
      if headerScanIndex == expected:
        self.radar_stuck_cnt = 0
        self.radar_skip_cnt = 0
      elif headerScanIndex == self.prev_headerScanIndex:
        self.radar_stuck_cnt += 1
        self.radar_skip_cnt = 0
      else:
        self.radar_skip_cnt += 1
        self.radar_stuck_cnt = 0
      self.prev_headerScanIndex = headerScanIndex

    if self.radar_stuck_cnt >= DELPHI_MRR_STUCK_THRESHOLD or self.radar_skip_cnt >= DELPHI_MRR_SKIP_THRESHOLD:
      self._reset_mrr_tracks()
      ret.errors.radarUnavailableTemporary = True
      return True

    # Brief stuck/skip: keep last tracks, do not ingest this cycle
    if self.radar_stuck_cnt > 0 or self.radar_skip_cnt > 0:
      return False

    # Use short-range scan 0 (~42 m) plus scan 2/3 for close stationary leads; scan 2/3 have +-60 m/s Doppler
    if headerScanIndex not in (0, 2, 3):
      return False

    if DELPHI_MRR_RADAR_RANGE_COVERAGE[headerScanIndex] != int(self.rcp.vl["MRR_Header_SensorCoverage"]["CAN_RANGE_COVERAGE"]):
      self.scan_index_invalid_cnt += 1
    else:
      self.scan_index_invalid_cnt = 0

    # Rarely MRR_Header_InformationDetections can fail to send a message. The scan index is skipped in this case
    if self.scan_index_invalid_cnt >= 5:
      ret.errors.wrongConfig = True

    for ii in range(1, DELPHI_MRR_RADAR_MSG_COUNT + 1):
      msg = self.rcp.vl[f"MRR_Detection_{ii:03d}"]

      # SCAN_INDEX rotates through 0..3 on each message for different measurement modes
      # Indexes 0 and 2 have a max range of ~40m, 1 and 3 are ~170m (MRR_Header_SensorCoverage->CAN_RANGE_COVERAGE)
      # Indexes 0 and 1 have a Doppler coverage of +-71 m/s, 2 and 3 have +-60 m/s
      scanIndex = msg[f"CAN_SCAN_INDEX_2LSB_{ii:02d}"]

      # Throw out old measurements. Very unlikely to happen, but is proper behavior
      if scanIndex != headerScanIndex:
        continue

      valid = bool(msg[f"CAN_DET_VALID_LEVEL_{ii:02d}"])

      # Long range measurement mode is more sensitive and can detect the road surface
      dist = msg[f"CAN_DET_RANGE_{ii:02d}"]  # m [0|255.984]
      if scanIndex in (1, 3) and dist < DELPHI_MRR_MIN_LONG_RANGE_DIST:
        valid = False

      if valid:
        azimuth = msg[f"CAN_DET_AZIMUTH_{ii:02d}"]              # rad [-3.1416|3.13964]
        distRate = msg[f"CAN_DET_RANGE_RATE_{ii:02d}"]          # m/s [-128|127.984]
        dRel = cos(azimuth) * dist                              # m from front of car
        yRel = -sin(azimuth) * dist                             # in car frame's y axis, left is positive

        self.points.append([dRel, yRel * 2, distRate * 2])

    # Cluster and publish using stored points once we've cycled through all 4 scan modes
    if headerScanIndex != 3:
      return False

    # Cluster points from this cycle against the centroids from the previous cycle
    prev_keys = [[p.dRel, p.yRel * 2, p.vRel * 2] for p in self.clusters]
    labels = cluster_points(prev_keys, self.points, DELPHI_MRR_CLUSTER_THRESHOLD)

    points_by_track_id = defaultdict(list)
    for idx, label in enumerate(labels):
      if label != -1:
        points_by_track_id[self.clusters[label].trackId].append(self.points[idx])
      else:
        points_by_track_id[self.track_id].append(self.points[idx])
        self.track_id += 1

    new_pts: dict[int, structs.RadarData.RadarPoint] = {}
    self.clusters = []
    alive_ids: set[int] = set()

    for track_id, pts in points_by_track_id.items():
      dRel_vals = [p[0] for p in pts]
      min_dRel = min(dRel_vals)
      dRel = sum(dRel_vals) / len(dRel_vals)

      yRel = [p[1] for p in pts]
      yRel = sum(yRel) / len(yRel) / 2

      vRel = [p[2] for p in pts]
      vRel = sum(vRel) / len(vRel) / 2

      # Always keep cluster for next-cycle association (even if not yet published)
      self.clusters.append(Cluster(dRel=dRel, yRel=yRel, vRel=vRel, trackId=track_id))
      alive_ids.add(track_id)

      prev = self._track_filters.get(track_id)
      pub_dRel, pub_yRel, pub_vRel = min_dRel, yRel, vRel

      if prev is None:
        # New track: start confirmation, do not publish single-frame clutter
        self._track_filters[track_id] = _TrackFilter(dRel=min_dRel, yRel=yRel, vRel=vRel, confirm_cnt=1)
        continue

      if self._mrr_kinematic_jump(prev, min_dRel, yRel, vRel):
        if prev.published:
          # Hold last good measurement for one cycle (reject spike)
          pub_dRel, pub_yRel, pub_vRel = prev.dRel, prev.yRel, prev.vRel
        else:
          # Unconfirmed + unstable → restart confirmation
          self._track_filters[track_id] = _TrackFilter(dRel=min_dRel, yRel=yRel, vRel=vRel, confirm_cnt=1)
          continue
      else:
        prev.dRel = min_dRel
        prev.yRel = yRel
        prev.vRel = vRel
        prev.confirm_cnt += 1
        if prev.confirm_cnt >= DELPHI_MRR_CONFIRM_CYCLES:
          prev.published = True
        pub_dRel, pub_yRel, pub_vRel = prev.dRel, prev.yRel, prev.vRel

      if not self._track_filters[track_id].published:
        continue

      if track_id not in self.pts:
        self.pts[track_id] = structs.RadarData.RadarPoint(measured=True, aRel=float('nan'), yvRel=float('nan'))

      pt = self.pts[track_id]
      pt.dRel = pub_dRel
      pt.yRel = pub_yRel
      pt.vRel = pub_vRel
      pt.trackId = track_id
      new_pts[track_id] = pt

    # Drop filters for tracks that disappeared this cycle
    for tid in list(self._track_filters.keys()):
      if tid not in alive_ids:
        self._track_filters.pop(tid, None)

    self.pts = new_pts
    self.points = []

    return True
