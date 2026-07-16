#!/usr/bin/env python3
import math
import numpy as np
from collections import deque
from typing import Any

import capnp
from cereal import messaging, log, car, custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.common.simple_kalman import KF1D

from opendbc.car import structs
from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP


# Default lead acceleration decay set to 50% at 1s
_LEAD_ACCEL_TAU = 1.5

# radar tracks
SPEED, ACCEL = 0, 1     # Kalman filter states enum

# stationary qualification parameters
V_EGO_STATIONARY = 4.   # no stationary object flag below this speed

RADAR_TO_CENTER = 2.7   # (deprecated) RADAR is ~ 2.7m ahead from center of car
RADAR_TO_CAMERA = 1.52  # RADAR is ~ 1.5m ahead from center of mesh frame (default / Toyota)

# Ford Delphi radar reports dRel from the front bumper; vision uses the camera origin
FORD_RADAR_TO_CAMERA = 1.35
FORD_LATERAL_MATCH_GATE = 2.0
FORD_LOW_SPEED_MIN_DREL = 0.5
FORD_LOW_SPEED_MAX_DREL = 25.0   # parking / creep: trust radar lead within 25 m
FORD_V_EGO_STATIONARY = 6.0      # ~22 km/h, covers parking-lot creep
FORD_LOW_SPEED_LATERAL = 1.5       # wider gate for bumper-mounted MRR at low speed

# Ford: vision gates empty-road radar FPs; close-range / creep prefers OEM radar distance
FORD_RADAR_CLOSE_DIST = 30.0       # within this, prefer radar distance over vision
FORD_RADAR_SEARCH_MARGIN = 20.0    # search radar around vision distance when unmatched
FORD_RADAR_SEARCH_MAX = 120.0
# Vision near-range distance is often biased high; allow large mismatch for association
FORD_MATCH_DIST_SANE = 15.0
# Above this ego speed, MRR stationary in-path points are treated as clutter unless vision agrees
FORD_REJECT_STATIONARY_VEGO = 8.0          # ~29 km/h
FORD_STATIONARY_VLEAD_MAX = 2.0            # m/s — radar "stopped"
FORD_VISION_STATIONARY_VLEAD_MAX = 4.0     # vision must also look slow to accept stopped radar
FORD_STATIONARY_DIST_AGREE = 12.0          # |dRel_radar - dRel_vision| must be within this
FORD_STATIONARY_MIN_TRACK_CNT = 8          # ignore brand-new stopped tracks at speed
FORD_MATCH_VEL_SANE = 6.0                  # tighter than default 10 m/s for vision↔radar match


def get_radar_to_camera(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_RADAR_TO_CAMERA
  return RADAR_TO_CAMERA


def get_lateral_match_gate(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_LATERAL_MATCH_GATE
  return 2.5


def get_low_speed_min_drel(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_LOW_SPEED_MIN_DREL
  return 0.75


def get_low_speed_max_drel(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_LOW_SPEED_MAX_DREL
  return 25.0


def get_v_ego_stationary(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_V_EGO_STATIONARY
  return V_EGO_STATIONARY


def get_low_speed_lateral(CP: structs.CarParams) -> float:
  if CP.brand == "ford":
    return FORD_LOW_SPEED_LATERAL
  return 1.0


class KalmanParams:
  def __init__(self, dt: float):
    # Lead Kalman Filter params, calculating K from A, C, Q, R requires the control library.
    # hardcoding a lookup table to compute K for values of radar_ts between 0.01s and 0.2s
    assert dt > .01 and dt < .2, "Radar time step must be between .01s and 0.2s"
    self.A = [[1.0, dt], [0.0, 1.0]]
    self.C = [1.0, 0.0]
    #Q = np.matrix([[10., 0.0], [0.0, 100.]])
    #R = 1e3
    #K = np.matrix([[ 0.05705578], [ 0.03073241]])
    dts = [i * 0.01 for i in range(1, 21)]
    K0 = [0.12287673, 0.14556536, 0.16522756, 0.18281627, 0.1988689,  0.21372394,
          0.22761098, 0.24069424, 0.253096,   0.26491023, 0.27621103, 0.28705801,
          0.29750003, 0.30757767, 0.31732515, 0.32677158, 0.33594201, 0.34485814,
          0.35353899, 0.36200124]
    K1 = [0.29666309, 0.29330885, 0.29042818, 0.28787125, 0.28555364, 0.28342219,
          0.28144091, 0.27958406, 0.27783249, 0.27617149, 0.27458948, 0.27307714,
          0.27162685, 0.27023228, 0.26888809, 0.26758976, 0.26633338, 0.26511557,
          0.26393339, 0.26278425]
    self.K = [[np.interp(dt, dts, K0)], [np.interp(dt, dts, K1)]]


class Track:
  def __init__(self, identifier: int, v_lead: float, kalman_params: KalmanParams):
    self.identifier = identifier
    self.cnt = 0
    self.aLeadTau = FirstOrderFilter(_LEAD_ACCEL_TAU, 0.45, DT_MDL)
    self.K_A = kalman_params.A
    self.K_C = kalman_params.C
    self.K_K = kalman_params.K
    self.kf = KF1D([[v_lead], [0.0]], self.K_A, self.K_C, self.K_K)

  def update(self, d_rel: float, y_rel: float, v_rel: float, v_lead: float, measured: float):
    # relative values, copy
    self.dRel = d_rel   # LONG_DIST
    self.yRel = y_rel   # -LAT_DIST
    self.vRel = v_rel   # REL_SPEED
    self.vLead = v_lead
    self.measured = measured   # measured or estimate

    # computed velocity and accelerations
    if self.cnt > 0:
      self.kf.update(self.vLead)

    self.vLeadK = float(self.kf.x[SPEED][0])
    self.aLeadK = float(self.kf.x[ACCEL][0])

    # Learn if constant acceleration
    if abs(self.aLeadK) < 0.5:
      self.aLeadTau.x = _LEAD_ACCEL_TAU
    else:
      self.aLeadTau.update(0.0)

    self.cnt += 1

  def get_RadarState(self, model_prob: float = 0.0):
    return {
      "dRel": float(self.dRel),
      "yRel": float(self.yRel),
      "vRel": float(self.vRel),
      "vLead": float(self.vLead),
      "vLeadK": float(self.vLeadK),
      "aLeadK": float(self.aLeadK),
      "aLeadTau": float(self.aLeadTau.x),
      "status": True,
      "fcw": self.is_potential_fcw(model_prob),
      "modelProb": model_prob,
      "radar": True,
      "radarTrackId": self.identifier,
    }

  def potential_low_speed_lead(self, v_ego: float, min_d_rel: float = 0.75, max_d_rel: float = 25.0,
                               lateral_max: float = 1.0, v_ego_stationary: float = V_EGO_STATIONARY):
    # stop for stuff in front of you and low speed, even without model confirmation
    # Radar points closer than min_d_rel are often glitches (0.75m on Toyota, 0.5m on Ford MRR)
    return (abs(self.yRel) < lateral_max and (v_ego < v_ego_stationary) and
            (min_d_rel < self.dRel < max_d_rel))

  def is_potential_fcw(self, model_prob: float):
    return model_prob > .9

  def __str__(self):
    ret = f"x: {self.dRel:4.1f}  y: {self.yRel:4.1f}  v: {self.vRel:4.1f}  a: {self.aLeadK:4.1f}"
    return ret


def laplacian_pdf(x: float, mu: float, b: float):
  b = max(b, 1e-4)
  return math.exp(-abs(x-mu)/b)


def match_vision_to_track(v_ego: float, lead: capnp._DynamicStructReader, tracks: dict[int, Track],
                          radar_to_camera: float = RADAR_TO_CAMERA, lateral_gate: float = 2.5,
                          dist_sane_min: float = 5.0, vel_sane_max: float = 10.0,
                          stationary_vlead_max: float | None = None,
                          reject_stationary_vego: float | None = None):
  offset_vision_dist = lead.x[0] - radar_to_camera
  vision_y = -lead.y[0]

  # Pre-filter by lane proximity before scoring (helps Ford MRR multi-cluster scenes)
  candidates = [c for c in tracks.values() if abs(c.yRel - vision_y) < lateral_gate]
  if not candidates:
    candidates = list(tracks.values())

  def prob(c):
    prob_d = laplacian_pdf(c.dRel, offset_vision_dist, lead.xStd[0])
    prob_y = laplacian_pdf(c.yRel, vision_y, lead.yStd[0])
    prob_v = laplacian_pdf(c.vRel + v_ego, lead.v[0], lead.vStd[0])

    # This isn't exactly right, but it's a good heuristic
    return prob_d * prob_y * prob_v

  track = max(candidates, key=prob)

  # if no 'sane' match is found return -1
  # stationary radar points can be false positives
  # Ford: raise dist_sane_min — vision near-range dRel is often biased high
  dist_sane = abs(track.dRel - offset_vision_dist) < max([(offset_vision_dist)*.25, dist_sane_min])
  vel_err = abs(track.vRel + v_ego - lead.v[0])
  v_lead_track = v_ego + track.vRel
  # Moving radar tracks get a looser gate; near-stationary must closely match vision speed
  # (otherwise MRR roadside clutter associates to a moving vision lead and trips FCW).
  if (stationary_vlead_max is not None and reject_stationary_vego is not None and
      v_lead_track <= stationary_vlead_max and v_ego >= reject_stationary_vego):
    vel_sane = vel_err < min(vel_sane_max, 4.0)
  else:
    vel_sane = (vel_err < vel_sane_max) or (v_lead_track > 3)
  if dist_sane and vel_sane:
    return track
  else:
    return None


def get_RadarState_from_vision(lead_msg: capnp._DynamicStructReader, v_ego: float, model_v_ego: float,
                               lead_prob: float, radar_to_camera: float = RADAR_TO_CAMERA):
  lead_v_rel_pred = lead_msg.v[0] - model_v_ego
  return {
    "dRel": float(lead_msg.x[0] - radar_to_camera),
    "yRel": float(-lead_msg.y[0]),
    "vRel": float(lead_v_rel_pred),
    "vLead": float(v_ego + lead_v_rel_pred),
    "vLeadK": float(v_ego + lead_v_rel_pred),
    "aLeadK": float(lead_msg.a[0]),
    "aLeadTau": 0.3,
    "fcw": False,
    "modelProb": float(lead_prob),
    "status": True,
    "radar": False,
    "radarTrackId": -1,
  }


def _vision_matched_track(v_ego: float, ready: bool, tracks: dict[int, Track],
                          lead_msg: capnp._DynamicStructReader, lead_prob: float,
                          radar_to_camera: float, lateral_gate: float,
                          dist_sane_min: float = 5.0, vel_sane_max: float = 10.0,
                          stationary_vlead_max: float | None = None,
                          reject_stationary_vego: float | None = None) -> Track | None:
  if len(tracks) > 0 and ready and lead_prob > .5:
    return match_vision_to_track(v_ego, lead_msg, tracks, radar_to_camera, lateral_gate,
                                 dist_sane_min, vel_sane_max, stationary_vlead_max,
                                 reject_stationary_vego)
  return None


def _closest_in_path_radar_track(tracks: dict[int, Track], lateral_gate: float,
                                 min_d_rel: float, max_d_rel: float) -> Track | None:
  candidates = [c for c in tracks.values()
                if abs(c.yRel) < lateral_gate and min_d_rel < c.dRel < max_d_rel]
  if not candidates:
    return None
  return min(candidates, key=lambda c: c.dRel)


def _ford_accept_radar_lead(v_ego: float, radar: dict[str, Any], vision: dict[str, Any],
                            track_cnt: int | None = None) -> bool:
  """Reject MRR stationary clutter that falsely looks like an in-path stopped car.

  At speed, Delphi MRR often reports roadside/overhead objects as vLead≈0 ahead of the
  bumper. Those lock longitudinal MPC into a crash prediction and fire FCW (also shown
  on the Ford IPC). Only keep stopped radar when vision agrees on a nearby slow lead,
  and the track has been stable for a few frames.
  """
  if not radar.get('status'):
    return False
  if v_ego < FORD_REJECT_STATIONARY_VEGO:
    return True

  v_lead_r = float(radar['vLead'])
  if v_lead_r > FORD_STATIONARY_VLEAD_MAX:
    return True

  # Brand-new stopped tracks at speed are almost always clutter
  if track_cnt is not None and track_cnt < FORD_STATIONARY_MIN_TRACK_CNT:
    return False

  if not vision.get('status'):
    return False

  v_lead_v = float(vision['vLead'])
  if v_lead_v > FORD_VISION_STATIONARY_VLEAD_MAX:
    # Vision says the lead is moving; radar stopped point is a mismatch / ghost
    return False

  if abs(float(radar['dRel']) - float(vision['dRel'])) > FORD_STATIONARY_DIST_AGREE:
    # Vision lead is much farther (or nearer) than the stopped radar blob
    return False

  return True


def get_ford_lead(v_ego: float, ready: bool, tracks: dict[int, Track],
                  lead_msg: capnp._DynamicStructReader, model_v_ego: float, lead_prob: float,
                  CP: structs.CarParams, CP_SP: structs.CarParamsSP) -> dict[str, Any]:
  """Ford lead: vision gates empty-road radar FPs; creep/close range uses OEM radar distance."""
  radar_to_camera = get_radar_to_camera(CP)
  lateral_gate = get_lateral_match_gate(CP)
  low_speed_min_drel = get_low_speed_min_drel(CP)
  low_speed_lateral = get_low_speed_lateral(CP)
  v_ego_stationary = get_v_ego_stationary(CP)
  vision_has_lead = ready and lead_prob > .5

  vision_lead: dict[str, Any] = {'status': False}
  if vision_has_lead:
    vision_lead = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego, lead_prob, radar_to_camera)

  # Match with relaxed distance sanity: vision near-range dRel is often several meters high.
  # Tighten velocity for stationary MRR points so roadside clutter cannot bind to a moving vision lead.
  track = _vision_matched_track(
    v_ego, ready, tracks, lead_msg, lead_prob,
    radar_to_camera, lateral_gate, FORD_MATCH_DIST_SANE, FORD_MATCH_VEL_SANE,
    FORD_STATIONARY_VLEAD_MAX, FORD_REJECT_STATIONARY_VEGO,
  )
  matched_radar: dict[str, Any] = {'status': False}
  matched_cnt: int | None = None
  if track is not None:
    matched_radar = track.get_RadarState(lead_prob)
    matched_radar = get_custom_yrel(CP, CP_SP, matched_radar, lead_msg)
    matched_cnt = track.cnt

  # Closest in-path radar for creep / park / close follow (does not require vision assoc)
  creep = v_ego < v_ego_stationary
  lat = low_speed_lateral if creep else lateral_gate
  max_d = FORD_LOW_SPEED_MAX_DREL if creep else FORD_RADAR_CLOSE_DIST
  if vision_has_lead and vision_lead.get('status'):
    # Prefer radar near the real bumper; vision dRel can be far too large
    max_d = max(max_d, min(float(vision_lead['dRel']) + FORD_RADAR_SEARCH_MARGIN, FORD_RADAR_SEARCH_MAX))
  closest = _closest_in_path_radar_track(tracks, lat, low_speed_min_drel, max_d)
  closest_radar: dict[str, Any] = {'status': False}
  closest_cnt: int | None = None
  if closest is not None:
    closest_radar = closest.get_RadarState(lead_prob if vision_has_lead else 0.0)
    closest_cnt = closest.cnt

  # --- Creep / standstill: OEM radar distance is the authority when available ---
  if creep:
    if closest_radar.get('status'):
      return closest_radar
    if matched_radar.get('status') and matched_radar['dRel'] <= FORD_LOW_SPEED_MAX_DREL:
      return matched_radar
    # No near radar: use vision only if it reports a close lead; else clear (allow start)
    if vision_lead.get('status') and vision_lead['dRel'] < FORD_LOW_SPEED_MAX_DREL:
      return vision_lead
    return {'status': False}

  # --- Above creep: ignore unmatched radar when vision sees no car (empty-road FP) ---
  if not vision_has_lead:
    return {'status': False}

  matched_ok = _ford_accept_radar_lead(v_ego, matched_radar, vision_lead, matched_cnt)
  closest_ok = _ford_accept_radar_lead(v_ego, closest_radar, vision_lead, closest_cnt)

  # Vision has lead: prefer radar distance within close range; vision is fallback only
  if matched_ok and matched_radar['dRel'] <= FORD_RADAR_CLOSE_DIST:
    return matched_radar
  if closest_ok and closest_radar['dRel'] <= FORD_RADAR_CLOSE_DIST:
    # Prefer nearer of unmatched in-path radar vs vision
    if (not vision_lead.get('status')) or closest_radar['dRel'] < vision_lead['dRel']:
      return closest_radar
  if matched_ok:
    return matched_radar
  if vision_lead.get('status'):
    return vision_lead
  return {'status': False}


def get_lead(v_ego: float, ready: bool, tracks: dict[int, Track], lead_msg: capnp._DynamicStructReader,
             model_v_ego: float, lead_prob: float, CP: structs.CarParams, CP_SP: structs.CarParamsSP,
             low_speed_override: bool = True) -> dict[str, Any]:
  radar_to_camera = get_radar_to_camera(CP)
  lateral_gate = get_lateral_match_gate(CP)
  low_speed_min_drel = get_low_speed_min_drel(CP)
  low_speed_max_drel = get_low_speed_max_drel(CP)
  v_ego_stationary = get_v_ego_stationary(CP)
  low_speed_lateral = get_low_speed_lateral(CP)

  # Determine leads, this is where the essential logic happens
  track = _vision_matched_track(v_ego, ready, tracks, lead_msg, lead_prob, radar_to_camera, lateral_gate)

  lead_dict = {'status': False}
  if track is not None:
    lead_dict = track.get_RadarState(lead_prob)
    lead_dict = get_custom_yrel(CP, CP_SP, lead_dict, lead_msg)
  elif (track is None) and ready and (lead_prob > .5):
    lead_dict = get_RadarState_from_vision(lead_msg, v_ego, model_v_ego, lead_prob, radar_to_camera)

  if low_speed_override:
    low_speed_tracks = [c for c in tracks.values()
                        if c.potential_low_speed_lead(v_ego, low_speed_min_drel, low_speed_max_drel,
                                                      low_speed_lateral, v_ego_stationary)]
    if len(low_speed_tracks) > 0:
      closest_track = min(low_speed_tracks, key=lambda c: c.dRel)

      # Only choose new track if it is actually closer than the previous one
      if (not lead_dict['status']) or (closest_track.dRel < lead_dict['dRel']):
        lead_dict = closest_track.get_RadarState()

  return lead_dict


def get_custom_yrel(CP: structs.CarParams, CP_SP: structs.CarParamsSP, lead_dict: dict[str, Any],
                    lead_msg: capnp._DynamicStructReader) -> dict[str, Any]:
  if CP.brand == "hyundai" and (CP_SP.flags & HyundaiFlagsSP.ENHANCED_SCC or
                                CP.flags & (HyundaiFlags.CANFD_CAMERA_SCC | HyundaiFlags.CAMERA_SCC)):
    lead_dict['yRel'] = float(-lead_msg.y[0])

  return lead_dict


class RadarD:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParams, delay: float = 0.0):
    self.CP = CP
    self.CP_SP = CP_SP

    self.current_time = 0.0

    self.tracks: dict[int, Track] = {}
    self.kalman_params = KalmanParams(DT_MDL)
    self.lead_prob_filters = [FirstOrderFilter(0.0, 0.2, DT_MDL) for _ in range(2)]

    self.v_ego = 0.0
    self.v_ego_hist = deque([0.0], maxlen=int(round(delay / DT_MDL))+1)
    self.last_v_ego_frame = -1

    self.radar_state: capnp._DynamicStructBuilder | None = None
    self.radar_state_valid = False

    self.ready = False

  def update(self, sm: messaging.SubMaster, rr: car.RadarData):
    self.ready = sm.seen['modelV2']
    self.current_time = 1e-9*max(sm.logMonoTime.values())

    if sm.recv_frame['carState'] != self.last_v_ego_frame:
      self.v_ego = sm['carState'].vEgo
      self.v_ego_hist.append(self.v_ego)
      self.last_v_ego_frame = sm.recv_frame['carState']

    ar_pts = {pt.trackId: [pt.dRel, pt.yRel, pt.vRel, pt.measured] for pt in rr.points}

    # *** remove missing points from meta data ***
    for ids in list(self.tracks.keys()):
      if ids not in ar_pts:
        self.tracks.pop(ids, None)

    # *** compute the tracks ***
    for ids in ar_pts:
      rpt = ar_pts[ids]

      # align v_ego by a fixed time to align it with the radar measurement
      v_lead = rpt[2] + self.v_ego_hist[0]

      # create the track if it doesn't exist or it's a new track
      if ids not in self.tracks:
        self.tracks[ids] = Track(ids, v_lead, self.kalman_params)
      self.tracks[ids].update(rpt[0], rpt[1], rpt[2], v_lead, rpt[3])

    # *** publish radarState ***
    self.radar_state_valid = sm.all_checks()
    self.radar_state = log.RadarState.new_message()
    self.radar_state.mdMonoTime = sm.logMonoTime['modelV2']
    self.radar_state.radarErrors = rr.errors
    self.radar_state.carStateMonoTime = sm.logMonoTime['carState']

    if len(sm['modelV2'].velocity.x):
      model_v_ego = sm['modelV2'].velocity.x[0]
    else:
      model_v_ego = self.v_ego
    leads_v3 = sm['modelV2'].leadsV3
    if len(leads_v3) > 1:
      for i in range(2):
        # Asymmetric filter on lead prob to keep lead when uncertain
        lead_prob = leads_v3[i].prob
        if lead_prob > self.lead_prob_filters[i].x:
          self.lead_prob_filters[i].x = lead_prob
        else:
          self.lead_prob_filters[i].update(lead_prob)

      if self.CP.brand == "ford":
        # Creep/close range: OEM radar distance; empty road: vision gates radar FPs
        self.radar_state.leadOne = get_ford_lead(
          self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego,
          self.lead_prob_filters[0].x, self.CP, self.CP_SP)
      else:
        self.radar_state.leadOne = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[0], model_v_ego,
                                            self.lead_prob_filters[0].x, self.CP, self.CP_SP,
                                            low_speed_override=True)
      self.radar_state.leadTwo = get_lead(self.v_ego, self.ready, self.tracks, leads_v3[1], model_v_ego,
                                          self.lead_prob_filters[1].x, self.CP, self.CP_SP,
                                          low_speed_override=False)

  def publish(self, pm: messaging.PubMaster):
    assert self.radar_state is not None

    radar_msg = messaging.new_message("radarState")
    radar_msg.valid = self.radar_state_valid
    radar_msg.radarState = self.radar_state
    pm.send("radarState", radar_msg)


# fuses camera and radar data for best lead detection
def main() -> None:
  config_realtime_process(5, Priority.CTRL_LOW)

  # wait for stats about the car to come in from controls
  cloudlog.info("radard is waiting for CarParams")
  CP = messaging.log_from_bytes(Params().get("CarParams", block=True), car.CarParams)
  cloudlog.info("radard got CarParams")

  cloudlog.info("radard is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(Params().get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("radard got CarParamsSP")

  # *** setup messaging
  sm = messaging.SubMaster(['modelV2', 'carState', 'liveTracks'], poll='modelV2')
  pm = messaging.PubMaster(['radarState'])

  RD = RadarD(CP, CP_SP, CP.radarDelay)

  while 1:
    sm.update()

    RD.update(sm, sm['liveTracks'])
    RD.publish(pm)


if __name__ == "__main__":
  main()
