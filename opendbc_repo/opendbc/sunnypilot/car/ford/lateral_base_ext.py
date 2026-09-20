"""
BluePilot Ford lateral control infrastructure (angle-primary).

Provides the shared SubMaster (modelV2 / liveParameters / selfdriveState / radarState /
liveDelay), the vehicle model, and the single measured-curvature source used by the
angle-primary lateral strategy (lateral_angle_ext.py). Mixed into CarController as
LateralBaseExt.
"""

import math
from collections import namedtuple

import cereal.messaging as messaging

from opendbc.car.vehicle_model import VehicleModel
from opendbc.sunnypilot.car.ford.values_ext import FordSafetyFlagsSP


# Result namedtuple returned by the lateral strategies.
LateralResult = namedtuple('LateralResult', [
  'apply_curvature',
  'curvature_rate',
  'path_offset',
  'path_angle',
  'ramp_type',
  'precision_type',
  'lateralUncertainty',
])


class LateralBaseExt:
  """BluePilot lateral infrastructure shared by the Ford angle-primary strategy.

  Mixed into CarController via multiple inheritance. Owns the SubMaster and vehicle
  model, and provides the measured-curvature source (get_current_curvature) used by the
  deviation clip, the stall detector, and the shadow curvature published to ford.h.
  """

  def __init__(self, CP, CP_SP):
    # SubMaster for model data, live parameters, and selfdrive state.
    self.sm = messaging.SubMaster(['modelV2', 'liveParameters', 'selfdriveState', 'radarState', 'liveDelay'])
    self.VM = VehicleModel(CP)
    self.model = None
    self.lp = None
    self.ss = None

    # BluePilot: steering-angle curvature measurement (bad-yaw-sensor workaround).
    # Mirrors the STEER_ANGLE_CURVATURE flag the safety firmware reads from
    # current_safety_param_sp -- both layers must always agree, so this is init-time
    # state from CP_SP (set by _initialize_ford at car init), never a live Params read:
    # a live flip against stale firmware would fight the panda.
    self.bp_pinion_curvature_enabled = bool(
      CP_SP is not None and (CP_SP.safetyParam & FordSafetyFlagsSP.STEER_ANGLE_CURVATURE))

    # Shared lateral state consumed by the angle strategy (lateral_angle_ext.py).
    self.precision_type = 1  # 1=Precise, 0=Comfortable
    self.lateralUncertainty = 0.0
    self.bp_curvature_deviation_limited = False
    self.lane_change = False
    self.path_angle_last = 0.0
    # Lane change scaling breakpoints (angle mode reads lane_change_factor_low).
    self.lane_change_factor_bp = [4.4, 40.23]  # speed breakpoints (m/s)
    self.lane_change_factor_low = 0.95

  def get_current_curvature(self, CS):
    """Measured curvature of the car right now (OP sign convention).

    The single measurement source for every BluePilot lateral consumer: the deviation
    clip, the stall detector, and the shadow curvature published to ford.h's angle-mode
    deviation check. The default source is the RCM yaw rate -- the same family stock
    ford.h derives its angle_meas from. The shadow value judged against that check must
    always come from the same measurement as the check's own reference, so route all
    reads through here.

    With the steering-angle curvature measurement enabled (FordPrefSteerAngleCurvature
    toggle -> FordSafetyFlagsSP.STEER_ANGLE_CURVATURE), the pinion angle via the vehicle
    model is used instead: some vehicles (e.g. a 2021 Explorer with a faulty RCM)
    broadcast implausible VehYaw_W_Actl (sign-inverted vs IMU and steering geometry)
    while its CAN quality flag still reads OK. The pinion angle (SteeringPinion_Data,
    PSCM) is an equivalent measurement, independently validated against the comma IMU
    (corr +0.99), and the panda safety angle_meas switches to the same source (see
    safety/modes/ford.h) -- the layers always agree. angleOffsetDeg/roll come from
    liveParameters (paramsd, IMU-derived, not the car yaw sensor).
    """
    if self.bp_pinion_curvature_enabled:
      angle_offset_deg = self.lp.angleOffsetDeg if self.lp is not None else 0.0
      roll = self.lp.roll if self.lp is not None else 0.0
      return -self.VM.calc_curvature(math.radians(CS.out.steeringAngleDeg - angle_offset_deg),
                                     CS.out.vEgoRaw, roll)
    return -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

  def update_sm(self):
    """Update SubMaster and vehicle model. Called each frame before lateral/long update."""
    self.sm.update(0)

    if self.sm.updated['modelV2']:
      self.model = self.sm['modelV2']
    if self.sm.updated['liveParameters']:
      self.lp = self.sm['liveParameters']
    if self.sm.updated['selfdriveState']:
      self.ss = self.sm['selfdriveState']

    if self.lp is not None:
      x = max(self.lp.stiffnessFactor, 0.1)
      sr = max(self.lp.steerRatio, 0.1)
      self.VM.update_params(x, sr)

  def update(self, CC, CS, actuators, apply_curvature_last, CP):
    # BluePilot: curvature-primary lateral control was removed; angle-primary
    # (lateral_angle_ext.py) is the only lateral strategy. Kept as a placeholder
    # to document the shared LateralResult interface.
    raise NotImplementedError("curvature-primary lateral control was removed")
