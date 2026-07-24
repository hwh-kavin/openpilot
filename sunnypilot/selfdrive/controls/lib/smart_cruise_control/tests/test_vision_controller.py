"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

import cereal.messaging as messaging
from cereal import custom, log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import (
  SmartCruiseControlVision,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_scc_lat_accel_max

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState


def generate_modelV2(speed=30.0, kappa=0.001):
  """Build modelV2 with constant plan curvature κ via yaw_rate = κ * v."""
  model = messaging.new_message('modelV2')
  n = len(ModelConstants.T_IDXS)
  t = np.array(ModelConstants.T_IDXS)
  position = log.XYZTData.new_message()
  position.x = [float(x) for x in speed * t]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  orientation.x = [0.0 for _ in range(n)]
  orientation.y = [0.0 for _ in range(n)]
  model.modelV2.orientation = orientation
  orientationRate = log.XYZTData.new_message()
  orientationRate.z = [float(kappa * speed) for _ in range(n)]
  orientationRate.zStd = [0.0 for _ in range(n)]
  model.modelV2.orientationRate = orientationRate
  velocity = log.XYZTData.new_message()
  velocity.x = [float(speed) for _ in range(n)]
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [0.0 for _ in range(n)]
  acceleration.y = [0.0 for _ in range(n)]
  model.modelV2.acceleration = acceleration
  return model


def generate_carState():
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = 30.0
  car_state.carState.standstill = False
  car_state.carState.vCruise = 50.0 * 3.6
  car_state.carState.steeringAngleDeg = 0.0
  return car_state


def generate_controlsState():
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = 0.0
  controls_state.controlsState.desiredCurvature = 0.0
  return controls_state


def generate_liveParameters():
  lp = messaging.new_message('liveParameters')
  lp.liveParameters.angleOffsetDeg = 0.0
  return lp


def generate_selfdriveState():
  ss = messaging.new_message('selfdriveState')
  ss.selfdriveState.personality = log.LongitudinalPersonality.standard
  return ss


def _ford_like_cp():
  from opendbc.car import structs
  CP = structs.CarParams()
  CP.steerRatio = 14.8
  CP.wheelbase = 2.7
  return CP


class TestSmartCruiseControlVision:

  def setup_method(self):
    self.params = Params()
    self.reset_params()
    self.scc_v = SmartCruiseControlVision(_ford_like_cp())

    mdl = generate_modelV2()
    cs = generate_carState()
    controls_state = generate_controlsState()
    ss = generate_selfdriveState()
    lp = generate_liveParameters()
    self.sm = {
      'modelV2': mdl.modelV2,
      'carState': cs.carState,
      'controlsState': controls_state.controlsState,
      'selfdriveState': ss.selfdriveState,
      'liveParameters': lp.liveParameters,
    }

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlVision", True, block=True)

  def test_initial_state(self):
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlVision", False, block=True)
    self.scc_v.enabled = self.params.get_bool("SmartCruiseControlVision")

    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active

  def test_disabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, False, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.enabled

  def test_entering_from_future_kappa(self):
    v_ego = 27.8
    # Sharp plan κ: v_corner = sqrt(1.65/0.008) ≈ 14.4 m/s << v_ego
    self.sm['modelV2'] = generate_modelV2(speed=v_ego, kappa=0.008).modelV2
    self.sm['controlsState'].curvature = 0.0
    self.sm['controlsState'].desiredCurvature = 0.0

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.has_curve_constraint
    assert self.scc_v.v_ego > self.scc_v.v_passable
    assert self.scc_v.state == VisionState.entering
    assert self.scc_v.is_active
    a_lat = get_scc_lat_accel_max(log.LongitudinalPersonality.standard)
    assert self.scc_v.v_passable == pytest.approx(max(MIN_V, (a_lat / 0.008) ** 0.5), rel=0.05)

  def test_mild_kappa_stays_enabled(self):
    v_ego = 27.8
    self.sm['modelV2'] = generate_modelV2(speed=v_ego, kappa=0.0004).modelV2
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.state == VisionState.enabled
    assert not self.scc_v.is_active

  def test_entering_from_current_steer_kappa(self):
    v_ego = 27.8
    # Mild future plan; hard current desired curvature
    self.sm['modelV2'] = generate_modelV2(speed=v_ego, kappa=0.0003).modelV2
    self.sm['controlsState'].desiredCurvature = 0.01
    self.sm['controlsState'].curvature = 0.01

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.kappa_now == pytest.approx(0.01, abs=1e-5)
    assert self.scc_v.has_curve_constraint
    assert self.scc_v.state == VisionState.entering

  def test_v_target_never_below_min_v(self):
    v_ego = 27.8
    self.sm['modelV2'] = generate_modelV2(speed=v_ego, kappa=0.2).modelV2
    self.sm['controlsState'].curvature = 0.2
    self.sm['controlsState'].desiredCurvature = 0.2

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    assert self.scc_v.v_target >= MIN_V
    assert self.scc_v.v_passable >= MIN_V
    if self.scc_v.is_active:
      assert self.scc_v.output_v_target >= MIN_V

  def test_steer_angle_raises_kappa_now(self):
    v_ego = 20.0
    self.sm['modelV2'] = generate_modelV2(speed=v_ego, kappa=0.0003).modelV2
    self.sm['controlsState'].curvature = 0.0
    self.sm['controlsState'].desiredCurvature = 0.0
    self.sm['carState'].steeringAngleDeg = 120.0

    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    assert self.scc_v.kappa_now > 0.01
    assert self.scc_v.actual_lat_acc > 0.5


# pytest.approx used above
import pytest  # noqa: E402
