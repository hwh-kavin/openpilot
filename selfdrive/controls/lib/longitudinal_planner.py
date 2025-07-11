#!/usr/bin/env python3
import math
import numpy as np
import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.acm import ACM
from openpilot.sunnypilot.selfdrive.controls.lib.accel_controller import AccelPersonalityController

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3

def limit_accel_in_turns(v_ego, lateral_curvature, a_target):
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * abs(lateral_curvature)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
  return [a_target[0], min(a_target[1], a_x_allowed)]

class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.prev_accel_clip = [ACCEL_MIN, ACCEL_MAX]
    self.output_a_target = 0.0
    self.output_should_stop = False
    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

    # 實例化移植模組
    self.acm = ACM()
    self.acm.enabled = True 
    self.accel_controller = AccelPersonalityController()

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
    else:
      x = v = a = np.zeros(len(T_IDXS_MPC))
    throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1] if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1 else 1.0
    return x, v, a, np.zeros(len(T_IDXS_MPC)), throttle_prob

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)
    accel_coast = get_coast_accel(sm['carControl'].orientationNED[1]) if len(sm['carControl'].orientationNED) == 3 else ACCEL_MAX
    v_ego = sm['carState'].vEgo
    v_cruise = min(sm['carState'].vCruise, V_CRUISE_MAX) * CV.KPH_TO_MS
    reset_state = (sm['controlsState'].longControlState == LongCtrlState.off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled) or sm['carState'].vCruise == V_CRUISE_UNSET
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # 呼叫 Accel Controller 更新設定
    self.accel_controller.update()
    if self.accel_controller.is_enabled():
        accel_clip = [ACCEL_MIN, self.accel_controller.get_max_accel(v_ego)]
        a_cruise_min_override = self.accel_controller.get_min_accel(v_ego)
    else:
        accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
        a_cruise_min_override = None

    accel_clip = limit_accel_in_turns(v_ego, sm['controlsState'].curvature, accel_clip)
    if reset_state:
      self.v_desired_filter.x = v_ego
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED
    if not self.allow_throttle:
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], max(accel_coast, accel_clip[0])])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)
    if sm['controlsState'].forceDecel: v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    # 傳入覆寫參數給 MPC
    self.mpc.update(sm['radarState'], v_cruise, personality=sm['selfdriveState'].personality, a_cruise_min_override=a_cruise_min_override)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)

    # ACM 模組介入：滑行與柔和煞車
    mode = 'blended' if self.is_e2e(sm) else 'acc'
    self.acm.update_states(sm['carControl'], sm['radarState'], reset_state, v_ego, v_cruise, mode=mode, personality=sm['selfdriveState'].personality)
    self.a_desired_trajectory = self.acm.update_a_desired_trajectory(self.a_desired_trajectory, v_ego=v_ego, lead=sm['radarState'].leadOne)

    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t = self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX, action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    
    if self.is_e2e(sm):
      output_a_target = min(sm['modelV2'].action.desiredAcceleration, output_a_target_mpc)
      self.output_should_stop = sm['modelV2'].action.shouldStop or output_should_stop_mpc
      if output_a_target < output_a_target_mpc: self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target, self.output_should_stop = output_a_target_mpc, output_should_stop_mpc

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'selfdriveState', 'radarState'])
    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time
    longitudinalPlan.speeds, longitudinalPlan.accels, longitudinalPlan.jerks = self.v_desired_trajectory.tolist(), self.a_desired_trajectory.tolist(), self.j_desired_trajectory.tolist()
    longitudinalPlan.hasLead, longitudinalPlan.longitudinalPlanSource, longitudinalPlan.fcw = sm['radarState'].leadOne.status, self.mpc.source, self.fcw
    longitudinalPlan.aTarget, longitudinalPlan.shouldStop, longitudinalPlan.allowBrake, longitudinalPlan.allowThrottle = float(self.output_a_target), bool(self.output_should_stop), True, bool(self.allow_throttle)
    pm.send('longitudinalPlan', plan_send)
    self.publish_longitudinal_plan_sp(sm, pm)
