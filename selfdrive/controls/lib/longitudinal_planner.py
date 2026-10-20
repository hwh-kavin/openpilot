#!/usr/bin/env python3
import math
import time
import numpy as np

import cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.radar_lead_filter import FilteredLead, RadarLeadFilter, get_vision_lead, lead_from_vision
from openpilot.selfdrive.controls.lib.radar_fallback import highway_radar_fallback_enabled, radar_state_degraded
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# BluePilot: e2e 纵向轻微负加速度死区，减少稳态巡航时模型小幅制动/补油的抖动。
# 仅在无停车请求、无前车风险、车速足够且允许给油时生效；最多把 -0.15 m/s² 提为 0。
E2E_LIGHT_DECEL_DEADBAND = 0.15  # m/s^2
E2E_DEADBAND_MIN_SPEED = 5.0     # m/s (~18 km/h)
E2E_DEADBAND_MIN_LEAD_GAP = 2.0  # s，前车时距小于该值时完全交给 MPC

# BluePilot: MPC 前车参数源切换速度带（车道收敛+滤波逻辑见 radar_lead_filter）
RADAR_LEAD_ENTER_V_MS = 60.0 * CV.KPH_TO_MS   # 低于此速度启用雷达点云前车参数
RADAR_LEAD_EXIT_V_MS = 62.0 * CV.KPH_TO_MS    # 高于此速度退出（迟滞）

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


class _RadarStateCopy:
  """radarState 替身：leadOne 替换为收敛+滤波后的 lead，leadTwo 原样传递。"""
  def __init__(self, leadOne, leadTwo):
    self.leadOne = leadOne
    self.leadTwo = leadTwo


def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """
  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
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

    # BluePilot: e2e 轻微负加速度死区状态（见 _apply_e2e_deadband）
    self._e2e_deadband_active = False
    self._mpc_lead_d_rel = float('inf')

    # BluePilot: MPC 前车参数的雷达点云处理（视觉车道收敛 + 滤波，与停车起步判定同源）
    self._radar_lead_filter = RadarLeadFilter()
    self._mpc_lead_status = False
    # 雷达点云前车参数启用条件：Ford + FordStockAccFusion + 车速 <60 km/h
    self._radar_lead_active = True
    self._is_ford = "FORD" in CP.carFingerprint
    self._fusion_params = None
    self._fusion_enabled_cached = True
    self._fusion_checked_t = 0.0
    self._highway_radar_fallback_active = False
    if self._is_ford:
      try:
        self._fusion_params = Params()
      except Exception:
        self._fusion_params = None

  @staticmethod
  def parse_model(model_msg):
    if (len(model_msg.position.x) == ModelConstants.IDX_N and
      len(model_msg.velocity.x) == ModelConstants.IDX_N and
      len(model_msg.acceleration.x) == ModelConstants.IDX_N):
      x = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.position.x)
      v = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.velocity.x)
      a = np.interp(T_IDXS_MPC, ModelConstants.T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    if len(model_msg.meta.disengagePredictions.gasPressProbs) > 1:
      throttle_prob = model_msg.meta.disengagePredictions.gasPressProbs[1]
    else:
      throttle_prob = 1.0
    return x, v, a, j, throttle_prob

  def _mpc_radar_state(self, sm):
    """MPC 前车参数源选择：
    - 车速 < 60 km/h 且雷达点云有效（Ford + FordStockAccFusion）时，使用
      视觉车道收敛 + dRel/vLead 滤波后的雷达点云数据（与停车起步判定同源）；
    - 否则（≥60 km/h / 雷达无效 / 车道校验拒绝）使用 OP 视觉模型前车距离/速度。
    - 雷达数据无效或报错时绝不使用过期/不可信雷达 lead。"""
    rs = sm['radarState']
    lead = rs.leadOne
    sm_valid = getattr(sm, 'valid', None)
    radar_ok = (sm_valid is None or sm_valid.get('radarState', False)) and not radar_state_degraded(rs)

    # 雷达点云启用带（迟滞）：<60 km/h 进入，>=62 km/h 退出
    v_ego = float(sm['carState'].vEgo)
    if v_ego < RADAR_LEAD_ENTER_V_MS:
      self._radar_lead_active = True
    elif v_ego >= RADAR_LEAD_EXIT_V_MS:
      self._radar_lead_active = False
    # 1s 刷新一次融合开关（避免每帧读 Params）
    now = time.monotonic()
    if self._is_ford and self._fusion_params is not None and now - self._fusion_checked_t > 1.0:
      self._fusion_checked_t = now
      try:
        self._fusion_enabled_cached = self._fusion_params.get_bool("FordStockAccFusion")
      except Exception:
        pass
    use_radar = self._is_ford and self._fusion_enabled_cached and self._radar_lead_active and radar_ok

    if use_radar and lead is not None and lead.status and lead.radar:
      filt = self._radar_lead_filter.update(lead, get_vision_lead(sm))
      if filt is not None and filt.status:
        self._mpc_lead_status = True
        self._mpc_lead_d_rel = float(filt.dRel)
        return _RadarStateCopy(filt, rs.leadTwo)
      # 相邻车道雷达目标/滤波无效：回退视觉 lead；无可靠视觉 lead 视为无前车
      lead_copy = lead_from_vision(get_vision_lead(sm))
      self._mpc_lead_status = lead_copy.status
      self._mpc_lead_d_rel = float(lead_copy.dRel) if lead_copy.status else float('inf')
      return _RadarStateCopy(lead_copy, rs.leadTwo if radar_ok else FilteredLead())

    # 视觉模型前车距离/速度（无可靠视觉 lead 且雷达有效时回退雷达 leadOne 原样保底）
    vlead = get_vision_lead(sm)
    if vlead is not None:
      lead_copy = lead_from_vision(vlead)
      self._mpc_lead_status = lead_copy.status
      self._mpc_lead_d_rel = float(lead_copy.dRel) if lead_copy.status else float('inf')
      return _RadarStateCopy(lead_copy, rs.leadTwo if radar_ok else FilteredLead())
    if radar_ok:
      self._mpc_lead_status = bool(lead.status) if lead is not None else False
      self._mpc_lead_d_rel = float(lead.dRel) if self._mpc_lead_status else float('inf')
      return rs
    # 视觉与雷达都不可用：明确按“无前车”处理，避免使用过期雷达数据
    self._mpc_lead_status = False
    self._mpc_lead_d_rel = float('inf')
    return _RadarStateCopy(FilteredLead(), FilteredLead())

  def _apply_e2e_deadband(self, a_target, v_ego, should_stop):
    """把 e2e 稳态巡航中的轻微负加速度提为 0，减少制动/补油抖动。

    安全门：
    - 无停车请求、无 FCW、车速 >= E2E_DEADBAND_MIN_SPEED
    - 允许给油（否则本来就在滑行）
    - 前车时距 >= E2E_DEADBAND_MIN_LEAD_GAP；更近时完全交给 MPC
    - 只处理 (-deadband, 0)，更大的减速度原样保留
    """
    lead_gap = self._mpc_lead_d_rel / max(v_ego, 1e-3)
    if (should_stop or self.fcw or v_ego < E2E_DEADBAND_MIN_SPEED or
        not self.allow_throttle or lead_gap < E2E_DEADBAND_MIN_LEAD_GAP):
      self._e2e_deadband_active = False
      return a_target

    # 进入/退出使用不同阈值，避免在边界附近来回切换
    limit = E2E_LIGHT_DECEL_DEADBAND * (1.5 if self._e2e_deadband_active else 1.0)
    if -limit < a_target < 0.0:
      self._e2e_deadband_active = True
      return 0.0

    self._e2e_deadband_active = False
    return a_target

  def update(self, sm):
    all_checks = getattr(sm, 'all_checks', None)
    self._highway_radar_fallback_active = bool(
      self._is_ford and self._fusion_params is not None and all_checks is not None and
      highway_radar_fallback_enabled(self.CP, sm['carState'].vEgo) and
      all_checks(['modelV2', 'carState', 'controlsState', 'selfdriveState'])
    )
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    force_slow_decel = sm['controlsState'].forceDecel

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    reset_state = reset_state or not v_cruise_initialized

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    accel_clip = [ACCEL_MIN, get_max_accel(v_ego)]
    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['liveParameters'].angleOffsetDeg
    accel_clip = limit_accel_in_turns(v_ego, steer_angle_without_offset, accel_clip, self.CP)

    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = np.clip(sm['carState'].aEgo, accel_clip[0], accel_clip[1])

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    _, _, _, _, throttle_prob = self.parse_model(sm['modelV2'])
    # Don't clip at low speeds since throttle_prob doesn't account for creep
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    if not self.allow_throttle:
      clipped_accel_coast = max(accel_coast, accel_clip[0])
      clipped_accel_coast_interp = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [accel_clip[1], clipped_accel_coast])
      accel_clip[1] = min(accel_clip[1], clipped_accel_coast_interp)

    # Get new v_cruise and a_desired from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.a_desired = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.a_desired, v_cruise)

    if force_slow_decel:
      v_cruise = 0.0

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    self.mpc.update(self._mpc_radar_state(sm), v_cruise, personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    self.a_desired = float(np.interp(self.dt, CONTROL_N_T_IDX, self.a_desired_trajectory))
    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.a_desired + a_prev) / 2.0

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc, output_should_stop_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                                                        action_t=action_t, vEgoStopping=self.CP.vEgoStopping)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    if self.is_e2e(sm):
      output_a_target = min(output_a_target_e2e, output_a_target_mpc)
      self.output_should_stop = output_should_stop_e2e or output_should_stop_mpc
      output_a_target = self._apply_e2e_deadband(output_a_target, v_ego, self.output_should_stop)
      if output_a_target < output_a_target_mpc:
        self.mpc.source = LongitudinalPlanSource.e2e
    else:
      output_a_target = output_a_target_mpc
      self.output_should_stop = output_should_stop_mpc

    for idx in range(2):
      accel_clip[idx] = np.clip(accel_clip[idx], self.prev_accel_clip[idx] - 0.05, self.prev_accel_clip[idx] + 0.05)
    self.output_a_target = np.clip(output_a_target, accel_clip[0], accel_clip[1])
    self.prev_accel_clip = accel_clip

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    valid_services = ['carState', 'controlsState', 'selfdriveState']
    if not self._highway_radar_fallback_active:
      valid_services.append('radarState')
    plan_send.valid = sm.all_checks(service_list=valid_services)

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = self._mpc_lead_status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
