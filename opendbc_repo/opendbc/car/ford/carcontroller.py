import math
import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_std_steer_angle_limits, structs
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags
from opendbc.car.interfaces import CarControllerBase, ISO_LATERAL_ACCEL, V_CRUISE_MAX

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation. higher actual roll raises lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = np.clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                              current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, steering_angle, lat_active, CarControllerParams.ANGLE_LIMITS)

  # Ford Q4/CAN FD has more torque available compared to Q3/CAN so we limit it based on lateral acceleration.
  # Safety is not aware of the road roll so we subtract a conservative amount at all times
  if CP.flags & FordFlags.CANFD:
    # Limit curvature to conservative max lateral acceleration
    curvature_accel_limit = MAX_LATERAL_ACCEL / (max(v_ego_raw, 1) ** 2)
    apply_curvature = float(np.clip(apply_curvature, -curvature_accel_limit, curvature_accel_limit))

  return apply_curvature


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)

    self.apply_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    # 如果巡航控制被取消
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    # 如果巡航控制需要恢复并且满足帧频率条件
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # 如果原车车道居中功能未关闭，则发送一个按钮按下消息来关闭它
    # 原车系统检查方向盘是否被按下，并最终断开巡航控制
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # 以20Hz的频率发送转向消息
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # 应用速率限制、曲率误差限制，并限制在信号范围内
      current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
      self.apply_curvature_last = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature,
                                                              CS.out.vEgoRaw, 0., CC.latActive, self.CP)

      if self.CP.flags & FordFlags.CANFD:
        # TODO: 扩展模式
        # Ford使用四个独立的信号来指示如何控制车辆。曲率（限制在0.02m/s^2）可以单独控制车辆的大部分横向运动。
        # 但是，为了获得对转向控制的进一步控制，其他三个信号是必要的。
        # Ford控制车辆的方式与其他大多数品牌不同。
        # 关于Ford控制的详细解释可以在这里找到：
        # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -self.apply_curvature_last, 0., counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

    # 以33Hz的频率发送lka消息
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      # 速度阈值定义 (8.33m/s=30kph, 11.11m/s=40kph)
      LOW_SPEED_THRESHOLD = 8.33
      HIGH_SPEED_THRESHOLD = 11.11

      if CC.longActive:
        # 混合控制模式：低速原车控制，高速视觉控制，中间过渡区混合
        if CS.out.vEgo < LOW_SPEED_THRESHOLD:
          # 纯原车控制模式
          use_openpilot_long = False
        elif CS.out.vEgo > HIGH_SPEED_THRESHOLD:
          # 纯视觉控制模式
          use_openpilot_long = self.CP.openpilotLongitudinalControl
        else:
          # 过渡区优化混合 (8.33-11.11m/s)
          blend_factor = min(1.0, max(0.0,
              (CS.out.vEgo - LOW_SPEED_THRESHOLD) / (HIGH_SPEED_THRESHOLD - LOW_SPEED_THRESHOLD)))

          # 增强型拥堵检测 (速度<15kph且在5秒内制动3次)
          is_low_speed = CS.out.vEgo < 4.17  # 15kph
          recent_braking = self.frame - self.last_brake_frame < 250  # 5秒(50Hz*5)
          is_traffic_jam = is_low_speed and (self.brake_request_count > 3 and recent_braking)

          # 动态混合阈值 (根据制动频率调整)
          dynamic_threshold = 0.5 - (self.brake_request_count * 0.05)  # 每多一次制动降低5%阈值
          use_openpilot_long = self.CP.openpilotLongitudinalControl and (
              blend_factor > max(0.3, dynamic_threshold) or
              is_traffic_jam
          )

          if use_openpilot_long:
            stock_accel = CS.out.aEgo  # 获取原车加速度
            blended_accel = blend_factor * accel + (1 - blend_factor) * stock_accel

            # 加速度变化率限制 (最大3.5m/s³)
            accel_delta = blended_accel - self.last_accel
            max_delta = 3.5 * DT_CTRL * CarControllerParams.ACC_CONTROL_STEP
            accel = self.last_accel + np.clip(accel_delta, -max_delta, max_delta)

            # 平滑过渡曲线 (Sigmoid函数)
            smooth_blend = 1 / (1 + np.exp(-12*(blend_factor-0.5)))
            accel = smooth_blend * accel + (1-smooth_blend) * stock_accel

        if use_openpilot_long:
        # 补偿低速时的发动机爬行。
        # 要么ABS没有考虑发动机爬行，要么校正非常慢
        # TODO: 验证这是否适用于电动汽车/混合动力汽车
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # 原车系统已经看到将制动加速度限制为5 m/s^3，
        # 但是即使3.5 m/s^3也会导致阶跃响应出现一些超调。
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # gas和accel都以m/s^2为单位，accel仅用于制动
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM对gas/accel应用俯仰补偿，但我们需要对制动/预充电位进行补偿
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: 研究使用执行器数据包来发送期望速度
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # 以1Hz的频率或当ui状态改变时发送lkas ui消息
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # 如果前车距离条与前一次不同，则设置send_ui为True
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    # 以5Hz的频率或当ui状态改变时发送acc ui消息
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars
    self.last_accel = accel  # 记录当前加速度
    if self.brake_request:
        self.brake_request_count += 1
        self.last_brake_frame = self.frame
    elif self.frame - self.last_brake_frame > 250:  # 5秒无制动则清零
        self.brake_request_count = 0

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
