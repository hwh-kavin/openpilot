from cereal import car
from opendbc.can.packer import CANPacker
from openpilot.common.numpy_fast import clip
from openpilot.selfdrive.car import apply_std_steer_angle_limits
from openpilot.selfdrive.car.ford import fordcan
from openpilot.selfdrive.car.ford.values import CarControllerParams, FordFlags
from openpilot.selfdrive.car.interfaces import CarControllerBase
from openpilot.selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX

from openpilot.frogpilot.controls.lib.frogpilot_acceleration import get_max_allowed_accel

GearShifter = car.CarState.GearShifter
LongCtrlState = car.CarControl.Actuators.LongControlState
VisualAlert = car.CarControl.HUDControl.VisualAlert


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                           current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, CarControllerParams)

  return clip(apply_curvature, -CarControllerParams.CURVATURE_MAX, CarControllerParams.CURVATURE_MAX)

def calculate_blend_ratio(steering_angle, steering_rate):
  # 角度影响因子 (0-1)
  angle_factor = min(1.0, steering_angle / 30.0)  # 30度时达到最大影响
    
  # 转向速度影响因子 (0-1)
  rate_factor = min(1.0, steering_rate / 150.0)  # 150度/秒时达到最大影响
    
  # 动态权重计算（角度占60%，速度占40%）
  dynamic_ratio = 0.6 * angle_factor + 0.4 * rate_factor
    
  # 限制在0.1-0.9范围内确保系统稳定性
  return max(0.1, min(0.9, dynamic_ratio))
  
class CarController(CarControllerBase):
  def __init__(self, dbc_name, CP, FPCP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.CAN = fordcan.CanBus(CP)
    self.frame = 0

    self.apply_curvature_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.human_turn = 0

  def update(self, CC, CS, now_nanos, frogpilot_toggles):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        #计算当前实际曲率
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

        # Handle steering wheel takeover logic
        if CS.out.steeringPressed:  # Driver takeover
          steering_angle = abs(CS.out.steeringAngleDeg)   
          steering_rate = abs(CS.out.steeringRateDeg)     
    
          if steering_angle > 40 :
            self.apply_curvature_last = 0
            self.human_turn = 2
          elif steering_angle > 10 and self.human_turn :
            blend_ratio = calculate_blend_ratio(steering_angle, steering_rate)
            self.apply_curvature_last = actuators.curvature*(1-blend_ratio) + current_curvature*blend_ratio
            self.human_turn = 1

          # steering_Rate = abs(CS.out.yawRate)
          # if steering_Rate > 45:  # Driver takeover - set curvature to 0
            # self.apply_curvature_last = actuators.curvature
            # self.human_turn = 1
          # elif steering_Rate > 10 and self.human_turn:
            # self.apply_curvature_last = actuators.curvature*0.3 + current_curvature*0.7

          else:
            self.human_turn = 0
        else:
          self.human_turn = 0

        # apply rate limits, curvature error limit, and clip to signal range
        # current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)
        # PFEIFER - FSH {{
        # Ignore limits while overriding, this prevents pull when releasing the wheel. This will cause messages to be
        # blocked by panda safety, usually while the driver is overriding and limited to at most 1 message while the
        # driver is not overriding.
        # if CS.out.steeringPressed:
          # self.apply_curvature_last = actuators.curvature
        # }} PFEIFER - FSH
        apply_curvature = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature, CS.out.vEgoRaw)
      else:
        apply_curvature = 0.

      self.apply_curvature_last = apply_curvature

      if self.CP.flags & FordFlags.CANFD:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -apply_curvature, 0., counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -apply_curvature, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      # Both gas and accel are in m/s^2, accel is used solely for braking
      if frogpilot_toggles.sport_plus and (CS.out.gearShifter == GearShifter.sport or not frogpilot_toggles.map_acceleration):
        accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, min(frogpilot_toggles.max_desired_acceleration, get_max_allowed_accel(CS.out.vEgo)))
      else:
        accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, min(frogpilot_toggles.max_desired_acceleration, CarControllerParams.ACCEL_MAX))
      gas = accel
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS
      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, v_ego_kph=V_CRUISE_MAX))

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, hud_control,
                                                 CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last

    self.frame += 1
    return new_actuators, can_sends
