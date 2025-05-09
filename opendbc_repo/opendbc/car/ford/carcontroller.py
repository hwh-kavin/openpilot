import math
import cereal.messaging as messaging
import numpy as np
from numpy import clip, interp
from collections import deque
from common.filter_simple import FirstOrderFilter
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_std_steer_angle_limits, structs
from opendbc.car.vehicle_model import VehicleModel
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from common.params import Params
from selfdrive.modeld.constants import ModelConstants  # for calculations
from common.pid import PIDController # PID control of lateral
from bluepilot.params.bp_params import load_custom_params, update_custom_params  # Import custom param functions
from opendbc.car.ford.helpers import compute_dm_msg_values
from bluepilot.logger.bp_logger import debug, info, warning, error, critical


LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

def index_function(idx, max_val=192, max_idx=32):
  return (max_val) * ((idx/max_idx)**2)

# ISO 11270
ISO_LATERAL_ACCEL = 3.0  # m/s^2  # TODO: import from test lateral limits file?

# Limit to average banked road since safety doesn't have the roll
EARTH_G = 9.81
AVERAGE_ROAD_ROLL = 0.06  # ~3.4 degrees, 6% superelevation
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (EARTH_G * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

# Model prediction variables
CONTROL_N = 17
IDX_N = 33
T_IDXS = [index_function(idx, max_val=10.0) for idx in range(IDX_N)]


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle=0., lat_active=True, CP=None):
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

    self.params = Params()

    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)


    # Load initial custom parameters from params.json
    load_custom_params(self, "carcontroller")

    # Initialize control variables
    self.apply_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.fcw_alert_last = False  # previous status of collision alert
    self.send_ui_last = False  # previous state of ui elements
    self.send_bars_ts_last = 0  # previous state of ui elements
    self.send_bars_last = False  # previous state of ACC Gap elements
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0
    self.accel_pitch_compensated = 0.0

    ################################## lateral control parameters ##############################################

    # Variables to initialize (these get updated every scan as part of the control code)
    self.path_lookup_time = CP.steerActuatorDelay  # how far into the future to we need to look for our path_angle signals.
    self.precision_type = 1  # precise or comfort
    self.human_turn = False  # have we detected a human override in a turn
    self.enable_AdvLatCtrl = False # Updated form UI: enable Advanced Lateral Control

    # Curvature variables
    self.curvature_lookup_time = 0.05
    self.lane_change_factor_bp = [4.4, 40.23] # what speed to adjust lane_change_factor
    self.lane_change_factor_low = 0.95 # lane_change_factor at 4.4 m/s
    self.lane_change_factor_high = 0.75 # updated from UI: lane_change_factor at 40.23 m/s
    self.pc_blend_ratio = 0.60 # Updated from UI: %-Predicted Curvature

    # Curvature rate variables
    self.curvature_rate_delta_t = 0.3  # [s] used in denominator for curvature rate calculation
    self.curvature_rate_deque = deque(maxlen=int(round(self.curvature_rate_delta_t / 0.05)))  # 0.3 seconds at 20Hz

    # path offset variables
    self.custom_path_offset = 0.0 # updated from UI: applies a custom offset to help with in-lane positioning

    # path angle variables
    self.path_angle_filter_samples = 10 # number of samples to use for the moving average filter
    self.path_angle_deque = deque(maxlen=self.path_angle_filter_samples) # deque to hold the samples
    self.path_angle_wheel_angle_conversion = (np.pi/180)/10 # degrees to radians, divide by 10 to increase the gain
    self.path_angle_k_p_bp = [8.94, 24.58]  # speed breakpoints in m/s
    self.path_angle_k_p_v = [0.5, 0.015]  # corresponding k_p values
    self.path_angle_k_i = 0.05
    self.path_angle_pid_controller = PIDController(k_p=(self.path_angle_k_p_bp, self.path_angle_k_p_v), k_i=self.path_angle_k_i, rate=20)
    self.wheel_angle_lookup_time = 0.1
    self.pswa_blend_ratio_low = 0.8
    self.pswa_blend_ratio_high = 0.3
    self.pswa_blend_ratio_bp = [8.94, 28.82] # blend ratio from 20mph to 65mph

    # max absolute values for all four signals
    self.path_angle_max = 0.25  # too much path angle is dangerious
    self.path_offset_max = 1.0  # too much path offset causes issues
    self.curvature_max = 0.02  # from dbc files
    self.curvature_rate_max = 0.001023  # from dbc files

    # values from previous frame
    self.curvature_rate_last = 0.0
    self.path_offset_last = 0.0
    self.path_angle_last = 0.0
    self.curvature_rate = 0  # initialize curvature_rate


    # Logging variables
    debug(f'Car Fingerprint (CarController): {CP.carFingerprint}', True)

    # Lane change transition tracking
    self.post_lane_change_timer = 0
    self.post_lane_change_active = False
    self.lane_change_last = False  # Track previous lane change state
    self.pre_lane_change_values = {
        'path_angle': 0.0,
        'path_offset': 0.0,
        'desired_curvature_rate': 0.0
    }

    # Maximum allowed changes per frame
    self.max_path_angle_change = 0.00125
    self.max_path_offset_change = 0.00125
    self.max_curvature_rate_change = 0.0001

    self.sm = messaging.SubMaster(['modelV2', 'liveParameters'])
    self.VM = VehicleModel(self.CP)
    self.curvature_lookup_time = 0.2

    self.model = None
    self.lp = None
    self.send_driver_monitor_can_msg = False
    self.send_lane_depart_can_msg = False
    self.send_hands_free_cluster_msg = False
    self.predictedSteeringAngleDeg_SP = 0.0

  def handle_post_lane_change_transition(self, path_angle, path_offset, desired_curvature_rate):
    """
    Manages smooth transition of control variables after lane change
    Returns: Tuple of (path_angle, path_offset, desired_curvature_rate)
    """
    # Detect lane change completion (transition from True to False)
    if self.lane_change_last and not self.lane_change:
        self.post_lane_change_active = True
        self.post_lane_change_timer = 0
        # Store current values as starting point
        self.pre_lane_change_values = {
            'path_angle': 0.0,  # Start from zero since we're coming out of lane change
            'path_offset': 0.0,
            'desired_curvature_rate': 0.0
        }

    # Update previous lane change state
    self.lane_change_last = self.lane_change

    # If we're in post-lane change state
    if self.post_lane_change_active:
        self.post_lane_change_timer += 1

        # Apply smooth transition using rate limiting
        new_path_angle = clip(
            path_angle,
            self.pre_lane_change_values['path_angle'] - self.max_path_angle_change,
            self.pre_lane_change_values['path_angle'] + self.max_path_angle_change
        )

        new_path_offset = clip(
            path_offset,
            self.pre_lane_change_values['path_offset'] - self.max_path_offset_change,
            self.pre_lane_change_values['path_offset'] + self.max_path_offset_change
        )

        new_curvature_rate = clip(
            desired_curvature_rate,
            self.pre_lane_change_values['desired_curvature_rate'] - self.max_curvature_rate_change,
            self.pre_lane_change_values['desired_curvature_rate'] + self.max_curvature_rate_change
        )

        # Update stored values
        self.pre_lane_change_values = {
            'path_angle': new_path_angle,
            'path_offset': new_path_offset,
            'desired_curvature_rate': new_curvature_rate
        }

        # Exit transition state after 40 frames
        if self.post_lane_change_timer >= 160:
            self.post_lane_change_active = False

        return (new_path_angle, new_path_offset, new_curvature_rate)

    return (path_angle, path_offset, desired_curvature_rate)

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []
    self.sm.update(0)

    if self.sm.updated['modelV2']:
      self.model = self.sm["modelV2"]

    if self.sm.updated['liveParameters']:
      self.lp = self.sm["liveParameters"]

    if self.lp is not None:
      x = max(self.lp.stiffnessFactor, 0.1)
      sr = max(self.lp.steerRatio, 0.1)
      self.VM.update_params(x, sr)

    # Trigger the update of the settings params
    # update_settings_params(self)
    update_custom_params(self, "carcontroller")

    actuators = CC.actuators
    hud_control = CC.hudControl
    main_on = CS.out.cruiseState.available
    # if self.fordVariables is None:
      # act = actuators.as_builder()
      # self.fordVariables = act.fordVariables

    # Calculate steer_alert and fcw_alert
    steer_alert = False
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    # Compute the DM message values
    tja_msg = 0
    tja_warn = 0
    if self.send_driver_monitor_can_msg:
      if self.send_hands_free_cluster_msg:
        # print(f'HudControl: {hud_control}')
        # print(f'tja_msg: {tja_msg} | tja_warn: {tja_warn}')
        tja_msg, tja_warn = compute_dm_msg_values(hud_control, self.send_hands_free_cluster_msg)
    else:
      steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)

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

    apply_curvature = 0.0 # initialize apply_curvature
    desired_curvature_rate = 0.0 # initialize desired_curvature_rate
    path_offset = 0.0 # initialize path_offset
    path_angle = 0.0 # initialize path_angle
    reset_steering = 0 # initialize reset_steering
    ramp_type = 2 # initialize ramp_type

    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        self.precision_type = 1
        steeringPressed = CS.out.steeringPressed
        steeringAngleDeg_PV = CS.out.steeringAngleDeg

        # calculate current curvature and model desired curvature
        current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)  # use canbus data to calculate current_curvature
        desired_curvature = actuators.curvature  # get desired curvature from model

        if self.enable_AdvLatCtrl:
          # extract predicted curvature from modelV2
          if self.model is not None and len(self.model.orientation.x) >= 17:
            # compute curvature from model predicted orientationRate, and blend with desired curvature based on max predicted curvature magnitude
            curvatures = np.array(self.model.orientationRate.z) / max(0.01, CS.out.vEgoRaw)
            predicted_steering_angle_curvature = interp(self.wheel_angle_lookup_time, ModelConstants.T_IDXS, curvatures)
            predicted_curvature = interp(self.path_lookup_time, ModelConstants.T_IDXS, curvatures)
            max_abs_predicted_curvature = max(np.abs(curvatures[:17]))  # max curvature magnitude over next 2.5s
          else:
            predicted_curvature = 0.0

          # calculate predicted steering angle
          self.predictedSteeringAngleDeg_SP = math.degrees(self.VM.get_steer_from_curvature(-predicted_steering_angle_curvature, CS.out.vEgoRaw, self.lp.roll))
          self.predictedSteeringAngleDeg_SP += self.lp.angleOffsetDeg

        else:
          predicted_curvature = 0.0
          predicted_steering_angle_curvature = 0.0
          self.pc_blend_ratio = 0.0
          max_abs_predicted_curvature = 0.0
          self.predictedSteeringAngleDeg_SP = 0.0
          self.pswa_blend_ratio = 0.0

        # determine requested curvature
        if self.enable_AdvLatCtrl:
          # equate apply_curvature to a blend of desired and predicted_curvature and apply curvature limits
          requested_curvature = (predicted_curvature * self.pc_blend_ratio) + (desired_curvature * (1 - self.pc_blend_ratio))
        else:
          requested_curvature = desired_curvature

        # determine if a lane change is active
        if (self.model.meta.laneChangeState == 1 or self.model.meta.laneChangeState == 2 or self.model.meta.laneChangeState == 3):
            self.lane_change = True
        else:
            self.lane_change = False

        # determine lane_change_factor based on speed
        lane_change_factor = interp(CS.out.vEgoRaw, self.lane_change_factor_bp, [self.lane_change_factor_low, self.lane_change_factor_high])

        # if changing lanes, modify curvature to smooth out the lane change
        if self.lane_change and (self.model.meta.laneChangeDirection == 1): # if we are changing lanes to the left
          if requested_curvature < 0: # and the curvature is taking us to the left
              requested_curvature = requested_curvature * lane_change_factor # reduce the curvature to smooth out the lane change
          else:
              requested_curvature = requested_curvature # if we are moving back right to correct for over travel, do not reduce curvature

          self.precision_type = 0 # use comfort mode

        if self.lane_change and (self.model.meta.laneChangeDirection == 2): # if we are changing lanes to the right
          if requested_curvature > 0: # and the curvature is taking us to the right
              requested_curvature = requested_curvature * lane_change_factor # reduce the curvature to smooth out the lane change
          else:
              requested_curvature = requested_curvature # if we are moving back left to correct for over travel, do not reduce curvature

          self.precision_type = 0 # use comfort mode

        # Apply curvature limits
        apply_curvature = apply_ford_curvature_limits(
          requested_curvature,
          self.apply_curvature_last,
          current_curvature,
          CS.out.vEgoRaw,
          CS.out.steeringAngleDeg,
          CC.latActive,
          self.CP
        )

        # compute curvature rate
        self.curvature_rate_deque.append(apply_curvature)
        if len(self.curvature_rate_deque) > 1:
          delta_t = (
            self.curvature_rate_delta_t if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen else (len(self.curvature_rate_deque) - 1) * 0.05
          )
          desired_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / max(0.01, CS.out.vEgoRaw)
        else:
          desired_curvature_rate = 0.0

        # Determine if a human is making a turn and trap the value
        if steeringPressed and abs(steeringAngleDeg_PV) > 45:
          self.human_turn = True
        else:
          self.human_turn = False

        # get path offset from model.position.y
        path_offset = interp(self.curvature_lookup_time, ModelConstants.T_IDXS, self.model.position.y)

        # no path_offset during lane changes (it will fight you until it swaps to new lane if you don't set to zero)
        if self.lane_change:
          path_offset = 0

        # Use path_angle to help with centering vehicle in lane, derive path_angle from the models desired steering wheel position
        # path_angle is a corrective variable, so subtract out current wheel position (associated with curvature)

        # first define the desired steering wheel angle
        desired_steeringAngleDeg_SP = actuators.steeringAngleDeg

        # second calculate the blend ratio
        self.pswa_blend_ratio = interp(CS.out.vEgoRaw, self.pswa_blend_ratio_bp, [self.pswa_blend_ratio_low, self.pswa_blend_ratio_high])

        # third calcualte the reqquested steering wheel angle
        requested_steeringAngleDeg_SP = (self.predictedSteeringAngleDeg_SP * self.pswa_blend_ratio) + ((1 - self.pswa_blend_ratio) * desired_steeringAngleDeg_SP)

        # calculate how far the steering wheel is from the desired steering wheel angle
        steering_wheel_delta = (steeringAngleDeg_PV - requested_steeringAngleDeg_SP) * self.path_angle_wheel_angle_conversion

        # if a human turn is active, zero out the steering_wheel_delta
        if self.human_turn:
          steering_wheel_delta = 0.0

        # if a lane change is active, zero out the steering_wheel_delta
        if self.lane_change:
          steering_wheel_delta = 0.0

        # if we are really low speeds, no steering wheel delta because the model does stupid things
        if CS.out.vEgoRaw < 2:
          steering_wheel_delta = 0.0

        # if we are not using Advanced Lateral Control, zero out the steering_wheel_delta
        if not self.enable_AdvLatCtrl:
          steering_wheel_delta = 0.0

        # use PID to calcualte path_angle.
        path_angle_PID = self.path_angle_pid_controller.update((steering_wheel_delta), speed=CS.out.vEgoRaw)

        # output the PID into path_angle
        path_angle = path_angle_PID

        # zero path_angle during lane changes
        if self.lane_change:
          path_angle = 0.0

        # rate limit path_angle
        path_angle_roc = interp(abs(CS.out.vEgoRaw), [5, 25], [0.003, 0.002])
        path_angle = clip(path_angle, self.path_angle_last - path_angle_roc, self.path_angle_last + path_angle_roc)

        # Apply post lane change transition logic
        path_angle, path_offset, desired_curvature_rate = self.handle_post_lane_change_transition(
            path_angle, path_offset, desired_curvature_rate
        )

        # clip all values to max.
        apply_curvature = clip(apply_curvature, -self.curvature_max, self.curvature_max)
        desired_curvature_rate = clip(desired_curvature_rate, -self.curvature_rate_max, self.curvature_rate_max)
        path_offset = clip(path_offset, -self.path_offset_max, self.path_offset_max)
        path_angle = clip(path_angle, -self.path_angle_max, self.path_angle_max)

        # if we are not using Advanced Lateral Control, zero out path_angle and path_offset
        if not self.enable_AdvLatCtrl:
          path_angle = 0.0
          path_offset = 0.0
          desired_curvature_rate = 0.0

        # Determine if a human is making a turn and trap the value
        # if a human turn is active, reset steering to prevent windup
        if steeringPressed and abs(steeringAngleDeg_PV) > 45:
          self.human_turn = True
        else:
          self.human_turn = False

        # Determine when to reset steering
        if (self.human_turn) and self.enable_human_turn_detection:
          reset_steering = 1
        else:
          reset_steering = 0

        # reset steering by setting all values to 0 and ramp_type to immediate
        if reset_steering == 1:
          apply_curvature = 0
          path_offset = 0
          path_angle = 0
          desired_curvature_rate = 0
          ramp_type = 3
          self.path_angle_deque.clear()
          self.path_angle_pid_controller.reset()
        else:
          ramp_type = 2
      else:
        apply_curvature = 0.0
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_angle = 0.0
        self.path_angle_deque.clear()
        self.path_angle_pid_controller.reset()
        ramp_type = 0

      self.apply_curvature_last = apply_curvature
      self.curvature_rate_last = desired_curvature_rate
      self.path_offset_last = path_offset
      self.path_angle_last = path_angle

      # set lat_active to the value of CC.latActive
      lat_active = CC.latActive

      if self.CP.flags & FordFlags.CANFD:
        # TODO: extended mode
        # Ford uses four individual signals to dictate how to drive to the car. Curvature alone (limited to 0.02m/s^2)
        # can actuate the steering for a large portion of any lateral movements. However, in order to get further control on
        # steer actuation, the other three signals are necessary. Ford controls vehicles differently than most other makes.
        # A detailed explanation on ford control can be found here:
        # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
        mode = 1 if lat_active else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(
          self.packer, self.CAN, mode, ramp_type, self.precision_type, -path_offset, -path_angle,
          -apply_curvature, -desired_curvature_rate, counter
        ))
      else:
        # Ford non-CANFD lateral control
        can_sends.append(fordcan.create_lat_ctl_msg(
          self.packer, self.CAN, lat_active, ramp_type, self.precision_type,
          -path_offset, -path_angle, -apply_curvature, -desired_curvature_rate
        ))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      lka_hud_control = None
      if self.send_lane_depart_can_msg:
        lka_hud_control = hud_control
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN, CC.latActive, lka_hud_control))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      accel = actuators.accel
      gas = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      # Both gas and accel are in m/s^2, accel is used solely for braking
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS

      # PCM applies pitch compensation to gas/accel, but we need to compensate for the brake/pre-charge bits
      accel_due_to_pitch = 0.0
      if len(CC.orientationNED) == 3:
        accel_due_to_pitch = math.sin(CC.orientationNED[1]) * ACCELERATION_DUE_TO_GRAVITY

      accel_pitch_compensated = accel + accel_due_to_pitch
      if accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = CC.actuators.longControlState == LongCtrlState.stopping
      # TODO: look into using the actuators packet to send the desired speed
      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping, self.brake_request, v_ego_kph=V_CRUISE_MAX))

      self.accel = accel
      self.gas = gas
      self.accel_pitch_compensated = accel_pitch_compensated

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    send_bars = False
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      send_bars = True

    # Logic to keep sending the bars for 4 seconds
    if not self.send_bars_last and send_bars:
      # Save the frame # for the last flip from False to True
      self.send_bars_ts_last = self.frame
      self.distance_bar_frame = self.frame

    # keep sending the bars for 4 seconds (400 at 100Hz)
    if (self.send_bars_ts_last > 0 and (self.frame - self.send_bars_ts_last) <= 400):
      send_ui = True
      send_bars = True

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(
        fordcan.create_acc_ui_msg(
          self.packer,
          self.CAN,
          self.CP,
          main_on,
          CC.latActive,
          fcw_alert,
          CS.out.cruiseState.standstill,
          hud_control,
          CS.acc_tja_status_stock_values,
          self.send_hands_free_cluster_msg,
          send_ui,
          send_bars,
          tja_warn,
          tja_msg,
        )
      )

    self.main_on_last = main_on
    self.send_ui_last = send_ui
    self.send_bars_last = send_bars
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.fcw_alert_last = fcw_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = float(apply_curvature)
    new_actuators.accel = float(self.accel)
    new_actuators.gas = float(self.gas)
    new_actuators.steeringAngleDeg = float(self.predictedSteeringAngleDeg_SP)
    self.frame += 1
    return new_actuators, can_sends
