import math
import time
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.lateral import AVERAGE_ROAD_ROLL, ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll, higher actual roll lowers lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

# Soften stock/OP positive accel a bit when fusion is active (m/s^2)
FUSION_ACCEL_SOFT_MAX = 1.2
# Ford stock ACC typically cannot *initially* set/enable below ~20 mph
FUSION_STOCK_MIN_V = 20.0 * CV.MPH_TO_MS  # ~8.94 m/s
# After a follow-stop, hand back from OP pullaway once moving (session already active)
FUSION_STOP_GO_RELEASE_V = 3.0  # m/s
FUSION_STOCK_GO_THRESH = 0.05  # m/s^2 — treat stock AccPrpl as "wants go"
# Floor accel when planner wants go but LongControl is still in stopping hold (-2 m/s^2)
FUSION_OP_PULLAWAY_ACCEL = 0.4  # m/s^2
PARAMS_UPDATE_FRAMES = 100  # ~1s at 100Hz
LOG_EVERY_FRAMES = 100


def _parse_stock_acc_accel(CS) -> float | None:
  """Raw stock ACC accel from camera ACCDATA, or None if signals look inactive."""
  if not getattr(CS, "stock_acc_enbl", False):
    return None

  pred = float(getattr(CS, "stock_acc_prpl_pred", CarControllerParams.INACTIVE_GAS))
  prpl = float(getattr(CS, "stock_acc_prpl", CarControllerParams.INACTIVE_GAS))
  brk = float(getattr(CS, "stock_acc_brk", 0.0))

  # AccPrpl_A_Pred is the raw request during stock operation when live
  if pred > CarControllerParams.INACTIVE_GAS + 0.05:
    return pred
  if prpl >= CarControllerParams.MIN_GAS:
    return prpl
  if brk < -0.05:
    return brk
  if prpl > CarControllerParams.INACTIVE_GAS + 0.05:
    return prpl
  return None


def get_stock_acc_accel(CS, *, session_active: bool = False, v_ego: float = 0.0) -> float | None:
  """
  Stock ACC accel for fusion.

  Min speed only gates *first enable*. Once the stock ACC session is active, requests
  remain valid down to a stop (stop-and-go). Before the session is latched, ignore stock
  below FUSION_STOCK_MIN_V so OP vision handles low-speed enable/pullaway.
  """
  if (not session_active) and v_ego < FUSION_STOCK_MIN_V:
    return None
  return _parse_stock_acc_accel(CS)


def fuse_stock_op_accel(op_a: float, stock_a: float | None, *, stop_go_op: bool = False,
                        stock_auto_resume: bool = False) -> tuple[float, str]:
  """
  Fuse stock ACC with OP (vision follow / SCC curve / planner / stop-go).

  Before stock session: below ~20 mph → stock_a None → OP only.
  After stock session: stock usable down to stop; OP still wins on earlier brake/curve.
  Stop-go pullaway: if stock AccPrpl requests go, follow stock (stock_go) / induce resume.
  Do not let OP stopping-hold brake override stock go — that deadlocks AccStopMde.
  If stock will not pull away, prefer OP vision/start (op_go).
  """
  op_a = float(op_a)
  if stock_a is None:
    if stop_go_op:
      return float(min(max(op_a, FUSION_OP_PULLAWAY_ACCEL), FUSION_ACCEL_SOFT_MAX)), "op_go"
    return op_a, "op_only"

  stock_a = float(stock_a)

  # Stop-go: stock requests pullaway — follow stock even if OP is still in stopping hold.
  if stock_auto_resume and stock_a > FUSION_STOCK_GO_THRESH:
    return float(min(stock_a, FUSION_ACCEL_SOFT_MAX)), "stock_go"

  # Stock not pulling away: do not let a stuck stock hold/zero block OP pullaway.
  # op_a may already be floored by the caller when LongControl is still stopping.
  if stop_go_op and op_a > stock_a + 1e-3:
    fused = min(max(op_a, FUSION_OP_PULLAWAY_ACCEL), FUSION_ACCEL_SOFT_MAX)
    return float(fused), "op_go"

  fused = min(op_a, stock_a, FUSION_ACCEL_SOFT_MAX)
  if fused < op_a - 1e-3 and fused < stock_a - 1e-3:
    mode = "soft_max"
  elif fused < stock_a - 1e-3:
    mode = "op_more_brake"  # OP vision/SCC more conservative
  elif fused < op_a - 1e-3:
    mode = "stock_more_brake"  # stock follow more conservative / softens OP accel
  else:
    mode = "match"
  return float(fused), mode


def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)

  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel

  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)

  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))


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
    self.anti_overshoot_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.brake_request = False
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    self._params = None
    self._fusion_enabled = False
    self._fusion_log = False
    self._last_fusion_mode = "off"
    self._fusion_stop_go = False
    # Latched once stock ACC has successfully been active above min engage speed
    self._stock_acc_session = False
    self._standstill_since: float | None = None

  def _update_fusion_params(self):
    if (self.frame % PARAMS_UPDATE_FRAMES) != 0 and self._params is not None:
      return
    try:
      if self._params is None:
        from openpilot.common.params import Params
        self._params = Params()
      self._fusion_enabled = self._params.get_bool("FordStockAccFusion")
      # Gated by Developer → 日志使能 (UiAlertLogEnable); writes to error.log
      self._fusion_log = self._params.get_bool("UiAlertLogEnable")
    except Exception:
      # Keep last known values if params unavailable
      pass

  def _log_fusion(self, msg: str):
    if not self._fusion_log:
      return
    carlog.info(msg)
    try:
      from openpilot.common.swaglog import cloudlog
      cloudlog.info(msg)
    except Exception:
      pass
    try:
      from openpilot.common.error_log import append_error_log
      append_error_log(msg, check_enable=False)
    except Exception:
      pass

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    self._update_fusion_params()

    # Standstill hold timing (logged; stop-go latch for stock pullaway / resume)
    at_stop = CS.out.standstill or CS.out.cruiseState.standstill
    if at_stop:
      if self._standstill_since is None:
        self._standstill_since = time.monotonic()
      self._fusion_stop_go = True
    elif CS.out.vEgo >= FUSION_STOP_GO_RELEASE_V:
      self._fusion_stop_go = False
      self._standstill_since = None

    standstill_hold_s = (time.monotonic() - self._standstill_since) if self._standstill_since is not None else 0.0
    stock_raw = _parse_stock_acc_accel(CS)
    stock_wants_go = stock_raw is not None and stock_raw > FUSION_STOCK_GO_THRESH
    # Resume while stop-go and stock wants go (not limited to the first ~2s window)
    induce_stock_resume = (
      self._fusion_enabled and self._stock_acc_session and self._fusion_stop_go and stock_wants_go
    )

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif (CC.cruiseControl.resume or induce_stock_resume) and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # Persistent 1Hz resume while ACC enabled and at standstill — lets stock ACC pull
    # away as soon as the lead moves, without waiting for shouldStop → False.
    # Only active when FordStockAccFusion is enabled; otherwise relies on OP vision start.
    elif self._fusion_enabled and CC.enabled and CS.out.cruiseState.standstill and (self.frame % 100) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # Bronco and some other cars consistently overshoot curv requests
      # Apply some deadzone + smoothing convergence to avoid oscillations
      if self.CP.carFingerprint in (CAR.FORD_BRONCO_SPORT_MK1, CAR.FORD_F_150_MK14):
        self.anti_overshoot_curvature_last = anti_overshoot(actuators.curvature, self.anti_overshoot_curvature_last, CS.out.vEgoRaw)
        apply_curvature = self.anti_overshoot_curvature_last
      else:
        apply_curvature = actuators.curvature

      # apply rate limits, curvature error limit, and clip to signal range
      current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

      self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature,
                                                              CS.out.vEgoRaw, 0., CC.latActive, self.CP)

      if self.CP.flags & FordFlags.CANFD:
        # TODO: extended mode
        # Ford uses four individual signals to dictate how to drive to the car. Curvature alone (limited to 0.02m/s^2)
        # can actuate the steering for a large portion of any lateral movements. However, in order to get further control on
        # steer actuation, the other three signals are necessary. Ford controls vehicles differently than most other makes.
        # A detailed explanation on ford control can be found here:
        # https://www.f150gen14.com/forum/threads/introducing-bluepilot-a-ford-specific-fork-for-comma3x-openpilot.24241/#post-457706
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 0., 0., -self.apply_curvature_last, 0., counter))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 0., 0., -self.apply_curvature_last, 0.))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      op_accel = float(actuators.accel)
      accel = op_accel
      gas = accel
      fusion_mode = "off"
      stock_a = None
      long_state = CC.actuators.longControlState

      # Latch once cruise/long has been engaged above min set speed (~20 mph).
      # After latch, stock ACC long can follow down to a stop; clear when cruise/long drops.
      if not CC.longActive or not CS.out.cruiseState.enabled:
        self._stock_acc_session = False
      elif CS.out.vEgo >= FUSION_STOCK_MIN_V:
        self._stock_acc_session = True

      # Also mark stop-go while longitudinal is in stopping state
      if long_state == LongCtrlState.stopping:
        self._fusion_stop_go = True
        if self._standstill_since is None:
          self._standstill_since = time.monotonic()

      standstill_hold_s = (time.monotonic() - self._standstill_since) if self._standstill_since is not None else 0.0

      # Stock ACC + OP fusion
      stock_pullaway = False
      if self._fusion_enabled and CC.longActive:
        below_stock_min = CS.out.vEgo < FUSION_STOCK_MIN_V
        stock_a = get_stock_acc_accel(
          CS, session_active=self._stock_acc_session, v_ego=CS.out.vEgo
        )
        stock_wants_go = stock_a is not None and stock_a > FUSION_STOCK_GO_THRESH
        # Prefer stock pullaway anytime during stop-go if stock requests go (avoids OP hold deadlock)
        stock_pullaway = self._fusion_stop_go and stock_wants_go
        # Planner cleared shouldStop → controlsd sets resume. LongControl may still output
        # stopping hold (-2) while cruiseState.standstill is latched — floor a pullaway accel.
        planner_wants_go = bool(CC.cruiseControl.resume)
        stop_go_op = (
          self._fusion_stop_go and
          (not stock_pullaway) and
          (long_state == LongCtrlState.starting or op_accel > 0.05 or planner_wants_go)
        )
        op_for_fuse = op_accel
        if stop_go_op and op_for_fuse < FUSION_OP_PULLAWAY_ACCEL:
          op_for_fuse = FUSION_OP_PULLAWAY_ACCEL
        accel, fusion_mode = fuse_stock_op_accel(
          op_for_fuse, stock_a,
          stop_go_op=stop_go_op,
          stock_auto_resume=stock_pullaway,
        )
        # Clarify log mode: OP used because session not yet latched below min speed
        if (not self._stock_acc_session) and below_stock_min and fusion_mode == "op_only":
          fusion_mode = "op_below_stock_min"
        gas = accel
      else:
        self._stock_acc_session = False
      self._last_fusion_mode = fusion_mode

      pulling_away = fusion_mode in ("stock_go", "op_go")

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        # Skip during stop-go pullaway: creep at standstill subtracts up to 0.6 m/s^2 and
        # turns mild stock_go (0.06–0.2) into braking, which deadlocks AccStopMde.
        if not pulling_away:
          accel = apply_creep_compensation(accel, CS.out.vEgo)

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))
        if pulling_away:
          # Keep gas/brake channels aligned so creep-skip cannot leave gas>0 with brake_request
          gas = accel

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
      if pulling_away or accel_pitch_compensated > 0.3 or not CC.longActive:
        self.brake_request = False
      elif accel_pitch_compensated < 0.0:
        self.brake_request = True

      stopping = long_state == LongCtrlState.stopping
      # Stock auto-resume / OP pullaway: clear stop request so PCM can move
      if pulling_away:
        stopping = False

      # With fusion: send real cruise / stock target speed (helps TCM upshift). Else keep legacy max.
      if self._fusion_enabled and CC.longActive:
        v_cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
        stock_v_trg = float(getattr(CS, "stock_acc_v_trg", 0.0))
        v_trg_kph = stock_v_trg if stock_v_trg > 1.0 else v_cruise_kph
        v_trg_kph = float(np.clip(v_trg_kph, 0.0, V_CRUISE_MAX))
      else:
        # TODO: look into using the actuators packet to send the desired speed
        v_trg_kph = V_CRUISE_MAX

      if self._fusion_log and (self.frame % LOG_EVERY_FRAMES) == 0:
        self._log_fusion(
          "FordStockAccFusion: mode=%s longActive=%s enbl=%s session=%s stop_go=%s "
          "below_min=%s hold_s=%.2f auto_resume=%s longState=%s op=%.2f stock=%s "
          "fused_gas=%.2f fused_brk=%.2f prpl=%.2f brk=%.2f pred=%.2f "
          "v_trg=%.1f v_ego=%.1f cruise=%.1f stock_min_mph=%.0f" % (
            fusion_mode,
            CC.longActive,
            getattr(CS, "stock_acc_enbl", False),
            self._stock_acc_session,
            self._fusion_stop_go,
            CS.out.vEgo < FUSION_STOCK_MIN_V,
            standstill_hold_s,
            stock_pullaway,
            str(long_state),
            op_accel,
            ("%.2f" % stock_a) if stock_a is not None else "None",
            gas,
            accel,
            getattr(CS, "stock_acc_prpl", 0.0),
            getattr(CS, "stock_acc_brk", 0.0),
            getattr(CS, "stock_acc_prpl_pred", 0.0),
            v_trg_kph,
            CS.out.vEgo * CV.MS_TO_KPH,
            CS.out.cruiseState.speed * CV.MS_TO_KPH,
            FUSION_STOCK_MIN_V * CV.MS_TO_MPH,
          )
        )

      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, accel, stopping,
                                              self.brake_request, v_ego_kph=v_trg_kph))

      self.accel = accel
      self.gas = gas

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))

    # send acc ui msg at 5Hz or if ui state changes
    if hud_control.leadDistanceBars != self.lead_distance_bars_last:
      send_ui = True
      self.distance_bar_frame = self.frame

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      show_distance_bars = self.frame - self.distance_bar_frame < 400
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                                 fcw_alert, CS.out.cruiseState.standstill, show_distance_bars,
                                                 hud_control, CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert
    self.lead_distance_bars_last = hud_control.leadDistanceBars

    new_actuators = actuators.as_builder()
    new_actuators.curvature = self.apply_curvature_last
    new_actuators.accel = self.accel
    new_actuators.gas = self.gas

    self.frame += 1
    return new_actuators, can_sends
