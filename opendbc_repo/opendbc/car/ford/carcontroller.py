import math
import time
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.lateral import AVERAGE_ROAD_ROLL, ISO_LATERAL_ACCEL
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from opendbc.sunnypilot.car.ford import fordcan_ext
from opendbc.sunnypilot.car.ford.lateral_curv_ext import LateralCurvExt, PrimaryLateralControl
from opendbc.sunnypilot.car.ford.lateral_angle_ext import LateralAngleExt

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

# CAN FD limits:
# Limit to average banked road since safety doesn't have the roll, higher actual roll lowers lateral acceleration
MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)  # ~2.4 m/s^2

# Soften stock/OP positive accel a bit when fusion is active (m/s^2)
FUSION_ACCEL_SOFT_MAX = 1.2
# Ford stock ACC typically cannot *initially* set/enable below ~20 mph
FUSION_STOCK_MIN_V = 20.0 * CV.MPH_TO_MS  # ~8.94 m/s
# Above this speed, ignore stock braking — OP owns decel (stock false brakes off-highway)
FUSION_OP_BRAKE_ONLY_V = 40.0 * CV.KPH_TO_MS  # ~11.11 m/s
# After a follow-stop, hand back from OP pullaway once moving (session already active)
FUSION_STOP_GO_RELEASE_V = 3.0  # m/s
# Min stock accel to count as pullaway (filters resume noise below ~0.12 m/s^2)
FUSION_STOCK_PULLAWAY_THRESH = 0.12  # m/s^2
# Consecutive frames stock must request go while context allows (~0.15s at 100Hz)
FUSION_STOCK_GO_DEBOUNCE_CYCLES = 15
# stock v_trg below cruise by this margin => lead moving, not spurious resume
FUSION_LEAD_MOVING_V_TRG_MARGIN_KPH = 5.0
# Stock intent-brake at/above this magnitude + lead detected => trust stock above 40 km/h
FUSION_STOCK_HARD_BRAKE = 1.0  # m/s^2
# Floor accel when planner wants go but LongControl is still in stopping hold (-2 m/s^2)
FUSION_OP_PULLAWAY_ACCEL = 0.4  # m/s^2
# Vision-only latch (A): above the high threshold, drop all stock fusion accel and use pure OP
# vision accel; hysteresis re-enables fusion only once speed falls below the low threshold.
FUSION_VISION_ONLY_HIGH_V = 70.0 * CV.KPH_TO_MS  # enable A above 70 km/h
FUSION_VISION_ONLY_LOW_V = 65.0 * CV.KPH_TO_MS  # disable A below 65 km/h
PARAMS_UPDATE_FRAMES = 100  # ~1s at 100Hz

# Pulse stock GAP so IPMA's AccTGap_D_Dsply matches speed-based 1–4 bars
STOCK_GAP_PRESS_HOLD_S = 0.12
STOCK_GAP_RETRY_S = 0.55
STOCK_GAP_DRIVER_HOLD_S = 8.0
STOCK_GAP_MAX_PRESSES = 6
STOCK_GAP_MIN = 1
STOCK_GAP_MAX = 5


def _stock_lead_moving(CS, cruise_kph: float) -> bool:
  """True when stock ACC target speed dropped — lead actually moving, not cruise default."""
  stock_v_trg = float(getattr(CS, "stock_acc_v_trg", 0.0))
  if stock_v_trg <= 1.0:
    return False
  return stock_v_trg < (cruise_kph - FUSION_LEAD_MOVING_V_TRG_MARGIN_KPH)


def _stock_pullaway_context(CC, CS, long_state, op_accel: float) -> bool:
  """Allow stock pullaway only when OP or lead confirms go — blocks resume-button false starts."""
  if CC.cruiseControl.resume:
    return True
  if long_state == LongCtrlState.starting:
    return True
  cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
  return _stock_lead_moving(CS, cruise_kph)


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
                        stock_auto_resume: bool = False, v_ego: float = 0.0,
                        stock_lead_detected: bool = False) -> tuple[float, str]:
  """
  Fuse stock ACC with OP (vision follow / SCC curve / planner / stop-go).

  Before stock session: below ~20 mph → stock_a None → OP only.
  After stock session: stock usable down to stop; OP still wins on earlier brake/curve.
  Above FUSION_OP_BRAKE_ONLY_V (~40 km/h): stock braking is ignored — OP owns decel
  (avoids stock false brakes on non-highways); exception: hard stock brake intent while
  a lead is confirmed is kept (stock_brake_keep) so real threats are not dropped.
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

  # Above 40 km/h: discard stock brake requests entirely (OP owns longitudinal braking).
  # Do not clamp to 0 — that would incorrectly zero OP accel when stock was falsely braking.
  # Exception: if stock confirms a lead (target speed well below cruise) and wants hard
  # braking, trust it — a real threat beats the false-brake filter.
  if v_ego > FUSION_OP_BRAKE_ONLY_V and stock_a < -0.05:
    if stock_lead_detected and stock_a < -FUSION_STOCK_HARD_BRAKE:
      return float(min(op_a, stock_a)), "stock_brake_keep"
    if stop_go_op:
      return float(min(max(op_a, FUSION_OP_PULLAWAY_ACCEL), FUSION_ACCEL_SOFT_MAX)), "op_go"
    return op_a, "op_brake_only"

  # Stop-go: stock requests pullaway — follow stock even if OP is still in stopping hold.
  if stock_auto_resume and stock_a > FUSION_STOCK_PULLAWAY_THRESH:
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


def _is_curvature_unwind(apply_curvature: float, apply_curvature_last: float) -> bool:
  """True when commanded |κ| is decreasing (post-apex unwind)."""
  if apply_curvature_last * apply_curvature < 0.:
    return True
  return abs(apply_curvature) + 1e-6 < abs(apply_curvature_last)


def _apply_ford_curvature_error_clip(apply_curvature: float, current_curvature: float, unwind: bool) -> float:
  tight = CarControllerParams.CURVATURE_ERROR
  loose = CarControllerParams.CURVATURE_ERROR_UNWIND
  if not unwind:
    return float(np.clip(apply_curvature, current_curvature - tight, current_curvature + tight))

  # Allow faster unwind away from measured yaw curvature; keep the "more turn" side tight.
  if current_curvature >= 0.:
    lo = current_curvature - loose
    hi = current_curvature + tight
  else:
    lo = current_curvature - tight
    hi = current_curvature + loose
  return float(np.clip(apply_curvature, lo, hi))


def _apply_ford_curvature_rate_limits(apply_curvature: float, apply_curvature_last: float, v_ego_raw: float,
                                    steering_angle: float, lat_active: bool, unwind: bool) -> float:
  steer_up = apply_curvature_last * apply_curvature >= 0. and abs(apply_curvature) > abs(apply_curvature_last)
  if steer_up:
    rate_limits = CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_UP
  elif unwind:
    rate_limits = CarControllerParams.UNWIND_ANGLE_RATE_LIMIT_DOWN
  else:
    rate_limits = CarControllerParams.ANGLE_LIMITS.ANGLE_RATE_LIMIT_DOWN

  angle_rate_lim = np.interp(v_ego_raw, rate_limits[0], rate_limits[1])
  new_apply_curvature = np.clip(apply_curvature, apply_curvature_last - angle_rate_lim, apply_curvature_last + angle_rate_lim)

  if not lat_active:
    new_apply_curvature = steering_angle

  max_curv = CarControllerParams.ANGLE_LIMITS.STEER_ANGLE_MAX
  return float(np.clip(new_apply_curvature, -max_curv, max_curv))


def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw, steering_angle, lat_active, CP):
  unwind = _is_curvature_unwind(apply_curvature, apply_curvature_last)

  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = _apply_ford_curvature_error_clip(apply_curvature, current_curvature, unwind)

  # Curvature rate limit after driver torque limit
  apply_curvature = _apply_ford_curvature_rate_limits(apply_curvature, apply_curvature_last, v_ego_raw,
                                                      steering_angle, lat_active, unwind)

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


class CarController(CarControllerBase, LateralCurvExt, LateralAngleExt):
  def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    # BluePilot: mixed-in lateral strategies (curvature / angle modes, dispatched in update()).
    LateralCurvExt.__init__(self, CP, CP_SP)
    LateralAngleExt.__init__(self, CP, CP_SP)
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
    self._fusion_stop_go = False
    # Vision-only latch (A): >70 km/h pure OP vision accel, <65 km/h back to stock/OP fusion
    self._vision_only_high_v = False
    # Latched once stock ACC has successfully been active above min engage speed
    self._stock_acc_session = False
    self._standstill_since: float | None = None
    self._stock_go_confirm = 0
    self._stock_gap_press_until = 0.0
    self._stock_gap_retry_until = 0.0
    self._stock_gap_driver_until = 0.0
    self._stock_gap_presses = 0
    self._stock_gap_target_last = 0

  def _update_stock_go_confirm(self, stock_a: float | None, allowed: bool) -> None:
    if allowed and stock_a is not None and stock_a > FUSION_STOCK_PULLAWAY_THRESH:
      self._stock_go_confirm = min(self._stock_go_confirm + 1, FUSION_STOCK_GO_DEBOUNCE_CYCLES + 1)
    else:
      self._stock_go_confirm = 0

  def _stock_pullaway_ready(self, stock_a: float | None, allowed: bool) -> bool:
    return (
      allowed and stock_a is not None and stock_a > FUSION_STOCK_PULLAWAY_THRESH and
      self._stock_go_confirm >= FUSION_STOCK_GO_DEBOUNCE_CYCLES
    )

  def _update_stock_gap_request(self, CS, target_bars: int) -> bool:
    """Hold True while a synthetic stock GAP press should be sent.

    Reads IPMA AccTGap_D_Dsply (camera ACCDATA_3) and pulses AccButtnGapTogglePress
    until it matches the speed-based 1–4 bar target. A real driver GAP press
    pauses auto-set for a few seconds.
    """
    now = time.monotonic()
    if CS.distance_button:
      self._stock_gap_driver_until = now + STOCK_GAP_DRIVER_HOLD_S
      self._stock_gap_presses = 0
      return False
    if now < self._stock_gap_driver_until:
      return False
    if not self._fusion_enabled or self._vision_only_high_v or not CS.out.cruiseState.available:
      self._stock_gap_presses = 0
      return False

    stock = int(getattr(CS, "stock_acc_tgap", 0) or 0)
    if stock < STOCK_GAP_MIN or stock > STOCK_GAP_MAX:
      return False

    target = int(np.clip(int(target_bars), 1, 4))
    if stock == target:
      self._stock_gap_presses = 0
      self._stock_gap_target_last = target
      return False

    if target != self._stock_gap_target_last:
      self._stock_gap_presses = 0
      self._stock_gap_target_last = target

    if now < self._stock_gap_press_until:
      return True
    if self._stock_gap_presses >= STOCK_GAP_MAX_PRESSES:
      return False
    if now < self._stock_gap_retry_until:
      return False

    self._stock_gap_presses += 1
    self._stock_gap_press_until = now + STOCK_GAP_PRESS_HOLD_S
    self._stock_gap_retry_until = now + STOCK_GAP_RETRY_S
    return True

  def _update_fusion_params(self):
    if (self.frame % PARAMS_UPDATE_FRAMES) != 0 and self._params is not None:
      return
    try:
      if self._params is None:
        from openpilot.common.params import Params
        self._params = Params()
      self._fusion_enabled = self._params.get_bool("FordStockAccFusion")
    except Exception:
      # Keep last known values if params unavailable
      pass

  def _update_vision_only_latch(self, v_ego: float, allowed: bool) -> None:
    """A-latch: >70 km/h enable pure-OP-vision (stock fusion accel off); <65 km/h re-enable fusion."""
    if not allowed:
      self._vision_only_high_v = False
    elif not self._vision_only_high_v and v_ego > FUSION_VISION_ONLY_HIGH_V:
      self._vision_only_high_v = True
    elif self._vision_only_high_v and v_ego < FUSION_VISION_ONLY_LOW_V:
      self._vision_only_high_v = False

  def _update_lateral_params(self):
    """Refresh BluePilot lateral strategy params (mode selection + tuning) roughly once per second."""
    if (self.frame % PARAMS_UPDATE_FRAMES) != 0 and self._params is not None:
      return
    try:
      if self._params is None:
        from openpilot.common.params import Params
        self._params = Params()
      self.update_lateral_params(self._params)
      self.update_angle_params(self._params)
    except Exception:
      # Keep last known values if params unavailable
      pass

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    self._update_fusion_params()
    self._update_lateral_params()
    # Refresh the BluePilot lateral SubMaster (modelV2 / liveParameters / selfdriveState).
    self.update_sm()

    # Standstill hold timing (stop-go latch for stock pullaway / resume)
    at_stop = CS.out.standstill or CS.out.cruiseState.standstill
    if at_stop:
      if self._standstill_since is None:
        self._standstill_since = time.monotonic()
      self._fusion_stop_go = True
    elif CS.out.vEgo >= FUSION_STOP_GO_RELEASE_V:
      self._fusion_stop_go = False
      self._standstill_since = None

    long_state = actuators.longControlState
    op_accel = float(actuators.accel)
    self._update_vision_only_latch(CS.out.vEgo, self._fusion_enabled and CC.longActive)
    pullaway_ctx = (
      self._fusion_enabled and not self._vision_only_high_v and self._fusion_stop_go and
      _stock_pullaway_context(CC, CS, long_state, op_accel)
    )
    stock_a_fusion = (
      None if self._vision_only_high_v else
      get_stock_acc_accel(CS, session_active=self._stock_acc_session, v_ego=CS.out.vEgo)
    ) if self._fusion_enabled else None
    self._update_stock_go_confirm(stock_a_fusion, pullaway_ctx)
    stock_pullaway_ready = self._stock_pullaway_ready(stock_a_fusion, pullaway_ctx)

    # Resume only when pullaway is debounced and context-valid (planner go / lead moving / starting)
    induce_stock_resume = (
      self._fusion_enabled and not self._vision_only_high_v and
      self._stock_acc_session and stock_pullaway_ready
    )
    want_stock_gap = self._update_stock_gap_request(CS, hud_control.leadDistanceBars)

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif (CC.cruiseControl.resume or induce_stock_resume) and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))
    elif want_stock_gap and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, gap_toggle=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, gap_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # BluePilot: dispatch the angle or curvature lateral strategy (FordPrefLateralControl).
      if self.primary_lateral_control == PrimaryLateralControl.angle:
        result = LateralAngleExt.update_angle_strategy(self, CC, CS, actuators, self.CP)
      else:
        result = LateralCurvExt.update(self, CC, CS, actuators, self.apply_curvature_last, self.CP)

      self.apply_curvature_last = result.apply_curvature

      # Human-turn override / post-override stall blip force lateral inactive (mode 0, all-zero
      # signals) in angle mode -- see lateral_angle_ext.py.
      force_inactive = (self.primary_lateral_control == PrimaryLateralControl.angle and
                        (self.angle_human_turn_active or self.angle_stall_blip_active))

      lat_active = CC.latActive and not force_inactive
      if not lat_active:
        self.apply_curvature_last = 0.0
        path_offset = 0.0
        path_angle = 0.0
        curvature = 0.0
        curvature_rate = 0.0
        ramp_type = 0
        precision_type = 1
      else:
        path_offset = result.path_offset
        path_angle = result.path_angle
        curvature = result.apply_curvature
        curvature_rate = result.curvature_rate
        ramp_type = result.ramp_type
        precision_type = result.precision_type

      # Wire convention: openpilot curvature/path_angle are positive-left; the LateralMotionControl
      # DBC signals are positive-right, so negate before packing (matches the stock controller's
      # -apply_curvature_last pattern and ford.h's angle_meas / shadow_curvature convention).
      if self.CP.flags & FordFlags.CANFD:
        mode = 1 if lat_active else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan_ext.create_lat_ctl2_msg(self.packer, self.CAN, mode, ramp_type, precision_type,
                                                         path_offset, -path_angle, -curvature, -curvature_rate, counter))
      else:
        can_sends.append(fordcan_ext.create_lat_ctl_msg(self.packer, self.CAN, lat_active, ramp_type, precision_type,
                                                        path_offset, -path_angle, -curvature, -curvature_rate))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      # BluePilot: in angle mode this message carries the angle_mode_engaged corroboration bit and
      # the shadow curvature (the kappa path_angle was derived from) for ford.h's deviation check.
      angle_mode_engaged = self.primary_lateral_control == PrimaryLateralControl.angle
      shadow_curvature = -self.bp_kappa_cmd if angle_mode_engaged else 0.0
      can_sends.append(fordcan_ext.create_lka_msg(self.packer, self.CAN, CC.latActive, hud_control,
                                                  angle_mode_engaged, shadow_curvature))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      op_accel = float(actuators.accel)
      accel = op_accel
      gas = accel
      fusion_mode = "off"
      stock_a = None

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

      # Stock ACC + OP fusion
      stock_pullaway = False
      if self._fusion_enabled and CC.longActive:
        if self._vision_only_high_v:
          # A-latch: pure OP vision accel, stock fusion requests fully ignored
          accel = op_accel
          fusion_mode = "vision_only_high_v"
          gas = accel
        else:
          below_stock_min = CS.out.vEgo < FUSION_STOCK_MIN_V
          stock_a = stock_a_fusion
          # Debounced stock pullaway — avoids resume-noise stock_go ↔ op_more_brake jerk at standstill
          stock_pullaway = stock_pullaway_ready
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
          # Stock confirms a lead when its own target speed drops well below cruise.
          cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
          stock_lead_detected = _stock_lead_moving(CS, cruise_kph)
          accel, fusion_mode = fuse_stock_op_accel(
            op_for_fuse, stock_a,
            stop_go_op=stop_go_op,
            stock_auto_resume=stock_pullaway,
            v_ego=CS.out.vEgo,
            stock_lead_detected=stock_lead_detected,
          )
          # Clarify log mode: OP used because session not yet latched below min speed
          if (not self._stock_acc_session) and below_stock_min and fusion_mode == "op_only":
            fusion_mode = "op_below_stock_min"
          gas = accel
      else:
        self._stock_acc_session = False

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
      if self._fusion_enabled and CC.longActive and not self._vision_only_high_v:
        v_cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
        stock_v_trg = float(getattr(CS, "stock_acc_v_trg", 0.0))
        v_trg_kph = stock_v_trg if stock_v_trg > 1.0 else v_cruise_kph
        v_trg_kph = float(np.clip(v_trg_kph, 0.0, V_CRUISE_MAX))
      else:
        # TODO: look into using the actuators packet to send the desired speed
        v_trg_kph = V_CRUISE_MAX

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
