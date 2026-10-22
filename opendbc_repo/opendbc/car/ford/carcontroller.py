import math
import time
import numpy as np
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from openpilot.common.params import Params

# BluePilot: lateral extension imports (angle-primary lateral control)
from opendbc.sunnypilot.car.ford.lateral_base_ext import LateralBaseExt
from opendbc.sunnypilot.car.ford.lateral_angle_ext import LateralAngleExt
from opendbc.sunnypilot.car.ford import fordcan_ext
from openpilot.selfdrive.controls.lib.radar_lead_filter import RadarLeadFilter, get_vision_lead

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert

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
# 低速之字形缓刹：减速后出现低于15 km/h且短于6秒的加速、随后再次减速时，
# 在前车距离/TTC安全时限制制动jerk；近距、FCW及强制动请求始终放行。
LOW_SPEED_GENTLE_BRAKE_V_MS = 15.0 * CV.KPH_TO_MS
LOW_SPEED_GENTLE_BRAKE_DIST_M = 2.0
LOW_SPEED_GENTLE_BRAKE_EMERGENCY_ACCEL = -2.5  # m/s^2，更深的紧急制动请求不拦截
LOW_SPEED_GENTLE_BRAKE_JERK = 2.0  # m/s^3
LOW_SPEED_GENTLE_BRAKE_TTC_S = 3.0
LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_MAX_S = 6.0
LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_THRESHOLD = 0.05
_ZIGZAG_WAIT_DECEL = 0
_ZIGZAG_WAIT_ACCEL = 1
_ZIGZAG_IN_ACCEL = 2
PARAMS_UPDATE_FRAMES = 100  # ~1s at 100Hz


def update_low_speed_gentle_brake(accel: float, requested_accel: float, previous_accel: float, v_ego: float,
                                  lead_distance: float | None, lead_v_rel: float | None,
                                  active: bool, zigzag_phase: int, accel_duration: float,
                                  fcw: bool, long_active: bool,
                                  standstill: bool) -> tuple[bool, int, float, float]:
  """Detect low-speed decel-accel-decel cycles and soften the final braking phase."""
  closing_speed = max(0.0, -(lead_v_rel or 0.0))
  ttc = lead_distance / closing_speed if lead_distance is not None and closing_speed > 0.0 else float('inf')
  urgent = (
    not long_active or standstill or lead_distance is None or
    lead_distance <= LOW_SPEED_GENTLE_BRAKE_DIST_M or
    ttc <= LOW_SPEED_GENTLE_BRAKE_TTC_S or fcw or
    requested_accel <= LOW_SPEED_GENTLE_BRAKE_EMERGENCY_ACCEL
  )
  if urgent:
    return False, _ZIGZAG_WAIT_DECEL, 0.0, accel

  if not active:
    if zigzag_phase == _ZIGZAG_WAIT_DECEL:
      if requested_accel < -LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_THRESHOLD:
        zigzag_phase = _ZIGZAG_WAIT_ACCEL
    elif zigzag_phase == _ZIGZAG_WAIT_ACCEL:
      if requested_accel > LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_THRESHOLD:
        if v_ego < LOW_SPEED_GENTLE_BRAKE_V_MS:
          zigzag_phase = _ZIGZAG_IN_ACCEL
          accel_duration = DT_CTRL
        else:
          zigzag_phase = _ZIGZAG_WAIT_DECEL
    elif zigzag_phase == _ZIGZAG_IN_ACCEL:
      if requested_accel < -LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_THRESHOLD:
        if accel_duration < LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_MAX_S:
          active = True
          zigzag_phase = _ZIGZAG_WAIT_DECEL
        else:
          zigzag_phase = _ZIGZAG_WAIT_ACCEL
        accel_duration = 0.0
      elif requested_accel > LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_THRESHOLD:
        if v_ego >= LOW_SPEED_GENTLE_BRAKE_V_MS:
          zigzag_phase = _ZIGZAG_WAIT_DECEL
          accel_duration = 0.0
        else:
          accel_duration += DT_CTRL
          if accel_duration >= LOW_SPEED_GENTLE_BRAKE_ZIGZAG_ACCEL_MAX_S:
            zigzag_phase = _ZIGZAG_WAIT_DECEL
            accel_duration = 0.0
      else:
        zigzag_phase = _ZIGZAG_WAIT_DECEL
        accel_duration = 0.0

  if active:
    target_accel = accel
    max_decel_step = LOW_SPEED_GENTLE_BRAKE_JERK * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL
    accel = max(accel, previous_accel - max_decel_step)
    if accel == target_accel:
      active = False
  return active, zigzag_phase, accel_duration, accel


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


def apply_creep_compensation(accel: float, v_ego: float) -> float:
  creep_accel = np.interp(v_ego, [1., 3.], [0.6, 0.])
  creep_accel = np.interp(accel, [0., 0.2], [creep_accel, 0.])
  accel -= creep_accel
  return float(accel)


class CarController(CarControllerBase, LateralBaseExt, LateralAngleExt):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    # BluePilot: initialize lateral extension mixins (angle-primary lateral control)
    LateralBaseExt.__init__(self, CP, CP_SP)
    LateralAngleExt.__init__(self, CP, CP_SP)

    self.params = Params()

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

    self._params = None
    self._fusion_enabled = False
    self._fusion_stop_go = False
    # Latched once stock ACC has successfully been active above min engage speed
    self._stock_acc_session = False
    self._standstill_since: float | None = None
    self._stock_go_confirm = 0
    # 低速防急刹：与 MPC 前车参数注入同源的雷达点云处理（视觉车道收敛+滤波）
    self._radar_filter = RadarLeadFilter()
    self._filtered_lead_frame = -1
    self._filtered_lead_d = None
    self._filtered_lead_v_rel = None
    self._low_speed_gentle_brake_active = False
    self._low_speed_zigzag_phase = _ZIGZAG_WAIT_DECEL
    self._low_speed_zigzag_accel_duration = 0.0

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

  def _get_filtered_lead_state(self, v_ego: float) -> tuple[float | None, float | None]:
    """前车距离与相对速度（每帧缓存）：滤波雷达优先，无效时视觉兜底。

    RadarLeadFilter 是 EMA 状态机，每帧只能推进一次；按 self.frame 缓存结果，
    避免同帧多次调用重复推进滤波状态。无有效前车返回 None。
    """
    if self._filtered_lead_frame == self.frame:
      return self._filtered_lead_d, self._filtered_lead_v_rel
    self._filtered_lead_frame = self.frame
    self._filtered_lead_d = None
    self._filtered_lead_v_rel = None
    try:
      sm = getattr(self, 'sm', None)
      if sm is not None:
        if sm.valid.get('radarState', False):
          lead = sm['radarState'].leadOne
          if lead is not None and getattr(lead, 'status', 0) == 1 and getattr(lead, 'radar', True):
            filt = self._radar_filter.update(lead, get_vision_lead(sm))
            if filt is not None and filt.status:
              self._filtered_lead_d = filt.dRel
              self._filtered_lead_v_rel = filt.vRel
        if self._filtered_lead_d is None:
          vlead = get_vision_lead(sm)
          if vlead is not None:
            self._filtered_lead_d = float(vlead.x[0])
            self._filtered_lead_v_rel = float(vlead.v[0]) - v_ego
    except Exception:
      self._filtered_lead_d = None
      self._filtered_lead_v_rel = None
    return self._filtered_lead_d, self._filtered_lead_v_rel

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

  def update(self, CC, CC_SP, CS, now_nanos):
    can_sends = []

    # BluePilot: update lateral SubMaster and runtime params each frame
    LateralBaseExt.update_sm(self)
    LateralAngleExt.update_angle_params(self, self.params)

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    self._update_fusion_params()

    # Stop-go latch for stock pullaway / resume
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
    pullaway_ctx = (
      self._fusion_enabled and self._fusion_stop_go and
      _stock_pullaway_context(CC, CS, long_state, op_accel)
    )
    stock_a_fusion = get_stock_acc_accel(
      CS, session_active=self._stock_acc_session, v_ego=CS.out.vEgo,
    ) if self._fusion_enabled else None
    self._update_stock_go_confirm(stock_a_fusion, pullaway_ctx)
    stock_pullaway_ready = self._stock_pullaway_ready(stock_a_fusion, pullaway_ctx)

    # Resume only when pullaway is debounced and context-valid (planner go / lead moving / starting)
    induce_stock_resume = (
      self._fusion_enabled and self._stock_acc_session and stock_pullaway_ready
    )

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

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      # BluePilot: angle-primary lateral control (κ → path_angle). Curvature mode removed.
      lat = LateralAngleExt.update_angle_strategy(self, CC, CS, actuators, self.CP)
      self.apply_curvature_last = lat.apply_curvature
      self.lateralUncertainty = lat.lateralUncertainty

      self.angleRateLimited = getattr(self, 'bp_angle_rate_limited', False)
      self.curvatureRateLimited = getattr(self, 'bp_curvature_rate_limited', False)
      self.curvatureDeviationLimited = getattr(self, 'bp_curvature_deviation_limited', False)
      self.humanTurnLateralPaused = self.angle_human_turn_active
      self.stallBlipActive = self.angle_stall_blip_active

      lat_active = CC.latActive and not (self.angle_human_turn_active or self.angle_stall_blip_active)
      if self.CP.flags & FordFlags.CANFD:
        mode = 1 if lat_active else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan_ext.create_lat_ctl2_msg(
          self.packer, self.CAN, mode, lat.ramp_type, lat.precision_type,
          -lat.path_offset, -lat.path_angle, -lat.apply_curvature, -lat.curvature_rate, counter
        ))
      else:
        can_sends.append(fordcan_ext.create_lat_ctl_msg(
          self.packer, self.CAN, lat_active, lat.ramp_type, lat.precision_type,
          -lat.path_offset, -lat.path_angle, -lat.apply_curvature, -lat.curvature_rate
        ))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      # BluePilot: angle mode is always engaged; publish shadow curvature to ford.h.
      shadow_curvature = -self.bp_kappa_cmd
      can_sends.append(fordcan_ext.create_lka_msg(
        self.packer, self.CAN, CC.latActive, hud_control, True, shadow_curvature
      ))

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
      requested_accel = accel

      if CC.longActive:
        # Compensate for engine creep at low speed.
        # Either the ABS does not account for engine creep, or the correction is very slow
        # TODO: verify this applies to EV/hybrid
        # Skip during stop-go pullaway: creep at standstill subtracts up to 0.6 m/s^2 and
        # turns mild stock_go (0.06–0.2) into braking, which deadlocks AccStopMde.
        if not pulling_away:
          accel = apply_creep_compensation(accel, CS.out.vEgo)
        requested_accel = accel

        # The stock system has been seen rate limiting the brake accel to 5 m/s^3,
        # however even 3.5 m/s^3 causes some overshoot with a step response.
        accel = max(accel, self.accel - (3.5 * CarControllerParams.ACC_CONTROL_STEP * DT_CTRL))
        if pulling_away:
          # Keep gas/brake channels aligned so creep-skip cannot leave gas>0 with brake_request
          gas = accel

      accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      gas = float(np.clip(gas, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))

      lead_d, lead_v_rel = self._get_filtered_lead_state(CS.out.vEgo)
      (self._low_speed_gentle_brake_active, self._low_speed_zigzag_phase,
       self._low_speed_zigzag_accel_duration, accel) = update_low_speed_gentle_brake(
        accel, requested_accel, self.accel, CS.out.vEgo, lead_d, lead_v_rel,
        self._low_speed_gentle_brake_active, self._low_speed_zigzag_phase,
        self._low_speed_zigzag_accel_duration, fcw_alert, CC.longActive, CS.out.standstill,
      )

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

    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      # BluePilot: OP long drives the cluster follow-gap display directly
      # (AccTGap_D_Dsply = leadDistanceBars in create_acc_ui_msg) — no stock
      # GAP button emulation. Show the bars continuously while enabled.
      show_distance_bars = CC.enabled
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
