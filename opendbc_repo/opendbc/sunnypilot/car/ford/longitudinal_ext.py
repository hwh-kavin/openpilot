"""
BluePilot Ford longitudinal follow control extension.

Implements smoother highway following by classifying lead vehicle behavior
(gaining, pacing, trailing) and applying gas/accel limits per state. Also
adds split brake/precharge hysteresis for smoother deceleration.

Key features:
  - Speed deadband: BP long engages above 50 mph, disengages below 45 mph
  - Lead classification: gaining (closing in), pacing (matching), trailing (falling behind)
  - Gas limits per state: zero gas when gaining within 1.5s, capped gas when pacing
  - Rate-limited accel changes to avoid stomping the brakes
  - TTC-based emergency bypass for imminent collision scenarios
  - Mutual exclusion: brake_actuate forces gas to INACTIVE_GAS
"""

from collections import namedtuple

import time

import numpy as np
from numpy import clip

from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.ford.values import CarControllerParams
from opendbc.car.interfaces import V_CRUISE_MAX
from openpilot.common.error_log import append_error_log


# Result namedtuple returned by LongitudinalExt.update()
LongitudinalResult = namedtuple('LongitudinalResult', [
  'accel',
  'gas',
  'brake_actuate',
  'precharge_actuate',
  'accel_pred_send',
  'stopping',
  'target_speed',
  'bp_long_used',
])


class LongitudinalExt:
  """
  BluePilot longitudinal follow control extension for Ford vehicles.

  Mixed into CarController via multiple inheritance. The stock carcontroller
  computes op_accel/op_gas using upstream logic, then calls
  LongitudinalExt.update() to apply BP follow control on top.

  The SubMaster (for radarState) is owned by LateralCurvExt and shared via self.sm
  since both classes are mixed into the same CarController instance.
  """

  def __init__(self, CP, CP_SP):
    # BP longitudinal state
    self._bp_long_active_last = False
    self.bp_gas_last = 0.0
    self.bp_accel_last = 0.0
    self.bpSpeedAllow = False

    # Thresholds
    self.MAX_URBAN_SPEED_MPH = 45.0
    self.following_accel_ROC = 0.002  # max accel change per scan in following mode

    # Stop-and-go longitudinal handover (hysteresis):
    #   vEgo <= 15 km/h -> stock (PCM) longitudinal
    #   vEgo >= 20 km/h -> OP longitudinal
    #   between -> keep the currently active controller (avoid repeated handover jerk)
    self.STOCK_LONG_MAX_V_MS = 15.0 * CV.KPH_TO_MS   # ~4.17 m/s
    self.OP_LONG_MIN_V_MS = 20.0 * CV.KPH_TO_MS      # ~5.56 m/s
    # Ford stock ACC cannot initially enable below ~20 mph; only trust the camera-bus
    # stock request once the cruise session has been established above this speed.
    self.STOCK_SESSION_MIN_V_MS = 20.0 * CV.MPH_TO_MS
    # Min stock accel request that counts as a pullaway (filters resume noise)
    self.STOCK_PULLAWAY_THRESH = 0.12  # m/s^2
    # Debounce both go sources so a 1-2 frame shouldStop flip or a stock AccPrpl
    # spike cannot bounce between launch and braking at the standstill.
    self.STOCK_GO_DEBOUNCE_CYCLES = 8   # ~0.16s at 50Hz
    self.OP_GO_DEBOUNCE_CYCLES = 4      # ~0.08s at 50Hz
    # sp-master260727 stop-and-go tuning
    self.FUSION_STOP_GO_RELEASE_V = 3.0           # m/s, above this stop-go latch releases
    self.FUSION_OP_PULLAWAY_ACCEL = 0.4           # m/s^2 floor while OP pulls away
    self.FUSION_ACCEL_SOFT_MAX = 1.2              # m/s^2 cap for stock/OP positive accel
    self.FUSION_LEAD_MOVING_V_TRG_MARGIN_KPH = 5.0  # stock v_trg below cruise by this => lead moving
    self._stock_long_active = False
    self._stock_session_latched = False
    self._fusion_enabled = False
    self._fusion_stop_go = False
    self._stock_go_confirm = 0
    self._op_go_confirm = 0
    self._radar_lead_min_dRel = -1.0
    self.induce_stock_resume = False
    self._sng_last_log_line = ""
    self._sng_log_time = 0.0

    # Brake hysteresis thresholds
    self.brake_actuate_target = -0.14   # engage brakes below this accel
    self.brake_actuate_release = -0.06  # release brakes above this accel
    self.precharge_actuate_target = -0.12
    self.precharge_actuate_release = -0.06
    self.op_brake_actuate_last = False

    # Stock-mode brake request thresholds (narrow hysteresis, near-linear follow)
    self.STOCK_BRAKE_ACTUATE_ACCEL = -0.03  # AccBrkDecel_B_Rq on below this
    self.STOCK_BRAKE_RELEASE_ACCEL = 0.0    # AccBrkDecel_B_Rq off above this
    self.STOCK_PRE_ACTUATE_ACCEL = -0.06    # AccBrkPrchg_B_Rq on below this

    # Toggles (updated from Params each frame)
    self.disable_BP_long_UI = False
    self.disable_downhill_comp_UI = True

  def update_long_params(self, params):
    """Read longitudinal-related Params from the UI. Called each frame."""
    self.disable_BP_long_UI = params.get_bool("disable_BP_long_UI")
    self.disable_downhill_comp_UI = params.get_bool("disable_downhill_comp_UI")
    # Stock-ACC + OP fusion only runs when explicitly enabled (sp-master260727
    # behavior); otherwise plain OP longitudinal below 15 km/h.
    self._fusion_enabled = params.get_bool("FordStockAccFusion")

  def _parse_stock_accel(self, CS) -> float | None:
    """Camera-bus stock ACC accel request (m/s^2), or None when the signals look inactive."""
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

  def _stock_lead_moving(self, CS) -> bool:
    """True when the stock ACC target speed dropped below cruise (lead actually moving).

    sp-master260727: the stock system's own follow target dropping is a reliable
    'lead is moving' signal even when OP vision is still waking up.
    """
    cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
    stock_v_trg = float(getattr(CS, "stock_acc_v_trg", 0.0))
    if stock_v_trg <= 1.0:
      return False
    return stock_v_trg < (cruise_kph - self.FUSION_LEAD_MOVING_V_TRG_MARGIN_KPH)

  def _radar_lead_departing(self, CS, at_stop: bool) -> bool:
    """True when the native radar lead is departing while the ego is stopped.

    Tracks the minimum radar lead distance during the stop. A lead that moves
    more than 1.0 m further away, or drives away at vRel > 0.5 m/s, counts as
    departing. Feeds the pullaway context so an engaged longitudinal launches
    even when OP vision has not noticed the departure yet (user spec).
    """
    try:
      sm = getattr(self, 'sm', None)
      if sm is None or not sm.valid.get('radarState', False):
        return False
      lead = sm['radarState'].leadOne
      has_lead = lead is not None and getattr(lead, 'status', 0) == 1
      if not at_stop or not has_lead:
        self._radar_lead_min_dRel = -1.0
        return False
      d_rel = float(getattr(lead, 'dRel', 0.0))
      v_rel = float(getattr(lead, 'vRel', 0.0))
      if self._radar_lead_min_dRel < 0.0 or d_rel < self._radar_lead_min_dRel:
        self._radar_lead_min_dRel = d_rel
      return (d_rel - self._radar_lead_min_dRel > 1.0) or (v_rel > 0.5)
    except Exception:
      return False

  def _fuse_stock_op_accel(self, op_a: float, stock_a: float | None, *, stop_go_op: bool,
                           stock_auto_resume: bool, soft_max_accel: float) -> tuple[float, str]:
    """Fuse stock ACC with OP longitudinal (sp-master260727 logic, stock mode).

    - stock_a None: OP only; during stop-go OP gets the pullaway floor (op_go).
    - Stock auto-resume (debounced stock pullaway): follow the stock request.
    - Stop-go: if OP wants to go while stock is silent/holding, use the OP
      pullaway floor so the stock stop-hold cannot deadlock the launch.
    - Otherwise the more conservative (lower) request wins.
    """
    op_a = float(op_a)
    if stock_a is None:
      if stop_go_op:
        return float(min(max(op_a, self.FUSION_OP_PULLAWAY_ACCEL), soft_max_accel)), "op_go"
      return op_a, "op_only"

    stock_a = float(stock_a)

    if stock_auto_resume and stock_a > self.STOCK_PULLAWAY_THRESH:
      return float(min(stock_a, soft_max_accel)), "stock_go"

    if stop_go_op and op_a > stock_a + 1e-3:
      return float(min(max(op_a, self.FUSION_OP_PULLAWAY_ACCEL), soft_max_accel)), "op_go"

    fused = min(op_a, stock_a, soft_max_accel)
    if fused < op_a - 1e-3 and fused < stock_a - 1e-3:
      mode = "soft_max"
    elif fused < stock_a - 1e-3:
      mode = "op_more_brake"
    elif fused < op_a - 1e-3:
      mode = "stock_more_brake"
    else:
      mode = "match"
    return float(fused), mode

  def _radar_brake_accel(self, CS) -> float | None:
    """毫米波雷达纵向距离刹车兜底（低速原车纵向跟车时 OP 视觉距离偏差大）。

    以毫米波雷达测量为准；雷达失效时回退 OP 视觉前车距离（modelV2 leadsV3）。
    用 dRel/vRel 判距：距离不足或接近太快时返回负加速度覆盖融合输出；
    None 表示无需干预。无前车（雷达无效且视觉无 lead）直接跳过。
    """
    try:
      sm = getattr(self, 'sm', None)
      if sm is None:
        return None
      d_rel = None
      v_rel = None

      # 1) 毫米波雷达优先
      if sm.valid.get('radarState', False):
        lead = sm['radarState'].leadOne
        if lead is not None and getattr(lead, 'status', 0) == 1:
          d_rel = float(getattr(lead, 'dRel', 0.0))
          v_rel = float(getattr(lead, 'vRel', 0.0))

      # 2) 雷达失效 → OP 视觉前车距离（modelV2 leadsV3）
      if d_rel is None and sm.valid.get('modelV2', False):
        leads = getattr(sm['modelV2'], 'leadsV3', [])
        if len(leads) > 0:
          lead = leads[0]
          prob = float(getattr(lead, 'prob', 0.0))
          x = float(getattr(lead, 'x', [0.0])[0])
          if prob >= 0.5 and x > 0.5:
            d_rel = x
            v_lead = float(getattr(lead, 'v', [0.0])[0])
            v_rel = v_lead - float(CS.out.vEgo)

      if d_rel is None or d_rel <= 0.0:
        return None
      v_ego = max(float(CS.out.vEgo), 0.1)

      # 期望跟车距离：~1.8s 时距，最短 2.0m（低速跟车）
      desired_d = max(v_ego * 1.8, 2.0)
      d_err = d_rel - desired_d
      if d_err >= 0.0:
        return None

      # 距离缺口刹车：每缺 1m 减 0.8 m/s^2；接近速率惩罚
      accel = d_err * 0.8
      if v_rel < 0.0:
        accel += v_rel * 1.5
        # 碰撞时距硬阈值：TTC 过小直接按 TTC 给制动力
        ttc = d_rel / -v_rel
        if ttc < 2.0:
          accel = min(accel, float(np.interp(ttc, [0.5, 2.0], [-2.5, -0.8])))
      return float(np.clip(accel, -2.5, 0.0))
    except Exception:
      return None

  def _log_sng(self, tag: str, v_ego: float, stock_a, op_accel: float, stopping: bool,
               resume: bool, at_standstill: bool) -> None:
    """1 Hz stop-and-go diagnostics into Developer → Error Log (UiAlertLogEnable gated)."""
    try:
      line = ("SNG %s vEgo=%.2f stock_a=%s op=%.2f stopping=%s resume=%s standstill=%s "
              "stockGo=%d/%d opGo=%d/%d" % (
                tag, v_ego * CV.MS_TO_KPH,
                ("%.2f" % stock_a) if stock_a is not None else "None",
                op_accel, stopping, resume, at_standstill,
                self._stock_go_confirm, self.STOCK_GO_DEBOUNCE_CYCLES,
                self._op_go_confirm, self.OP_GO_DEBOUNCE_CYCLES))
      now = time.monotonic()
      if line != self._sng_last_log_line and now - self._sng_log_time >= 1.0:
        self._sng_log_time = now
        self._sng_last_log_line = line
        append_error_log(line)
    except Exception:
      pass

  def update(self, CC, CS, op_accel, op_gas, accel_due_to_pitch, v_ego_mph, stopping, target_speed):
    """
    Apply BluePilot longitudinal follow control on top of stock op_accel/op_gas.

    Called at 50Hz from CarController.update() inside the ACC_CONTROL_STEP block,
    after stock creep compensation and rate limiting have been applied.

    Args:
      CC: CarControl with longActive
      CS: CarState with vEgo, gasPressed, brakePressed
      op_accel: Stock openpilot accel after creep comp + rate limit (m/s^2)
      op_gas: Stock openpilot gas value (m/s^2)
      accel_due_to_pitch: Pitch compensation value (m/s^2, may be clamped by downhill toggle)
      v_ego_mph: Current speed in mph
      stopping: True if in stopping state
      target_speed: Target cruise speed (km/h)

    Returns:
      LongitudinalResult namedtuple with final accel, gas, brake, precharge values.
    """
    # Downhill compensation disable: clamp negative pitch to 0 (already applied by caller,
    # but this is where the logic lives conceptually)

    # Op brake actuate hysteresis
    accel_pitch_compensated = op_accel + accel_due_to_pitch
    op_brake_actuate = self.op_brake_actuate_last
    if accel_pitch_compensated > self.brake_actuate_release or not CC.longActive:
      op_brake_actuate = False
    elif accel_pitch_compensated < self.brake_actuate_target:
      op_brake_actuate = True

    # --- Stock longitudinal handover (stop-and-go) ---
    # Hysteresis switch: <=15 km/h stock longitudinal, >=20 km/h OP longitudinal,
    # keep the active controller inside the band to avoid repeated handovers.
    v_ego = float(CS.out.vEgo)
    if v_ego <= self.STOCK_LONG_MAX_V_MS:
      self._stock_long_active = True
    elif v_ego >= self.OP_LONG_MIN_V_MS:
      self._stock_long_active = False
    if not self._stock_long_active:
      self._stock_go_confirm = 0
      self._op_go_confirm = 0
      self.induce_stock_resume = False

    # Latch once the stock ACC session has been established above its min speed;
    # below that the camera-bus stock request is not yet trustworthy.
    if not CC.longActive or not CS.out.cruiseState.enabled:
      self._stock_session_latched = False
    elif v_ego >= self.STOCK_SESSION_MIN_V_MS:
      self._stock_session_latched = True
    if not self._stock_long_active or not self._fusion_enabled:
      self.induce_stock_resume = False

    # Stock-ACC + OP fusion (sp-master260727 behavior): only when the user
    # enables FordStockAccFusion. It runs in the stock-long handover band
    # (<=15 km/h), NOT only after the 20mph session latch — otherwise low-speed
    # city follow never arms the launch logic and the AccStopMde hold deadlocks
    # the pullaway. The session latch only gates how much we trust the
    # camera-bus stock request below.
    if self._stock_long_active and self._fusion_enabled:
      # sp-master260727 stop-and-go latch: entering a stop latches it; it only
      # releases once the car is moving again above the release speed.
      at_stop = bool(CS.out.standstill) or bool(CS.out.cruiseState.standstill)
      if at_stop:
        self._fusion_stop_go = True
      elif v_ego >= self.FUSION_STOP_GO_RELEASE_V:
        self._fusion_stop_go = False

      # Camera-bus stock request is only trustworthy once the stock ACC session
      # has been established above ~20 mph. Below that, treat it as unavailable so
      # the OP pullaway floor (op_go) still launches the car from AccStopMde.
      stock_a = self._parse_stock_accel(CS) if self._stock_session_latched else None

      # Launch context: OP decided to go, or the stock system sees the lead moving
      # (its follow target dropped below cruise), or the native radar sees the
      # lead departing (user spec: with longitudinal enabled, a departing lead
      # must make the ego follow — even if OP vision hasn't noticed yet).
      planner_wants_go = bool(CC.cruiseControl.resume)
      op_wants_go = (not stopping and op_accel > 0.05)
      radar_lead_departing = self._radar_lead_departing(CS, at_stop)
      pullaway_ctx = (planner_wants_go or op_wants_go or self._stock_lead_moving(CS) or
                      radar_lead_departing)

      # Debounced OP launch intent: filters the 1-2 frame shouldStop flip during
      # a follow-stop, which otherwise bounces between launch and braking.
      if planner_wants_go or op_wants_go or radar_lead_departing:
        self._op_go_confirm = min(self._op_go_confirm + 1, self.OP_GO_DEBOUNCE_CYCLES + 1)
      else:
        self._op_go_confirm = 0
      op_go_confirmed = self._op_go_confirm >= self.OP_GO_DEBOUNCE_CYCLES

      # Debounced stock pullaway: once AccStopMde releases, the stock ACC resumes
      # positive requests — follow that profile for the smoothest launch.
      if pullaway_ctx and stock_a is not None and stock_a > self.STOCK_PULLAWAY_THRESH:
        self._stock_go_confirm = min(self._stock_go_confirm + 1, self.STOCK_GO_DEBOUNCE_CYCLES + 1)
      else:
        self._stock_go_confirm = 0
      stock_pullaway = self._stock_go_confirm >= self.STOCK_GO_DEBOUNCE_CYCLES

      # sp-master260727: RESUME is induced only once the stock system is itself
      # pulling away (re-engages the stock session smoothly). OP-vision go is
      # covered by CC.cruiseControl.resume, which the carcontroller sends as a
      # plain single RESUME press — no repeated pulse train (CCM button-spam
      # faults shut the ACC bus down).
      self.induce_stock_resume = bool(stock_pullaway)

      # sp-master260727 stop-go pullaway floor: OP vision decided to launch —
      # floor OP accel so the launch cannot deadlock behind the stock stop-hold.
      # Never while stock is braking.
      stop_go_op = (CC.longActive and self._fusion_stop_go and not stock_pullaway and op_go_confirmed and
                    (stock_a is None or stock_a > -0.05))
      op_for_fuse = op_accel
      # 停车起步未确认时（低速蠕行/停车），原车请求不可用也不允许 OP 正加速度漏出，
      # 过滤跟随停车时 shouldStop 抖动引起的 1-2 帧启动脉冲。
      if stock_a is None and v_ego < 1.0 and not stop_go_op:
        op_for_fuse = min(op_for_fuse, 0.0)
      if stop_go_op and op_for_fuse < self.FUSION_OP_PULLAWAY_ACCEL:
        op_for_fuse = self.FUSION_OP_PULLAWAY_ACCEL

      accel, fusion_mode = self._fuse_stock_op_accel(op_for_fuse, stock_a,
                                                     stop_go_op=stop_go_op,
                                                     stock_auto_resume=stock_pullaway,
                                                     soft_max_accel=self.FUSION_ACCEL_SOFT_MAX)

      # 毫米波雷达纵向距离刹车兜底：低速跟车时 OP 视觉前车距离偏差大，
      # 用雷达 dRel 判距，需要更强制动时覆盖融合输出。仅在 OP 纵向激活时
      # 允许——纵向未激活时 panda 安全层禁止任何制动请求，发送即整帧拦截。
      radar_accel = self._radar_brake_accel(CS)
      if CC.longActive and radar_accel is not None and radar_accel < accel:
        accel = radar_accel
        fusion_mode = "radar_brake"

      # 纵向未激活：所有纵向请求必须为不活跃哨兵（accel=0、无制动、无预充），
      # 与上游/参考分支一致。否则 panda 安全层拦截 ACCDATA → CCM 收不到帧 →
      # 锁存 CcStat_D_Actl=1（Denied），仪表报 ACC 错误且按键失效。
      if not CC.longActive:
        accel = 0.0

      pulling_away = fusion_mode in ("stock_go", "op_go")
      if pulling_away:
        stopping_out = False
      elif at_stop and not op_go_confirmed and CC.longActive:
        # 起步未确认且车仍停着：保持 AccStopStat 请求，防止 LongControl 瞬时
        # 离开 stopping 导致停车保持位被清零。仅在 OP 纵向激活时发送——
        # 纵向未激活（启动/ACC 关闭）时发 AccStopStat_B_Rq 会让 CCM 拒绝
        # （CcStat_D_Actl=1 锁存），原车 ACC 退出且按键失效，需断电重来。
        stopping_out = True
      else:
        stopping_out = stopping

      # Clear brake intent while pulling away; otherwise follow the request with a
      # narrow hysteresis so braking is near-linear instead of switching on/off.
      if pulling_away:
        brake_actuate = False
        precharge_actuate = False
      else:
        brake_actuate = self.op_brake_actuate_last
        if accel > self.STOCK_BRAKE_RELEASE_ACCEL or not CC.longActive:
          brake_actuate = False
        elif accel < self.STOCK_BRAKE_ACTUATE_ACCEL:
          brake_actuate = True
        precharge_actuate = CC.longActive and accel < self.STOCK_PRE_ACTUATE_ACCEL

      gas = accel if (pulling_away or accel >= CarControllerParams.MIN_GAS) else CarControllerParams.INACTIVE_GAS
      if brake_actuate:
        gas = CarControllerParams.INACTIVE_GAS
      # Longitudinal NOT active (startup / disengaged): the protocol requires the
      # inactive sentinel (-5.0 m/s^2 -> raw 0) in AccPrpl_A_Rq. Sending a real
      # request value (0.0 -> raw 500) with Cmbb_B_Enbl=0 is an implausible combo:
      # the panda safety blocks every ACCDATA frame (longitudinal_gas_checks), the
      # CCM then sees no ACCDATA at all and latches CcStat_D_Actl=1 (Denied).
      # Upstream sends INACTIVE_GAS here as well.
      if not CC.longActive:
        gas = CarControllerParams.INACTIVE_GAS

      # With fusion active, send the real cruise / stock target speed (helps PCM
      # shifting and cluster display; sp727 behavior). When longitudinal is NOT
      # active (startup / disengaged), send the legacy max — sending 0 (unset
      # cruise) in AccVeh_V_Trg makes the CCM deny the ACC and latch
      # CcStat_D_Actl=1, which kills the stock ACC + buttons until ignition cycle.
      if CC.longActive:
        cruise_kph = float(CS.out.cruiseState.speed) * CV.MS_TO_KPH
        stock_v_trg = float(getattr(CS, "stock_acc_v_trg", 0.0))
        target_speed = stock_v_trg if stock_v_trg > 1.0 else cruise_kph
      else:
        target_speed = V_CRUISE_MAX
      target_speed = float(np.clip(target_speed, 0.0, 255.0))

      self.bp_gas_last = gas
      self.bp_accel_last = accel
      self.op_brake_actuate_last = brake_actuate

      self._log_sng("fusion:" + fusion_mode, v_ego, stock_a, op_accel, stopping_out, planner_wants_go, at_stop)

      return LongitudinalResult(
        accel=accel,
        gas=gas,
        brake_actuate=brake_actuate,
        precharge_actuate=precharge_actuate,
        accel_pred_send=CarControllerParams.INACTIVE_GAS,
        stopping=stopping_out,
        target_speed=target_speed,
        bp_long_used=False,
      )

    # Speed deadband: engage above 50 mph, disallow below 45 mph
    bpSpeedTooSlow = v_ego_mph < self.MAX_URBAN_SPEED_MPH
    bpSpeedHighEnough = v_ego_mph > self.MAX_URBAN_SPEED_MPH + 5
    if bpSpeedHighEnough:
      self.bpSpeedAllow = True
    if bpSpeedTooSlow:
      self.bpSpeedAllow = False

    # BP longitudinal follow control
    if not self.disable_BP_long_UI:
      # Read lead vehicle data from radarState (SubMaster is on self via mixin)
      v_ego = max(CS.out.vEgo, 0.5)
      lead_time_sec = 999.0
      lead = None
      v_rel = 0.0
      v_lead = 0.0

      if self.sm.valid.get('radarState', False):
        rs = self.sm['radarState']
        lead = getattr(rs, 'leadOne', None)
        if lead is not None and getattr(lead, 'status', 0) != 1:
          lead = None
        if lead:
          d_rel = float(getattr(lead, 'dRel', 0))
          v_rel = float(getattr(lead, 'vRel', 0))
          v_lead = float(getattr(lead, 'vLead', 0))
          if d_rel > 0:
            lead_time_sec = d_rel / v_ego

      lead_time_sec = float(np.clip(lead_time_sec, 0.0, 999.0))
      v_lead_mph = v_lead * 2.23694

      # Time to collision
      ttc_sec = 120.0
      if lead:
        d_rel = float(getattr(lead, 'dRel', 0))
        v_rel = float(getattr(lead, 'vRel', 0))
        if d_rel > 0 and v_rel < 0:
          ttc_sec = d_rel / (-v_rel)
        else:
          ttc_sec = 60.0
      ttc_sec = float(np.clip(ttc_sec, 0.2, 120.0))

      # Classify lead state: gaining, pacing, or trailing
      gaining = False
      pacing = False
      trailing = False
      max_follow_gas = op_gas
      min_follow_gas = op_gas
      max_follow_accel = op_accel
      min_follow_accel = op_accel
      bp_brake_actuate = False
      bp_precharge_actuate = False

      if lead:
        if v_rel < -0.1:
          gaining = True
        elif v_rel > 0.1:
          trailing = True
        else:
          pacing = True

      # Gas/accel limits per state
      if gaining:
        if lead_time_sec < 1.5:
          max_follow_gas = 0.0  # within 1.5s and gaining — no gas
          min_follow_gas = 0.0
        else:
          max_follow_gas = op_gas
          min_follow_gas = op_gas
        max_follow_accel = op_accel
        min_follow_accel = op_accel

      if pacing:
        max_follow_gas = 0.2 + accel_due_to_pitch  # cap gas when pacing
        min_follow_gas = 0.0
        max_follow_accel = op_accel
        min_follow_accel = op_accel

      if trailing:
        max_follow_gas = op_gas
        min_follow_gas = op_gas
        max_follow_accel = op_accel
        min_follow_accel = op_accel

      if lead is None:
        max_follow_gas = op_gas
        min_follow_gas = op_gas
        max_follow_accel = 0
        min_follow_accel = 0

      # Apply BP gas and accel targets
      bp_gas = clip(op_gas, min_follow_gas, max_follow_gas)
      bp_accel = clip(op_accel, min_follow_accel, max_follow_accel)

      # Rate limit downward accel changes (dampen initial brake hit)
      # Skip rate limit if imminent collision risk
      if ttc_sec > 8.0 and lead_time_sec > 0.5:
        bp_accel = clip(bp_accel, self.bp_accel_last - self.following_accel_ROC, 999)

      # BP brake/precharge hysteresis
      if bp_accel < self.brake_actuate_target:
        bp_brake_actuate = True
      if bp_accel > self.brake_actuate_release:
        bp_brake_actuate = False
      if bp_accel < self.precharge_actuate_target:
        bp_precharge_actuate = True
      if bp_accel > self.precharge_actuate_release:
        bp_precharge_actuate = False

      # Decide whether to apply BP long
      gasPressed = CS.out.gasPressed
      brakePressed = CS.out.brakePressed
      apply_bp_long = (not self.disable_BP_long_UI and self.bpSpeedAllow and
                       not gasPressed and not brakePressed and
                       (lead is None or v_lead_mph > 40.0))

      if apply_bp_long and CC.longActive:
        accel = bp_accel
        gas = bp_gas
        brake_actuate = bp_brake_actuate
        precharge_actuate = bp_precharge_actuate
      else:
        accel = op_accel
        gas = op_gas
        brake_actuate = op_brake_actuate
        precharge_actuate = op_brake_actuate

      self.bp_gas_last = bp_gas
      self.bp_accel_last = bp_accel
      bp_long_used = apply_bp_long
    else:
      # BP long disabled — pass through stock values
      accel = op_accel
      gas = op_gas
      brake_actuate = op_brake_actuate
      precharge_actuate = op_brake_actuate
      bp_long_used = False

    # Mutual exclusion: no brake and gas at the same time
    if brake_actuate:
      gas = CarControllerParams.INACTIVE_GAS

    # Clip to ford.h ACCDATA safety limits
    accel = float(clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
    if gas != CarControllerParams.INACTIVE_GAS:
      gas = float(clip(gas, CarControllerParams.MIN_GAS, CarControllerParams.ACCEL_MAX))
    accel_pred_send = CarControllerParams.INACTIVE_GAS

    self._bp_long_active_last = bp_long_used
    self.op_brake_actuate_last = op_brake_actuate

    return LongitudinalResult(
      accel=accel,
      gas=gas,
      brake_actuate=brake_actuate,
      precharge_actuate=precharge_actuate,
      accel_pred_send=accel_pred_send,
      stopping=stopping,
      target_speed=target_speed,
      bp_long_used=bp_long_used,
    )
