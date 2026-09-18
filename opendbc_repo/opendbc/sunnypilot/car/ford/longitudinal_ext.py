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

    # Radar-primary handover (hysteresis):
    #   vEgo < 60 km/h -> 雷达点云为主（视觉车道匹配+滤波，视觉兜底）
    #   vEgo >= 62 km/h -> 普通 OP 视觉纵向
    #   between -> keep the currently active controller (avoid repeated handover jerk)
    self.RADAR_LONG_ENTER_V_MS = 60.0 * CV.KPH_TO_MS  # ~16.7 m/s
    self.RADAR_LONG_EXIT_V_MS = 62.0 * CV.KPH_TO_MS   # ~17.2 m/s
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
    self._radar_long_active = False
    self._stock_session_latched = False
    self._fusion_enabled = False
    self._fusion_stop_go = False
    self._stock_go_confirm = 0
    self._op_go_confirm = 0
    self._radar_lead_min_dRel = -1.0
    # MRR 点云水平角宽，可能锁到相邻车道车辆并把它距离当成本车 lead（导致停车距离拉长）。
    # 用视觉 lead 的横向位置与雷达 leadOne.yRel 做交叉校验，EMA 收敛后超出半车道宽则拒绝。
    self._radar_lane_diff_ema = 0.0
    self.RADAR_LANE_TOLERANCE_M = 1.8  # |radar_yRel + vision_y| 超过此值 => 相邻车道
    self.RADAR_LANE_EMA = 0.4          # 横向差 EMA 的新数据权重
    # 视觉强烈制动否决：OP 视觉比雷达闭环多要求的制动量(m/s^2)超过该值
    # （红灯/切入/静止障碍等雷达跟车不会减速的场景）时以视觉为准。
    self.RADAR_VISION_VETO_MARGIN = 0.5
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

    # 刹车优先：原车正在制动（AccBrkTot_A_Rq<0）时返回制动请求，不能被推进
    # 请求（AccPrpl_A_Pred/AccPrpl_A_Rq）掩盖。否则低速跟停时读到的 stock_a
    # 是正推进值，原车自己的制动被丢掉，OP 视觉噪声才会占据停车制动。
    if brk < -0.05:
      return brk

    # AccPrpl_A_Pred is the raw request during stock operation when live
    if pred > CarControllerParams.INACTIVE_GAS + 0.05:
      return pred
    if prpl >= CarControllerParams.MIN_GAS:
      return prpl
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

  def _radar_lane_ok(self) -> bool:
    """雷达 leadOne 与视觉 lead 的车道交叉校验（收敛+滤波）。

    MRR 点云水平角很宽，会把相邻车道车辆的距离误当成本车道 lead，导致跟车
    /停车距离拉长。用视觉 lead 的横向位置 y（右正）与雷达 yRel（左正，故取负
    后等价）求横向差，EMA 收敛后超过半车道宽就判为相邻车道目标、拒绝。
    无可靠视觉 lead 可比对时保留雷达结果（不拒绝），避免误杀。
    """
    try:
      sm = getattr(self, 'sm', None)
      if sm is None or not sm.valid.get('radarState', False):
        return True
      lead = sm['radarState'].leadOne
      if lead is None or getattr(lead, 'status', 0) != 1:
        return False
      if not getattr(lead, 'radar', True):  # 该 lead 本就来自视觉，必然在车道内
        return True
      if not sm.valid.get('modelV2', False):
        return True
      leads = getattr(sm['modelV2'], 'leadsV3', [])
      if len(leads) == 0:
        return True
      vlead = leads[0]
      if float(getattr(vlead, 'prob', 0.0)) < 0.5:
        return True
      radar_y = float(getattr(lead, 'yRel', 0.0))       # 左正
      vision_y = float(getattr(vlead, 'y', [0.0])[0])   # 右正
      diff = abs(radar_y + vision_y)
      self._radar_lane_diff_ema = (1.0 - self.RADAR_LANE_EMA) * self._radar_lane_diff_ema + self.RADAR_LANE_EMA * diff
      return self._radar_lane_diff_ema < self.RADAR_LANE_TOLERANCE_M
    except Exception:
      return True

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
      # 相邻车道雷达目标不判为前车起步（避免误触发起步）
      if not self._radar_lane_ok():
        self._radar_lead_min_dRel = -1.0
        return False
      d_rel = float(getattr(lead, 'dRel', 0.0))
      v_rel = float(getattr(lead, 'vRel', 0.0))
      if self._radar_lead_min_dRel < 0.0 or d_rel < self._radar_lead_min_dRel:
        self._radar_lead_min_dRel = d_rel
      return (d_rel - self._radar_lead_min_dRel > 1.0) or (v_rel > 0.5)
    except Exception:
      return False

  def _op_pullaway_accel(self, op_a: float, stop_go_op: bool, soft_max_accel: float) -> tuple[float, str]:
    """OP 纵向为唯一主控的基准加速度（不再跟随原车请求）。

    - 起步确认（op_go）：OP 视觉起步意图去抖确认后给起步地板加速度，
      防止停车保持把起步锁死。
    - 否则直接返回 OP 视觉纵向请求；低速跟车距离闭环由 update() 中的
      雷达点云接管（op_only 兜底）。
    """
    op_a = float(op_a)
    if stop_go_op:
      return float(min(max(op_a, self.FUSION_OP_PULLAWAY_ACCEL), soft_max_accel)), "op_go"
    return op_a, "op_only"

  def _radar_follow_accel(self, CS) -> float | None:
    """<60 km/h 跟车：毫米波雷达点云距离闭环（OP 纵向优先使用雷达数据）。

    雷达距离稳定，直接用 dRel/vRel 做 P+D 距离闭环：正=跟上/收近，负=刹车。
    目标停车距离 ~3.0m，随速度按 1.5s 时距平滑拉大。车道校验由 _radar_lane_ok
    把关（视觉车道匹配+滤波）。无前车/雷达失效返回 None，回退 OP 视觉。
    """
    try:
      sm = getattr(self, 'sm', None)
      if sm is None or not sm.valid.get('radarState', False):
        return None
      lead = sm['radarState'].leadOne
      if lead is None or getattr(lead, 'status', 0) != 1:
        return None
      # 相邻车道雷达目标误判：拒绝，回退视觉/原车纵向，避免停车距离被拉长
      if not self._radar_lane_ok():
        return None
      d_rel = float(getattr(lead, 'dRel', 0.0))
      v_rel = float(getattr(lead, 'vRel', 0.0))
      if d_rel <= 0.0:
        return None
      v_ego = max(float(CS.out.vEgo), 0.0)

      # 目标距离：停车 3.0m，随速度以 1.5s 时距拉大（15km/h 时约 9.2m）
      desired_d = 3.0 + v_ego * 1.5
      d_err = d_rel - desired_d

      # P + D：距离误差 0.4，相对速度阻尼 0.6
      accel = 0.4 * d_err + 0.6 * v_rel

      # 碰撞时距硬阈值：TTC 过小直接给足制动力，防止雷达跟车收不住
      if v_rel < -0.5:
        ttc = d_rel / -v_rel
        if ttc < 2.0:
          accel = min(accel, float(np.interp(ttc, [0.5, 2.0], [-3.5, -1.5])))

      return float(np.clip(accel, -2.5, 1.2))
    except Exception:
      return None

  def _radar_brake_accel(self, CS) -> float | None:
    """毫米波雷达紧急制动兜底（低速融合主控之上的安全网）。

    以毫米波雷达测量为准；雷达失效时回退 OP 视觉前车距离（modelV2 leadsV3）。
    用 dRel/vRel 判距：距离不足或接近太快时返回负加速度覆盖原车请求；
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

  def _log_sng(self, tag: str, v_ego: float, stock_a, op_accel: float, accel: float,
               stopping: bool, resume: bool, at_standstill: bool) -> None:
    """1 Hz stop-and-go diagnostics into Developer → Error Log (UiAlertLogEnable gated)."""
    try:
      line = ("SNG %s vEgo=%.2f stock_a=%s op=%.2f a=%.2f stopping=%s resume=%s standstill=%s "
              "stockGo=%d/%d opGo=%d/%d" % (
                tag, v_ego * CV.MS_TO_KPH,
                ("%.2f" % stock_a) if stock_a is not None else "None",
                op_accel, accel, stopping, resume, at_standstill,
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

    # --- 雷达点云主控带（hysteresis）---
    # vEgo < 60 km/h：雷达点云为前车距离/速度主源（视觉车道匹配+滤波）；
    # vEgo >= 62 km/h：普通 OP 视觉纵向；带内保持当前控制器避免来回切换。
    v_ego = float(CS.out.vEgo)
    if v_ego < self.RADAR_LONG_ENTER_V_MS:
      self._radar_long_active = True
    elif v_ego >= self.RADAR_LONG_EXIT_V_MS:
      self._radar_long_active = False
    if not self._radar_long_active:
      self._stock_go_confirm = 0
      self._op_go_confirm = 0
      self.induce_stock_resume = False

    # Latch once the stock ACC session has been established above its min speed;
    # below that the camera-bus stock request is not yet trustworthy.
    if not CC.longActive or not CS.out.cruiseState.enabled:
      self._stock_session_latched = False
    elif v_ego >= self.STOCK_SESSION_MIN_V_MS:
      self._stock_session_latched = True
    if not self._radar_long_active or not self._fusion_enabled:
      self.induce_stock_resume = False

    # 雷达主控融合（需 FordStockAccFusion 开启）：在 <60 km/h 带内运行。
    # 纵向请求一律由 OP 决定：雷达点云距离闭环主控、OP 视觉兜底；原车请求只
    # 用于起步上下文（自动恢复检测）与日志。20mph 会话锁存只影响原车请求的
    # 可信度，不影响 OP 主控地位。
    if self._radar_long_active and self._fusion_enabled:
      # sp-master260727 stop-and-go latch: entering a stop latches it; it only
      # releases once the car is moving again above the release speed.
      at_stop = bool(CS.out.standstill) or bool(CS.out.cruiseState.standstill)
      if at_stop:
        self._fusion_stop_go = True
      elif v_ego >= self.FUSION_STOP_GO_RELEASE_V:
        self._fusion_stop_go = False

      # 原车请求仅用于起步上下文（自动恢复检测）与日志，不再作为跟车指令。
      # 20 mph 以下会话未建立时原车请求不可信，按不可用处理。
      stock_a = self._parse_stock_accel(CS) if self._stock_session_latched else None

      # Launch context: OP decided to go, or the stock system sees the lead moving
      # (its follow target dropped below cruise), or the native radar sees the
      # lead departing (user spec: with longitudinal enabled, a departing lead
      # must make the ego follow — even if OP vision hasn't noticed yet).
      planner_wants_go = bool(CC.cruiseControl.resume)
      op_wants_go = (not stopping and op_accel > 0.05)
      radar_lead_departing = self._radar_lead_departing(CS, at_stop)
      # RESUME 按压触发：原车自动恢复（stock_pullaway）或雷达看到前车起步
      # （radar_lead_departing，>3s 停车保持只有 RESUME 能解除）。纵向请求
      # 一律由 OP 决定，原车请求只作为起步上下文与日志。
      pullaway_ctx = (planner_wants_go or op_wants_go or self._stock_lead_moving(CS) or
                      radar_lead_departing)

      # Debounced OP launch intent: filters the 1-2 frame shouldStop flip during
      # a follow-stop, which otherwise bounces between launch and braking.
      if planner_wants_go or op_wants_go or radar_lead_departing:
        self._op_go_confirm = min(self._op_go_confirm + 1, self.OP_GO_DEBOUNCE_CYCLES + 1)
      else:
        self._op_go_confirm = 0
      op_go_confirmed = self._op_go_confirm >= self.OP_GO_DEBOUNCE_CYCLES

      # Debounced stock pullaway: AccStopMde 释放后原车自动恢复检测，
      # 用于触发 RESUME 按压与起步上下文（不再跟随其加速度）。
      if pullaway_ctx and stock_a is not None and stock_a > self.STOCK_PULLAWAY_THRESH:
        self._stock_go_confirm = min(self._stock_go_confirm + 1, self.STOCK_GO_DEBOUNCE_CYCLES + 1)
      else:
        self._stock_go_confirm = 0
      stock_pullaway = self._stock_go_confirm >= self.STOCK_GO_DEBOUNCE_CYCLES

      # sp-master260727: RESUME is induced once the stock system is itself
      # pulling away (re-engages the stock session smoothly), or the native
      # radar sees the lead departing while stopped (user spec: with
      # longitudinal enabled, a departing lead must launch immediately, without
      # waiting for OP vision/plannerd to notice — the AccStopMde hold only
      # releases on a RESUME press, and CC.cruiseControl.resume lags the radar
      # by up to a second). The carcontroller sends this as a plain single
      # RESUME press (level held), never a pulse train (CCM button-spam faults
      # shut the ACC bus down).
      self.induce_stock_resume = bool(stock_pullaway or radar_lead_departing)

      # sp-master260727 stop-go pullaway floor: OP vision decided to launch —
      # floor OP accel so the launch cannot deadlock behind the stock stop-hold.
      # Never while stock is braking.
      stop_go_op = (CC.longActive and self._fusion_stop_go and op_go_confirmed and
                    (stock_a is None or stock_a > -0.05))
      op_for_fuse = op_accel
      # 停车/规划停车且起步未确认：不允许 OP 正加速度漏出，
      # 过滤跟随停车时 shouldStop 抖动引起的 1-2 帧启动脉冲。
      if (at_stop or stopping) and not stop_go_op:
        op_for_fuse = min(op_for_fuse, 0.0)
      if stop_go_op and op_for_fuse < self.FUSION_OP_PULLAWAY_ACCEL:
        op_for_fuse = self.FUSION_OP_PULLAWAY_ACCEL

      accel, fusion_mode = self._op_pullaway_accel(op_for_fuse, stop_go_op=stop_go_op,
                                                   soft_max_accel=self.FUSION_ACCEL_SOFT_MAX)

      # <60 km/h 纵向主控 = OP：雷达点云距离闭环（稳定 ~3m 停车距离）。雷达
      # lead 无效/车道校验拒绝（相邻车道目标）时回退 OP 视觉请求（op_only）兜底。
      # 起步（op_go）不覆盖；雷达失效返回 None。
      if CC.longActive and fusion_mode != "op_go":
        radar_follow = self._radar_follow_accel(CS)
        if radar_follow is not None:
          # 停车保持/规划停车时不允许正加速度漏出，防止与 AccStopStat 冲突导致蠕动
          if at_stop or stopping:
            radar_follow = min(radar_follow, 0.0)
          # 视觉强烈制动否决：视觉比雷达闭环多要求 VETO_MARGIN 以上制动时
          # （红灯/切入/静止障碍等雷达跟车不会减速的场景）以视觉为准。
          if op_for_fuse < radar_follow - self.RADAR_VISION_VETO_MARGIN:
            fusion_mode = "op_vision_veto"
          else:
            accel = radar_follow
            fusion_mode = "radar_follow"

      # 毫米波雷达近距离紧急兜底：需要更强制动时覆盖（雷达失效时内部回退视觉）。
      # 仅在 OP 纵向激活时允许——未激活发制动会被 panda 整帧拦截。
      radar_accel = self._radar_brake_accel(CS)
      if CC.longActive and radar_accel is not None and radar_accel < accel:
        accel = radar_accel
        fusion_mode = "radar_brake"

      # 纵向未激活：所有纵向请求必须为不活跃哨兵（accel=0、无制动、无预充），
      # 与上游/参考分支一致。否则 panda 安全层拦截 ACCDATA → CCM 收不到帧 →
      # 锁存 CcStat_D_Actl=1（Denied），仪表报 ACC 错误且按键失效。
      if not CC.longActive:
        accel = 0.0

      pulling_away = fusion_mode == "op_go"
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

      self._log_sng("fusion:" + fusion_mode, v_ego, stock_a, op_accel, accel,
                    stopping_out, planner_wants_go, at_stop)

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
