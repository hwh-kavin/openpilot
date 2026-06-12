import time
import numpy as np
from cereal import log
from openpilot.common.swaglog import cloudlog

# 匯入 Openpilot 原廠 MPC 相關公式
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, STOP_DISTANCE, get_safe_obstacle_distance,
  get_stopped_equivalence_factor, get_T_FOLLOW
)

# =========================================================
# 參數設定區
# =========================================================

# --- 加速意圖偵測參數 ---
TRAJECTORY_HORIZON  = 6      # 預判軌跡前 6 個點 (約 0.6 秒)
INTENT_LOOKAHEAD    = 3      # 在 6 個點中有 3 個點符合加速即判定有意圖
INTENT_V_LOW        = 0.0    # 0 km/h
INTENT_V_HIGH       = 22.22  # 80 km/h (80 / 3.6)
INTENT_FRAMES_LOW   = 1      # 低速時只需 1 幀即觸發意圖 (極靈敏)
INTENT_FRAMES_HIGH  = 20     # 高速時需連續 20 幀才鎖定意圖 (防震盪)

# --- 原有參數維持與優化 ---
SPEED_OFFSET_MIN_KPH = 1.0             
SPEED_OFFSET_MAX_FLAT_KPH = 15.0       
SPEED_OFFSET_MAX_DOWNHILL_KPH = 5.0    

PITCH_SMOOTH_ALPHA_UP = 0.30           
PITCH_SMOOTH_ALPHA_DOWN = 0.05         
PITCH_UPHILL_THRESHOLD = 0.050         
PITCH_DOWNHILL_THRESHOLD = -0.030      

SOFT_HOLD_PITCH_START = 0.050          
SOFT_HOLD_PITCH_MAX = 0.080            

TTC_BP = [10., 30.]                    
TTC_V  = [3.0, 3.0]                    
EMERGENCY_TTC = 2.0                    
EMERGENCY_RELATIVE_SPEED = 10.0        
EMERGENCY_DECEL_THRESHOLD = -1.5       

LEAD_COOLDOWN_TIME = 0.5               
SPEED_BP = [0., 10., 20., 30.]         
MIN_DIST_V = [5., 10., 15., 20.]       

SOFT_HOLD_RANGE_MIN = 0.70             
SOFT_HOLD_RANGE_MAX = 0.99             
SOFT_HOLD_TTC_THRESHOLD = 2.5          
VREL_DEBOUNCE_TIME = 0.6               

SOFT_HOLD_SPEED_BP = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
SOFT_HOLD_ACCEL_V  = [1.1,  0.90,  0.70,  0.50,  0.30,  0.10]

# =========================================================
# 防震盪參數 (V12架構保留，數值微調以提升舒適度)
# =========================================================
RATIO_ENTER_THRESHOLD = 1.08     # 距離比率進入閾值（觸發取消Soft Hold）
RATIO_EXIT_THRESHOLD = 1.05      # [修正] 距離比率退出閾值，放寬至1.05避免煞車咬死
TARGET_FACTOR_FILTER_ALPHA = 0.3 # 目標因子平滑濾波係數
SOFT_HOLD_HYSTERESIS_TIME = 1.0  # Soft Hold開關滯後時間（秒）

# =========================================================
# 邏輯模組 1：純滑行控制器
# =========================================================
class CoastingLogic:
  def __init__(self):
    self.active = False                
    self.current_max_offset = 0.0      
    self._has_lead = False             
    self._last_lead_time = 0.0         
    self._active_prev = False          

  def check_emergency(self, lead, v_ego, current_time):
    if not lead or not lead.status:
      return False

    # [修正] 還原 V11 邏輯，避免 closing_speed <= 0 造成防錯機制突兀介入
    closing_speed = max(v_ego - lead.vLead, 0.1)
    lead_ttc = lead.dRel / closing_speed

    relative_speed = v_ego - lead.vLead
    min_dist_for_speed = np.interp(v_ego, SPEED_BP, MIN_DIST_V)

    if (lead_ttc < EMERGENCY_TTC) or \
       (relative_speed > EMERGENCY_RELATIVE_SPEED) or \
       (lead.dRel < min_dist_for_speed and relative_speed > 0):
      self._last_lead_time = current_time
      return True
    return False

  def update_lead_status(self, lead, v_ego, current_time):
    if lead and lead.status:
      # [修正] 還原 V11 邏輯，維持平順的 TTC 判斷
      closing_speed = max(v_ego - lead.vLead, 0.1)
      lead_ttc = lead.dRel / closing_speed
      current_ttc_threshold = np.interp(v_ego, TTC_BP, TTC_V) 
      if lead_ttc < current_ttc_threshold:
        self._has_lead = True
        self._last_lead_time = current_time
      else:
        self._has_lead = False 
    else:
      self._has_lead = False

  def update_states(self, enabled, user_ctrl_lon, v_ego, v_cruise, current_pitch, dtsc_is_active, current_time):
    if not enabled:
      self.active = False
      return

    if current_pitch < PITCH_DOWNHILL_THRESHOLD:
        self.current_max_offset = SPEED_OFFSET_MAX_DOWNHILL_KPH
    else:
        self.current_max_offset = SPEED_OFFSET_MAX_FLAT_KPH

    upper_bound = v_cruise + (self.current_max_offset / 3.6) 
    is_in_coast_window = (v_ego >= v_cruise and v_ego < upper_bound)
    in_cooldown = (current_time - self._last_lead_time) < LEAD_COOLDOWN_TIME

    should_activate = (not dtsc_is_active and
                       current_pitch <= PITCH_UPHILL_THRESHOLD and
                       not user_ctrl_lon and     
                       not self._has_lead and    
                       not in_cooldown and       
                       is_in_coast_window)
    
    self.active = should_activate
    self._active_prev = self.active

  def process_trajectory(self, a_desired_trajectory, lead):
    traj = np.copy(a_desired_trajectory)
    if self.active:
      min_accel = np.min(traj)
      if min_accel < EMERGENCY_DECEL_THRESHOLD:
        self.active = False
      else:
        if not (lead is not None and lead.status):
          # [保留 V12 優化] Numpy 矩陣運算，搭配拋物線衰減公式消除微點頭
          mask = (traj > -0.15) & (traj < 0.0)
          traj[mask] = traj[mask] * (np.abs(traj[mask]) / 0.15)
    return traj


# =========================================================
# 邏輯模組 2：增強版柔和跟車控制器
# =========================================================
class SoftHoldLogic:
  def __init__(self):
    self._soft_hold_factor = 1.0          
    self._vrel_high_start_time = 0.0      
    self._vrel_high_active = False        
    
    # 狀態記憶
    self._last_lead_time = 0.0            
    self._last_target_factor = 1.0        
    self._last_soft_hold_accel = 0.0      

    # 加速意圖計數器與狀態
    self.accel_intent_counter = 0
    self.intent_accelerating = False
    self._accel_intent_strength = 0.0     
    
    # 防震盪與平滑狀態
    self._ratio_hysteresis_state = False  
    self._cancel_filter = 0.0             
    self._target_factor_smooth = 1.0      
    
    # 標準防震盪狀態追蹤器
    self._last_stable_cancel_state = False
    self._state_change_time = 0.0

  def process_trajectory(self, a_desired_trajectory, v_ego, lead, current_pitch, t_follow):
    should_cancel_soft_hold = False
    current_time = time.monotonic()
    
    # 軌跡與意圖分析
    recent_trajectory = a_desired_trajectory[:TRAJECTORY_HORIZON]
    has_valid_lead = lead is not None and lead.status

    # 1. 動態計算所需的偵測幀數
    v_ratio = max(0.0, min((v_ego - INTENT_V_LOW) / (INTENT_V_HIGH - INTENT_V_LOW), 1.0))
    dynamic_intent_frames = int(round(INTENT_FRAMES_LOW + v_ratio * (INTENT_FRAMES_HIGH - INTENT_FRAMES_LOW)))

    # 2. 判斷當下是否具備加速跡象
    moment_accel = sum(1 for a in recent_trajectory if a > 0.05) >= INTENT_LOOKAHEAD and (lead.vRel > 0.05 if has_valid_lead else True)

    target_factor = 1.0   
    v_ego_kph = v_ego * 3.6
    current_soft_hold_accel = np.interp(v_ego_kph, SOFT_HOLD_SPEED_BP, SOFT_HOLD_ACCEL_V)
    is_lead_braking_strict = False
    skip_state_2 = False

    # 狀態機 1：雷達防閃爍與加速意圖判斷
    if not has_valid_lead:
        self._vrel_high_active = False
        
        # 使用與增加率相同的對稱衰減，避免一幀雜訊秒殺意圖
        decrement = 1.0 / max(dynamic_intent_frames, 1)
        self._accel_intent_strength = max(0.0, self._accel_intent_strength - decrement)
        if self._accel_intent_strength < 0.1:
            self.intent_accelerating = False
            self.accel_intent_counter = 0

        if (current_time - self._last_lead_time) < 0.5:
            if self._last_soft_hold_accel >= 0.0:
                target_factor = self._last_target_factor
                current_soft_hold_accel = self._last_soft_hold_accel
            else:
                target_factor = 0.0
                current_soft_hold_accel = 0.0
            should_cancel_soft_hold = False
            skip_state_2 = True 
        else:
            should_cancel_soft_hold = True
            skip_state_2 = True
    else:
        self._last_lead_time = current_time 
        
        if moment_accel:
            increment = 1.0 / max(dynamic_intent_frames, 1)
            self._accel_intent_strength = min(1.0, self._accel_intent_strength + increment)
            self.accel_intent_counter += 1
        else:
            decrement = 1.0 / max(dynamic_intent_frames, 1)
            self._accel_intent_strength = max(0.0, self._accel_intent_strength - decrement)
            self.accel_intent_counter = 0

        # 觸發加速意圖狀態鎖定
        if self._accel_intent_strength > 0.7:  
            self.intent_accelerating = True
        elif self._accel_intent_strength < 0.3:  
            self.intent_accelerating = False

        if self.intent_accelerating:
            cancel_probability = min(1.0, self._accel_intent_strength * 1.5) 
            self._cancel_filter = 0.8 * self._cancel_filter + 0.2 * cancel_probability
            if self._cancel_filter > 0.5:
                should_cancel_soft_hold = True

        if lead.vRel > 1.0:
            if not self._vrel_high_active:
                self._vrel_high_active = True
                self._vrel_high_start_time = current_time
            elif (current_time - self._vrel_high_start_time) > VREL_DEBOUNCE_TIME:
                should_cancel_soft_hold = True
        else:
            self._vrel_high_active = False
            
        if current_pitch > SOFT_HOLD_PITCH_MAX:
            should_cancel_soft_hold = True

    # 狀態機 2：精細計算
    if not skip_state_2:
        # [修正] 絕對關鍵！還原 V11 寬鬆的前車停止判定，容許雜訊，大幅提升舒適度
        is_lead_stopped = (lead.vLead < 1.0) and (lead.vRel <= 0.3)
        
        if v_ego_kph <= 10.0:
            is_lead_braking_strict = (lead.aLeadK < -0.1 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 30.0:
            is_lead_braking_strict = (lead.aLeadK < -0.5 or is_lead_stopped) and (lead.vRel < 0.5)
        elif v_ego_kph <= 40.0:
            is_lead_braking_strict = lead.aLeadK < -1.0 or is_lead_stopped
        else: 
            is_lead_braking_strict = lead.aLeadK < -1.25 or is_lead_stopped

        closing_speed = max(v_ego - lead.vLead, 0.1)
        current_ttc = lead.dRel / closing_speed
        desired_dist = get_safe_obstacle_distance(v_ego, t_follow)
        lead_obstacle_dist = lead.dRel + get_stopped_equivalence_factor(lead.vLead)

        ratio = 10.0 if desired_dist < 0.1 else (lead_obstacle_dist / desired_dist)
        
        if not should_cancel_soft_hold:
            if ratio > RATIO_ENTER_THRESHOLD:
                self._ratio_hysteresis_state = True
            elif ratio < RATIO_EXIT_THRESHOLD:
                self._ratio_hysteresis_state = False
            
            if self._ratio_hysteresis_state:
                should_cancel_soft_hold = True

    # 使用正確且標準的防震盪(Debounce)計時器邏輯
    if should_cancel_soft_hold != self._last_stable_cancel_state:
        if self._state_change_time == 0.0:
            # 狀態剛發生改變，開始計時
            self._state_change_time = current_time
        elif (current_time - self._state_change_time) > (SOFT_HOLD_HYSTERESIS_TIME / 2):
            # 狀態維持超過半個滯後時間，正式接受改變
            self._last_stable_cancel_state = should_cancel_soft_hold
            self._state_change_time = 0.0
    else:
        # 狀態無變化，歸零計時器
        self._state_change_time = 0.0

    # 覆寫為濾波後的最終決策
    should_cancel_soft_hold = self._last_stable_cancel_state

    # === 最終結算 ===
    if should_cancel_soft_hold:
        # 【終極保證】：只要決定放手，目標因子絕對鎖死在 1.0 (100% 還原動力)
        target_factor = 1.0  
        alpha = 0.60 if self.intent_accelerating else 0.30 
        
    elif not skip_state_2: 
        distance_factor = 1.0 
        if current_pitch <= SOFT_HOLD_PITCH_MAX:
            if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and current_ttc <= SOFT_HOLD_TTC_THRESHOLD:
                distance_factor = 0.0

        v_rel_factor = np.interp(lead.vRel, [-2.0, 0.5], [0.0, 1.0])
        target_factor = max(distance_factor, v_rel_factor)

        if SOFT_HOLD_RANGE_MIN < ratio < SOFT_HOLD_RANGE_MAX and is_lead_braking_strict:
            if current_pitch > SOFT_HOLD_PITCH_START:
                smooth_factor = float(np.interp(current_pitch, [SOFT_HOLD_PITCH_START, SOFT_HOLD_PITCH_MAX], [0.0, 1.0]))
                target_factor = smooth_factor  
                current_soft_hold_accel = current_soft_hold_accel * smooth_factor 
            else:
                if is_lead_stopped:
                    current_soft_hold_accel = float(np.interp(v_ego_kph, [0.0, 150.0], [0.0, -0.30]))
                elif v_ego_kph >= 50.0:
                    if lead.vRel < -0.1 and lead.aLeadK <= -1.5:
                        dynamic_brake = lead.aLeadK * 0.30
                        current_soft_hold_accel = np.clip(dynamic_brake, -1.0, 0.0)
                    else:
                        current_soft_hold_accel = 0.0 
                else:
                    current_soft_hold_accel = 0.0
                target_factor = 0.0 

        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 
    else:
        alpha = 0.10 if target_factor > self._soft_hold_factor else 0.20 

    self._last_target_factor = target_factor
    self._last_soft_hold_accel = current_soft_hold_accel
    
    # 低通濾波
    self._soft_hold_factor = (1.0 - alpha) * self._soft_hold_factor + alpha * target_factor
    self._target_factor_smooth = (1.0 - TARGET_FACTOR_FILTER_ALPHA) * self._target_factor_smooth + TARGET_FACTOR_FILTER_ALPHA * self._soft_hold_factor

    traj = np.copy(a_desired_trajectory)
    if self._target_factor_smooth < 0.99: 
        hold_strength = 1.0 - self._target_factor_smooth
        dynamic_limit = np.maximum(traj, 0.0) * self._target_factor_smooth + current_soft_hold_accel * hold_strength
        
        # [保留 V12 優化並微調] 向量化取代 for 迴圈，Blend_factor 降至 0.5 讓煞車釋放更平順
        blend_factor = 0.5
        exceeds_mask = traj > dynamic_limit
        traj = np.where(
            exceeds_mask,
            dynamic_limit * blend_factor + traj * (1.0 - blend_factor),
            traj
        )

    return traj

# =========================================================
# 統一對外接口
# =========================================================
class ACM:
  def __init__(self):
    self.enabled = False                  
    self.current_pitch = 0.0              
    self._is_first_pitch = True           
    self.personality = log.LongitudinalPersonality.standard 
    self._dtsc_is_active = False          
    self._is_normal_mode = True           

    self.coasting = CoastingLogic()       
    self.soft_hold = SoftHoldLogic()      

  @property
  def active(self):
    return self.coasting.active

  def update_states(self, cc, rs, user_ctrl_lon, v_ego, v_cruise, mode='acc', personality=log.LongitudinalPersonality.standard, dtsc_is_active=False):
    self.personality = personality
    self._dtsc_is_active = dtsc_is_active 
    self._is_normal_mode = (mode == 'acc')

    if not self.enabled or len(cc.orientationNED) != 3:
      self.coasting.active = False
      return

    new_pitch = cc.orientationNED[1]
    if self._is_first_pitch:
        self.current_pitch = new_pitch
        self._is_first_pitch = False
    else:
        alpha = PITCH_SMOOTH_ALPHA_UP if new_pitch > self.current_pitch else PITCH_SMOOTH_ALPHA_DOWN
        self.current_pitch = alpha * new_pitch + (1.0 - alpha) * self.current_pitch

    current_time = time.monotonic()
    lead = rs.leadOne 

    if self.coasting.check_emergency(lead, v_ego, current_time):
      self.coasting.active = False
      return

    self.coasting.update_lead_status(lead, v_ego, current_time)
    self.coasting.update_states(self.enabled, user_ctrl_lon, v_ego, v_cruise, self.current_pitch, dtsc_is_active, current_time)

  def update_a_desired_trajectory(self, a_desired_trajectory, v_ego=0.0, lead=None, t_follow=None):
    if self._dtsc_is_active or not self._is_normal_mode:
        return a_desired_trajectory

    if t_follow is None:
        t_follow = get_T_FOLLOW(self.personality)

    traj = self.coasting.process_trajectory(a_desired_trajectory, lead)
    traj = self.soft_hold.process_trajectory(traj, v_ego, lead, self.current_pitch, t_follow)
    
    return traj
