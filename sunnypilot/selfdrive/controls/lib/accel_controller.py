"""
Copyright (c) 2021-, rav4kumar, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from cereal import custom
import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params

# 定義加速度性格的列舉值
AccelPersonality = custom.LongitudinalPlanSP.AccelerationPersonality
ACCEL_PERSONALITY_OPTIONS = [AccelPersonality.eco, AccelPersonality.normal, AccelPersonality.sport]

# 加速度設定檔 (Max Accel Profiles)
# 根據不同的性格與車速(v_ego)，定義車輛允許的最大正加速度
MAX_ACCEL_PROFILES = {
  AccelPersonality.eco:       [1.00, 0.40, 1.00, 1.40,  1.20, 1.00, 0.80, 0.40,  0.40, 0.20, 0.12, 0.08],
  AccelPersonality.normal:    [1.50, 0.60, 1.20, 1.80,  1.40, 1.20, 1.00, 0.80,  0.60, 0.40, 0.30, 0.10],
  AccelPersonality.sport:     [2.00, 0.80, 1.40, 2.00,  1.60, 1.40, 1.20, 1.00,  0.90, 0.80, 0.50, 0.25],
}
# 對應最大加速度的車速節點 (單位: m/s)
MAX_ACCEL_BREAKPOINTS =       [0.0,  0.5,  1.0,  4.0,   6.0,  9.0,  11.0, 16.0,  20.0, 25.0, 30.0, 55.0]

# 煞車設定檔 (Min Accel Profiles)
# 根據不同的性格與車速，定義車輛允許的最大減速度 (負值)
MIN_ACCEL_PROFILES = {
  AccelPersonality.eco:    [-.002, -.003, -0.25, -0.27, -0.30, -0.35, -0.44, -2.0],
  AccelPersonality.normal: [-.002, -.003, -0.25, -0.27, -0.30, -0.50, -0.76, -2.0],
  AccelPersonality.sport:  [-.002, -.003, -0.25, -0.27, -0.30, -0.55, -0.80, -2.0],
}
# 對應最大減速度的車速節點 (單位: m/s)
MIN_ACCEL_BREAKPOINTS =    [2.0, 3.0, 4.5, 5.0, 6.0, 7.0, 9.0, 25]

# 平滑化參數 (Exponential Moving Average)
DECEL_SMOOTH_ALPHA = 0.40  # 煞車平滑度：數值越低，煞車變化越平滑/緩慢
ACCEL_SMOOTH_ALPHA = 0.90  # 加速平滑度：數值越高，加速反應越快/直接

# 不對稱速率限制 (Asymmetric rate limiting)
MAX_DECEL_INCREASE_RATE = 1.3  # 加重煞車時的最大變化率 (m/s² per second)
MAX_DECEL_DECREASE_RATE = 1.0  # 放開煞車時的最大變化率 (m/s² per second)

class AccelPersonalityController:
  def __init__(self):
    self.params = Params()
    self.frame = 0  # 用於追蹤目前的執行幀數

    # 狀態變數：儲存上一次計算的加減速限制
    self.last_max_accel = 2.0
    self.last_min_accel = -0.01

    # 緩存變數：避免同一個 frame 內重複計算導致狀態異常推進
    self.calculated_frame = -1
    self.current_max_accel = 2.0
    self.current_min_accel = -0.01

    self.first_run = True

    # 初始化性格與啟用狀態，使用安全讀取機制
    self._accel_personality = self._load_personality()
    self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  def _load_personality(self) -> int:
    """
    安全地從 Params 讀取並轉換性格設定。
    若資料型別錯誤 (無法轉 int) 或數值不在允許的選項中，則退回 normal 模式，防止系統崩潰。
    """
    val = self.params.get('AccelPersonality')
    if val is not None:
      try:
        parsed_val = int(val)
        if parsed_val in ACCEL_PERSONALITY_OPTIONS:
          return parsed_val
      except ValueError:
        pass # 轉換失敗，忽略並使用下方預設值
    return int(AccelPersonality.normal)

  def update(self):
    """
    每個模型週期 (通常為 20Hz, 取決於 DT_MDL) 呼叫一次。
    用於降頻檢查 Params，避免每個 frame 都去讀取硬碟或記憶體降低效能。
    """
    self.frame += 1
    # 每 1 秒 (1.0 / DT_MDL 幀) 更新一次使用者設定
    if self.frame % int(1.0 / DT_MDL) == 0:
      self._accel_personality = self._load_personality()
      self._enabled = self.params.get_bool('AccelPersonalityEnabled')

  @property
  def accel_personality(self) -> int:
    return self._accel_personality

  def get_accel_personality(self) -> int:
    return int(self._accel_personality)

  def set_accel_personality(self, personality: int):
    """
    設定加速度性格並存入 Params。
    """
    if personality in ACCEL_PERSONALITY_OPTIONS:
      self._accel_personality = personality
      # Params.put 需要傳入字串或 bytes，將 int 轉為 str 避免 TypeError
      self.params.put('AccelPersonality', str(int(personality)))

  def cycle_accel_personality(self) -> int:
    """
    在 eco -> normal -> sport 之間循環切換。
    """
    current = self._accel_personality
    next_personality = ACCEL_PERSONALITY_OPTIONS[(ACCEL_PERSONALITY_OPTIONS.index(current) + 1) % len(ACCEL_PERSONALITY_OPTIONS)]
    self.set_accel_personality(next_personality)
    return int(next_personality)

  def get_accel_limits(self, v_ego: float) -> tuple[float, float]:
    """
    根據當前車速計算並回傳 (最小加速度, 最大加速度)。
    包含平滑化處理與速率限制。
    """
    # 檢查此 frame 是否已經計算過，若是則直接回傳快取，避免 getter 重複修改內部狀態
    if self.calculated_frame == self.frame:
      return float(self.current_min_accel), float(self.current_max_accel)

    v_ego = max(0.0, v_ego)

    # 使用 numpy 線性插值取得目標加減速度
    target_max = np.interp(v_ego, MAX_ACCEL_BREAKPOINTS, MAX_ACCEL_PROFILES[self.accel_personality])
    target_min = np.interp(v_ego, MIN_ACCEL_BREAKPOINTS, MIN_ACCEL_PROFILES[self.accel_personality])

    # 首次執行：直接賦值，不進行平滑與速率限制
    if self.first_run:
      self.last_max_accel, self.last_min_accel = target_max, target_min
      self.current_max_accel, self.current_min_accel = target_max, target_min
      self.first_run = False
      self.calculated_frame = self.frame
      return float(target_min), float(target_max)

    # 1. 平滑化 (Smoothing) - 使用指數移動平均
    self.last_max_accel = ACCEL_SMOOTH_ALPHA * target_max + (1 - ACCEL_SMOOTH_ALPHA) * self.last_max_accel
    smoothed_decel = DECEL_SMOOTH_ALPHA * target_min + (1 - DECEL_SMOOTH_ALPHA) * self.last_min_accel

    # 2. 速率限制 (Rate Limiting)
    raw_change = smoothed_decel - self.last_min_accel

    # 合併不對稱的速率限制寫法，設定單幀最大變化量
    lower_limit = -MAX_DECEL_INCREASE_RATE * DT_MDL # 加重煞車的極限
    upper_limit = MAX_DECEL_DECREASE_RATE * DT_MDL  # 放開煞車的極限

    decel_change = np.clip(raw_change, lower_limit, upper_limit)
    self.last_min_accel += decel_change

    # 3. 動態安全走廊 (Dynamic Safety Corridor)
    # 確保最小加速度始終嚴格小於最大加速度，保留至少 0.1 或 5% 的容錯空間，
    # 防止 MPC Solver (模型預測控制求解器) 在高加速度時發生崩潰。
    gap = max(0.1, abs(self.last_max_accel) * 0.05)

    if self.last_min_accel > self.last_max_accel - gap:
      self.last_min_accel = self.last_max_accel - gap

    # 更新緩存變數與標記，確保同一個 frame 內再次呼叫時直接回傳
    self.current_min_accel = self.last_min_accel
    self.current_max_accel = self.last_max_accel
    self.calculated_frame = self.frame

    return float(self.current_min_accel), float(self.current_max_accel)

  def get_min_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[0]

  def get_max_accel(self, v_ego: float) -> float:
    return self.get_accel_limits(v_ego)[1]

  def is_enabled(self) -> bool:
    return self._enabled

  def set_enabled(self, enabled: bool):
    self._enabled = enabled
    self.params.put_bool('AccelPersonalityEnabled', enabled)

  def toggle_enabled(self) -> bool:
    current = self._enabled
    self.set_enabled(not current)
    return not current

  def reset(self):
    """
    重置所有狀態與 Params 設定。
    """
    self._accel_personality = AccelPersonality.normal
    self.params.put('AccelPersonality', str(int(AccelPersonality.normal)))

    self.frame = 0
    self.calculated_frame = -1
    self.last_max_accel = 2.0
    self.last_min_accel = -0.01
    self.current_max_accel = 2.0
    self.current_min_accel = -0.01
    self.first_run = True
