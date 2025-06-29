import pyray as rl
from dataclasses import dataclass
from cereal.messaging import SubMaster
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.common.conversions import Conversions as CV

# Constants
SET_SPEED_NA = 255
KM_TO_MILE = 0.621371
CRUISE_DISABLED_CHAR = '–'


@dataclass(frozen=True)
class UIConfig:
  header_height: int = 300
  border_size: int = 30
  button_size: int = 192
  set_speed_width_metric: int = 200
  set_speed_width_imperial: int = 172
  set_speed_height: int = 204
  wheel_icon_size: int = 144


@dataclass(frozen=True)
class FontSizes:
  current_speed: int = 176
  speed_unit: int = 66
  max_speed: int = 40
  set_speed: int = 90


@dataclass(frozen=True)
class Colors:
  white: rl.Color = rl.Color(255, 255, 255, 255)
  disengaged: rl.Color = rl.Color(145, 155, 149, 255)
  override: rl.Color = rl.Color(145, 155, 149, 255)  # Added
  engaged: rl.Color = rl.Color(128, 216, 166, 255)
  disengaged_bg: rl.Color = rl.Color(0, 0, 0, 153)
  override_bg: rl.Color = rl.Color(145, 155, 149, 204)
  engaged_bg: rl.Color = rl.Color(128, 216, 166, 204)
  grey: rl.Color = rl.Color(166, 166, 166, 255)
  dark_grey: rl.Color = rl.Color(114, 114, 114, 255)
  black_translucent: rl.Color = rl.Color(0, 0, 0, 166)
  white_translucent: rl.Color = rl.Color(255, 255, 255, 200)
  border_translucent: rl.Color = rl.Color(255, 255, 255, 75)
  header_gradient_start: rl.Color = rl.Color(0, 0, 0, 114)
  header_gradient_end: rl.Color = rl.Color(0, 0, 0, 0)


UI_CONFIG = UIConfig()
FONT_SIZES = FontSizes()
COLORS = Colors()


class HudRenderer:
  def __init__(self):
    """Initialize the HUD renderer."""
    self.is_cruise_set: bool = False
    self.is_cruise_available: bool = False
    self.set_speed: float = SET_SPEED_NA
    self.speed: float = 0.0
    self.v_ego_cluster_seen: bool = False
    self._wheel_texture: rl.Texture = gui_app.texture('icons/chffr_wheel.png', UI_CONFIG.wheel_icon_size, UI_CONFIG.wheel_icon_size)
    self._font_semi_bold: rl.Font = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)

  def _update_state(self, sm: SubMaster) -> None:
    """Update HUD state based on car state and controls state."""
    if sm.recv_frame["carState"] < ui_state.started_frame:
      self.is_cruise_set = False
      self.set_speed = SET_SPEED_NA
      self.speed = 0.0
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']

    v_cruise_cluster = car_state.vCruiseCluster
    self.set_speed = (
      controls_state.vCruiseDEPRECATED if v_cruise_cluster == 0.0 else v_cruise_cluster
    )
    self.is_cruise_set = 0 < self.set_speed < SET_SPEED_NA
    self.is_cruise_available = self.set_speed != -1

    if self.is_cruise_set and not ui_state.is_metric:
      self.set_speed *= KM_TO_MILE

    v_ego_cluster = car_state.vEgoCluster
    self.v_ego_cluster_seen = self.v_ego_cluster_seen or v_ego_cluster != 0.0
    v_ego = v_ego_cluster if self.v_ego_cluster_seen else car_state.vEgo
    speed_conversion = CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH
    self.speed = max(0.0, v_ego * speed_conversion)

  def draw(self, rect: rl.Rectangle, sm: SubMaster) -> None:
    """Render HUD elements to the screen."""
    self._update_state(sm)
    rl.draw_rectangle_gradient_v(
      int(rect.x),
      int(rect.y),
      int(rect.width),
      UI_CONFIG.header_height,
      COLORS.header_gradient_start,
      COLORS.header_gradient_end,
    )

    if self.is_cruise_available:
      self._draw_set_speed(rect)

    self._draw_current_speed(rect)
    self._draw_wheel_icon(rect)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    # 根据当前单位设置，计算设置速度的宽度
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    # 计算设置速度矩形框的左上角x坐标
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    # 计算设置速度矩形框的左上角y坐标
    y = rect.y + 45

    # 创建设置速度矩形框
    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    # 绘制圆角矩形框
    rl.draw_rectangle_rounded(set_speed_rect, 0.2, 30, COLORS.black_translucent)
    # 绘制圆角矩形框的边框
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.2, 30, 6, COLORS.border_translucent)

    # 设置最大速度文本的颜色
    max_color = COLORS.grey
    # 设置设置速度文本的颜色
    set_speed_color = COLORS.dark_grey
    # 如果已经设置了巡航速度
    if self.is_cruise_set:
      # 设置设置速度文本的颜色为白色
      set_speed_color = COLORS.white
      # 根据UI状态设置最大速度文本的颜色
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.engaged
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.disengaged
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.override

    # 最大速度文本
    max_text = "MAX"
    # 计算最大速度文本的宽度
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    # 绘制最大速度文本
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    # 设置速度文本，如果没有设置巡航速度则显示为禁用字符
    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    # 计算设置速度文本的宽度
    speed_text_width = measure_text_cached(self._font_bold, set_speed_text, FONT_SIZES.set_speed).x
    # 绘制设置速度文本
    rl.draw_text_ex(
      self._font_bold,
      set_speed_text,
      rl.Vector2(x + (set_speed_width - speed_text_width) / 2, y + 77),
      FONT_SIZES.set_speed,
      0,
      set_speed_color,
    )

  def _draw_current_speed(self, rect: rl.Rectangle) -> None:
    """Draw the current vehicle speed and unit."""
    # 将速度四舍五入并转换为字符串
    speed_text = str(round(self.speed))
    # 计算速度文本的尺寸
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    # 计算速度文本的位置
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    # 绘制速度文本
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.white)

    # 根据单位系统确定速度单位
    unit_text = "km/h" if ui_state.is_metric else "mph"
    # 计算速度单位文本的尺寸
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    # 计算速度单位文本的位置
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    # 绘制速度单位文本
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.white_translucent)

  def _draw_wheel_icon(self, rect: rl.Rectangle) -> None:
    """Draw the steering wheel icon with parameters."""
    if not self._wheel_icon:
      return

    # 方向盘图标绘制
    wheel_color = COLORS.white
    if self.car_state.steeringPressed:
      wheel_color = COLORS.override
    elif self.lat_active:
      wheel_color = COLORS.engaged

    center_x = rect.x + rect.width // 2
    center_y = rect.y + UI_CONFIG.header_height // 2
    rl.draw_texture_pro(
      self._wheel_icon,
      rl.Rectangle(0, 0, self._wheel_icon.width, self._wheel_icon.height),
      rl.Rectangle(center_x - 25, center_y - 25, 50, 50),
      rl.Vector2(25, 25),
      -self.car_state.steeringAngleDeg * 0.8,
      wheel_color
    )

    # 在方向盘下方显示参数
    if hasattr(self, 'sm') and self.sm.alive.get('carState', False) and self.sm.alive.get('controlsState', False):
      cs = self.sm['carState']
      ctrl = self.sm['controlsState']

      params = [
        ("预测曲率", f"{ctrl.curvature:.4f}", "m⁻¹"),
        ("期望曲率", f"{ctrl.desiredCurvature:.4f}", "m⁻¹"),
        ("当前曲率", f"{-cs.yawRate/max(cs.vEgoRaw,0.1):.4f}", "m⁻¹"),
        ("方向盘角度", f"{cs.steeringAngleDeg:.1f}", "°"),
        ("输出扭矩", f"{ctrl.actuatorsOutput.steer:.2f}", "Nm"),
        ("横滚角", f"{math.degrees(cs.roll):.1f}" if hasattr(cs, 'roll') else "N/A", "°"),
        ("俯仰角", f"{math.degrees(cs.pitch):.1f}" if hasattr(cs, 'pitch') else "N/A", "°"),
        ("偏航角", f"{math.degrees(cs.yaw):.1f}" if hasattr(cs, 'yaw') else "N/A", "°")
      ]

      start_x = center_x - 100
      start_y = center_y + 40  # 方向盘图标下方40px
      for i, (label, value, unit) in enumerate(params):
        # 中文标签
        rl.draw_text_ex(
          self._font_medium,
          f"{label}:",
          rl.Vector2(start_x, start_y + i * 22),
          18,
          0,
          COLORS.grey
        )
        # 数值+单位
        rl.draw_text_ex(
          self._font_medium,
          f"{value} {unit}",
          rl.Vector2(start_x + 80, start_y + i * 22),
          18,
          0,
          COLORS.white
        )