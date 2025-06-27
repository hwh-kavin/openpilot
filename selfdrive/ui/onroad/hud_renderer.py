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
  blue: rl.Color = rl.Color(0, 122, 255, 255)  # Added for speed display   kavin 2025年6月27日
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
    self.lead_d_rel: float = 0.0
    self.lead_v_rel: float = 0.0
    self.lead_visible: bool = False
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
      self.steering_pressed = False
      return

    controls_state = sm['controlsState']
    car_state = sm['carState']
    self.steering_pressed = car_state.steeringPressed

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

    # Update lead vehicle data
    self.lead_d_rel = sm['modelV2'].lead.dRel if sm.alive['modelV2'] and hasattr(sm['modelV2'], 'lead') else 0.0
    self.lead_v_rel = sm['modelV2'].lead.vRel if sm.alive['modelV2'] and hasattr(sm['modelV2'], 'lead') else 0.0
    self.lead_visible = sm['longitudinalPlan'].hasLead if sm.alive['longitudinalPlan'] else False

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
    self._draw_curvature_info(rect, sm)
    self._draw_lead_info(rect)

  def _draw_set_speed(self, rect: rl.Rectangle) -> None:
    """Draw the MAX speed indicator box."""
    set_speed_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + (UI_CONFIG.set_speed_width_imperial - set_speed_width) // 2
    y = rect.y + 45

    set_speed_rect = rl.Rectangle(x, y, set_speed_width, UI_CONFIG.set_speed_height)
    rl.draw_rectangle_rounded(set_speed_rect, 0.2, 30, COLORS.black_translucent)
    rl.draw_rectangle_rounded_lines_ex(set_speed_rect, 0.2, 30, 6, COLORS.border_translucent)

    max_color = COLORS.grey
    set_speed_color = COLORS.dark_grey
    if self.is_cruise_set:
      set_speed_color = COLORS.white
      if ui_state.status == UIStatus.ENGAGED:
        max_color = COLORS.engaged
      elif ui_state.status == UIStatus.DISENGAGED:
        max_color = COLORS.disengaged
      elif ui_state.status == UIStatus.OVERRIDE:
        max_color = COLORS.override

    max_text = "MAX"
    max_text_width = measure_text_cached(self._font_semi_bold, max_text, FONT_SIZES.max_speed).x
    rl.draw_text_ex(
      self._font_semi_bold,
      max_text,
      rl.Vector2(x + (set_speed_width - max_text_width) / 2, y + 27),
      FONT_SIZES.max_speed,
      0,
      max_color,
    )

    set_speed_text = CRUISE_DISABLED_CHAR if not self.is_cruise_set else str(round(self.set_speed))
    speed_text_width = measure_text_cached(self._font_bold, set_speed_text, FONT_SIZES.set_speed).x
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
    speed_text = str(round(self.speed))
    speed_text_size = measure_text_cached(self._font_bold, speed_text, FONT_SIZES.current_speed)
    speed_pos = rl.Vector2(rect.x + rect.width / 2 - speed_text_size.x / 2, 180 - speed_text_size.y / 2)
    rl.draw_text_ex(self._font_bold, speed_text, speed_pos, FONT_SIZES.current_speed, 0, COLORS.blue)

    unit_text = "km/h" if ui_state.is_metric else "mph"
    unit_text_size = measure_text_cached(self._font_medium, unit_text, FONT_SIZES.speed_unit)
    unit_pos = rl.Vector2(rect.x + rect.width / 2 - unit_text_size.x / 2, 290 - unit_text_size.y / 2)
    rl.draw_text_ex(self._font_medium, unit_text, unit_pos, FONT_SIZES.speed_unit, 0, COLORS.white_translucent)

  def _draw_wheel_icon(self, rect: rl.Rectangle) -> None:
    """Draw the steering wheel icon with status-based opacity and driver takeover indication."""
    # Calculate wheel icon center position
    center_x = int(rect.x + rect.width - UI_CONFIG.border_size - UI_CONFIG.button_size / 2)
    center_y = int(rect.y + UI_CONFIG.border_size + UI_CONFIG.button_size / 2)

    # Draw semi-transparent black circle background
    rl.draw_circle(center_x, center_y, UI_CONFIG.button_size / 2, COLORS.black_translucent)

    # Set wheel color based on driver takeover status
    steering_pressed = getattr(self, 'steering_pressed', False)
    wheel_color = COLORS.disengaged if steering_pressed else COLORS.engaged

    # Set wheel icon opacity based on UI status
    opacity = 0.7 if ui_state.status == UIStatus.DISENGAGED else 1.0
    img_pos = rl.Vector2(center_x - self._wheel_texture.width / 2, center_y - self._wheel_texture.height / 2)
    rl.draw_texture_v(self._wheel_texture, img_pos, rl.Color(wheel_color.r, wheel_color.g, wheel_color.b, int(255 * opacity)))

  def _draw_curvature_info(self, rect: rl.Rectangle, sm: SubMaster) -> None:
    """Draw curvature and orientation information at the bottom of the screen."""
    if not sm.alive.get('carState', False) or not sm.alive.get('controlsState', False):
      return

    car_state = sm['carState']
    controls_state = sm['controlsState']

    # Get curvature data
    current_curvature = -car_state.yawRate / max(car_state.vEgoRaw, 0.1)
    desired_curvature = controls_state.curvature
    steering_angle = car_state.steeringAngleDeg
    torque_output = controls_state.actuatorsOutput.steer

    # Get orientation data (roll, pitch, yaw in degrees)
    roll_deg = math.degrees(car_state.roll) if hasattr(car_state, 'roll') else 0.0
    pitch_deg = math.degrees(car_state.pitch) if hasattr(car_state, 'pitch') else 0.0
    yaw_deg = math.degrees(car_state.yaw) if hasattr(car_state, 'yaw') else 0.0

    # Format strings with units
    info_lines = [
      f"预测曲率: {desired_curvature:.4f} m⁻¹",
      f"期望曲率: {controls_state.desiredCurvature:.4f} m⁻¹",
      f"当前曲率: {current_curvature:.4f} m⁻¹",
      f"方向盘角度: {steering_angle:.1f}°",
      f"输出扭矩: {torque_output:.2f} Nm",
      f"横滚角: {roll_deg:.1f}°",
      f"俯仰角: {pitch_deg:.1f}°",
      f"偏航角: {yaw_deg:.1f}°"
    ]

    # Draw each line at the bottom of the screen
    y_pos = rect.y + rect.height - 180  # Start 180px from bottom to accommodate more lines
    for i, line in enumerate(info_lines):
      text_size = measure_text_cached(self._font_medium, line, 30)
      x_pos = rect.x + (rect.width - text_size.x) / 2  # Center horizontally
      rl.draw_text_ex(
        self._font_medium,
        line,
        rl.Vector2(x_pos, y_pos + i * 30),  # 30px line spacing for compact display
        30,  # Font size
        0,
        COLORS.blue
      )
    opacity = 0.7 if ui_state.status == UIStatus.DISENGAGED else 1.0
    img_pos = rl.Vector2(center_x - self._wheel_texture.width / 2, center_y - self._wheel_texture.height / 2)
    # 绘制方向盘图标，透明度根据UI状态调整
    # Draw texture with opacity based on UI status
    rl.draw_texture_v(self._wheel_texture, img_pos, rl.Color(255, 255, 255, int(255 * opacity)))

  def _draw_lead_info(self, rect: rl.Rectangle) -> None:
    """Draw lead vehicle distance and relative speed."""
    if not self.lead_visible or self.lead_d_rel <= 0:
      return

    # Calculate display position (below speed unit)
    x = rect.x + rect.width / 2
    y = rect.y + 350  # Below speed unit

    # Format display text
    distance_text = f"{self.lead_d_rel:.1f}m"
    speed_text = f"{self.lead_v_rel:+.1f}m/s"

    # Measure text size
    distance_size = measure_text_cached(self._font_medium, distance_text, 40)
    speed_size = measure_text_cached(self._font_medium, speed_text, 40)

    # Draw distance
    rl.draw_text_ex(
      self._font_medium,
      distance_text,
      rl.Vector2(x - distance_size.x / 2, y),
      40,
      0,
      COLORS.white
    )

    # Draw relative speed (red for negative, white for positive)
    rl.draw_text_ex(
      self._font_medium,
      speed_text,
      rl.Vector2(x - speed_size.x / 2, y + 50),  # 50px below distance
      40,
      0,
      COLORS.white if self.lead_v_rel >= 0 else rl.Color(255, 100, 100, 255)
    )
