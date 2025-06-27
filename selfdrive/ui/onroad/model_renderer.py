import colorsys
import numpy as np
import pyray as rl
from cereal import messaging, car
from dataclasses import dataclass, field
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import DEFAULT_FPS
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT


CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0
PATH_COLOR_TRANSITION_DURATION = 0.5  # Seconds for color transition animation
PATH_BLEND_INCREMENT = 1.0 / (PATH_COLOR_TRANSITION_DURATION * DEFAULT_FPS)

MAX_POINTS = 200

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))

@dataclass
class LeadVehicle:
  glow: list[float] = field(default_factory=list)
  chevron: list[float] = field(default_factory=list)
  fill_alpha: int = 0


class ModelRenderer:
  def __init__(self):
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_factor = 1.0
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]
    self._wheel_icon = rl.load_texture("icons/steering_wheel.png")  # Load steering wheel icon
    self._font_medium = rl.load_font("fonts/Roboto-Medium.ttf", 24)  # Add font for text rendering

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None
    self._rect = None

    # Pre-allocated arrays for polygon conversion
    self._temp_points_3d = np.empty((MAX_POINTS * 2, 3), dtype=np.float32)
    self._temp_proj = np.empty((3, MAX_POINTS * 2), dtype=np.float32)

    self._exp_gradient = {
      'start': (0.0, 1.0),  # Bottom of path
      'end': (0.0, 0.0),  # Top of path
      'colors': [],
      'stops': [],
    }

    # Get longitudinal control setting from car parameters
    if car_params := Params().get("CarParams"):
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

  def set_transform(self, transform: np.ndarray):
    # 将输入的 transform 数组转换为 float32 类型，并赋值给 _car_space_transform 属性
    self._car_space_transform = transform.astype(np.float32)
    # 设置 _transform_dirty 属性为 True，表示变换信息已修改
    self._transform_dirty = True

  def draw(self, rect: rl.Rectangle, sm: messaging.SubMaster):
    # Store sm reference for later use
    self.sm = sm
    self._rect = rect

    # Check if data is up-to-date
    if (sm.recv_frame["liveCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Draw steering wheel icon
    if sm.alive.get('carState', False) and sm.alive.get('controlsState', False):
      cs = sm['carState']
      controls_state = sm['controlsState']

      # Calculate center position between lane lines
      if len(self._lane_lines) >= 2 and len(self._lane_lines[0].projected_points) > 0 and len(self._lane_lines[1].projected_points) > 0:
        left_lane = self._lane_lines[0].projected_points[-1]  # Get farthest point of left lane
        right_lane = self._lane_lines[1].projected_points[-1]  # Get farthest point of right lane
        center_x = (left_lane[0] + right_lane[0]) / 2
        center_y = (left_lane[1] + right_lane[1]) / 2

        # Set color and transparency based on lateral control state
        if controls_state.lateralControlActive:
          wheel_color = rl.Color(0, 122, 255, 255) if cs.steeringPressed else rl.Color(0, 200, 0, 255)  # Blue or Green
        else:
          wheel_color = rl.Color(255, 255, 255, 128)  # White with 50% transparency

        # Draw steering wheel icon (40x40 pixels)
        rl.draw_texture_pro(
          self._wheel_icon,
          rl.Rectangle(0, 0, self._wheel_icon.width, self._wheel_icon.height),
          rl.Rectangle(center_x - 20, center_y - 20, 40, 40),
          rl.Vector2(0, 0),
          0,
          wheel_color
        )
      return

    # 设置裁剪区域
    # Set up clipping region
    self._rect = rect
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # 更新状态
    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['liveCalibration']
    self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]

    if sm.updated['carParams']:
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = self._longitudinal_control and radar_state is not None

    # 当需要时更新模型数据
    # Update model data when needed
    model_updated = sm.updated['modelV2']
    if model_updated or sm.updated['radarState'] or self._transform_dirty:
      if model_updated:
        self._update_raw_points(model)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        return

      self._update_model(lead_one, path_x_array)
      if render_lead_indicator:
        self._update_leads(radar_state, path_x_array)
      self._transform_dirty = False


    # 绘制元素
    # Draw elements
    self._draw_lane_lines()
    self._draw_path(sm)

    if render_lead_indicator and radar_state:
      self._draw_lead_indicator()

  def _update_raw_points(self, model):
    """Update raw 3D points from model data"""
    # 更新模型路径的原始点
    self._path.raw_points = np.array([model.position.x, model.position.y, model.position.z], dtype=np.float32).T

    # 遍历模型的车道线
    for i, lane_line in enumerate(model.laneLines):
      # 更新车道线的原始点
      self._lane_lines[i].raw_points = np.array([lane_line.x, lane_line.y, lane_line.z], dtype=np.float32).T

    # 遍历模型的道路边缘
    for i, road_edge in enumerate(model.roadEdges):
      # 更新道路边缘的原始点
      self._road_edges[i].raw_points = np.array([road_edge.x, road_edge.y, road_edge.z], dtype=np.float32).T

    # 更新车道线的概率
    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)

    # 更新道路边缘的标准差
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)

    # 更新加速度的x分量
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_leads(self, radar_state, path_x_array):
    """Update positions of lead vehicles"""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]  # 初始化两个前车对象

    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        # 从路径中获取前车位置的z坐标
        # Get z-coordinate from path at the lead vehicle position
        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        point = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)
        if point:
          # 更新前车对象
          self._lead_vehicles[i] = self._update_lead_vehicle(d_rel, v_rel, point, self._rect)

  def _update_model(self, lead, path_x_array):
    """Update model visualization data based on model message"""
    # 将路径最后一个点的距离限制在最小和最大绘制距离之间
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    # 获取路径长度的索引，用于后续处理
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # 使用原始点更新车道线
    # Update lane lines using raw points
    for i, lane_line in enumerate(self._lane_lines):
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, 0.025 * self._lane_line_probs[i], 0.0, max_idx
      )

    # 使用原始点更新道路边缘
    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, 0.025, 0.0, max_idx)

    # 使用原始点更新路径
    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      # 更新最大距离，基于前车的相对距离
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    # 获取路径长度的索引，基于新的最大距离
    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    # 更新路径的投影点
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, 0.9, self._path_offset_z, max_idx, allow_invert=False
    )

    # 更新实验性梯度
    self._update_experimental_gradient(self._rect.height)

  def _update_experimental_gradient(self, height):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      track_idx = max_len - i - 1  # flip idx to start from bottom right
      track_y = self._path.projected_points[track_idx][1]
      if track_y < 0 or track_y > height:
        i += 1
        continue

          # 计算基于加速度的颜色
      # Calculate color based on acceleration
      lin_grad_point = (height - track_y) / height

          # 加速：120，减速：0
      # speed up: 120, slow down: 0
      path_hue = max(min(60 + self._acceleration_x[i] * 35, 120), 0)
      path_hue = int(path_hue * 100 + 0.5) / 100

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = self._map_val(saturation, 0.0, 1.0, 0.95, 0.62)
      alpha = self._map_val(lin_grad_point, 0.75 / 2.0, 0.75, 0.4, 0.0)

          # 使用HSL到RGB的转换
      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

          # 除非下一个是最后一个，否则跳过一个点
      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

      # 将梯度存储在路径对象中
    # Store the gradient in the path object
    self._exp_gradient['colors'] = segment_colors
    self._exp_gradient['stops'] = gradient_stops

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 2.35
    x = np.clip(point[0], 0.0, rect.width - sz / 2)
    y = min(point[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    # Store distance and speed for text display
    lead = LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))
    lead.d_rel = d_rel
    lead.v_rel = v_rel
    return lead
    # 箭头形状位置
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    # 返回前车对象
    return LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))

  def _draw_lane_lines(self):
    """绘制车道线和道路边缘"""
    # 遍历车道线列表
    for i, lane_line in enumerate(self._lane_lines):
      # 如果车道线投影点为空，则跳过当前车道线
      if lane_line.projected_points.size == 0:
        continue

      # 计算车道线的透明度
      alpha = np.clip(self._lane_line_probs[i], 0.0, 0.7)
      # 根据透明度计算颜色
      color = rl.Color(255, 255, 255, int(alpha * 255))
      # 绘制车道线多边形
      draw_polygon(self._rect, lane_line.projected_points, color)

    # 遍历道路边缘列表
    for i, road_edge in enumerate(self._road_edges):
      # 如果道路边缘投影点为空，则跳过当前道路边缘
      if road_edge.projected_points.size == 0:
        continue

      # 计算道路边缘的透明度
      alpha = np.clip(1.0 - self._road_edge_stds[i], 0.0, 1.0)
      # 根据透明度计算颜色
      color = rl.Color(255, 0, 0, int(alpha * 255))
      # 绘制道路边缘多边形
      draw_polygon(self._rect, road_edge.projected_points, color)

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    if self._experimental_mode:
      # 绘制带有加速度色彩的路径
      # Draw with acceleration coloring
      if len(self._exp_gradient['colors']) > 2:
        draw_polygon(self._rect, self._path.projected_points, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
    else:
      # 绘制带有油门/无油门渐变色的路径
      # Draw with throttle/no throttle gradient
      allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control

      # 如果油门状态改变，则开始过渡
      # Start transition if throttle state changes
      if allow_throttle != self._prev_allow_throttle:
        self._prev_allow_throttle = allow_throttle
        self._blend_factor = max(1.0 - self._blend_factor, 0.0)

      # 更新混合因子
      # Update blend factor
      if self._blend_factor < 1.0:
        self._blend_factor = min(self._blend_factor + PATH_BLEND_INCREMENT, 1.0)

      begin_colors = NO_THROTTLE_COLORS if allow_throttle else THROTTLE_COLORS
      end_colors = THROTTLE_COLORS if allow_throttle else NO_THROTTLE_COLORS

      # 根据过渡混合颜色
      # Blend colors based on transition
      blended_colors = self._blend_colors(begin_colors, end_colors, self._blend_factor)
      gradient = {
        'start': (0.0, 1.0),  # 路径底部
        # Bottom of path
        'end': (0.0, 0.0),  # 路径顶部
        # Top of path
        'colors': blended_colors,
        'stops': [0.0, 0.5, 1.0],
      }
      draw_polygon(self._rect, self._path.projected_points, gradient=gradient)

  def _draw_lead_indicator(self):
    # Draw lead vehicles if available
    for i, lead in enumerate(self._lead_vehicles):
      if not lead.glow or not lead.chevron:
        continue

      rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(218, 202, 37, 255))
      rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(201, 34, 49, lead.fill_alpha))

      # Calculate text position below the triangle
      text_y = lead.chevron[0][1] + 30  # 30 pixels below triangle
      text_x = lead.chevron[0][0] - 20  # centered below triangle

      # Draw distance text (blue)
      distance_text = f"{lead.d_rel:.1f}m"
      rl.draw_text_ex(
        self._font_medium,
        distance_text,
        rl.Vector2(text_x, text_y),
        24,  # font size
        0,
        rl.Color(0, 122, 255, 255)  # Blue color
      )

      # Draw relative speed text (blue)
      speed_text = f"{lead.v_rel:+.1f}m/s"
      rl.draw_text_ex(
        self._font_medium,
        speed_text,
        rl.Vector2(text_x, text_y + 30),  # 30px below distance
        24,  # font size
        0,
        rl.Color(0, 122, 255, 255)  # Blue color
      )

    # Draw vehicle dynamics info at bottom
    if hasattr(self, 'sm') and self.sm.alive.get('carState', False) and self.sm.alive.get('controlsState', False):
      car_state = self.sm['carState']
      controls_state = self.sm['controlsState']

      # Format and display info
      info_lines = [
        f"预测曲率: {controls_state.curvature:.4f} m⁻¹",
        f"期望曲率: {controls_state.desiredCurvature:.4f} m⁻¹",
        f"当前曲率: {-car_state.yawRate/max(car_state.vEgoRaw,0.1):.4f} m⁻¹",
        f"方向盘角度: {car_state.steeringAngleDeg:.1f}°",
        f"输出扭矩: {controls_state.actuatorsOutput.steer:.2f} Nm",
        f"横滚角: {math.degrees(car_state.roll):.1f}°" if hasattr(car_state, 'roll') else "横滚角: N/A",
        f"俯仰角: {math.degrees(car_state.pitch):.1f}°" if hasattr(car_state, 'pitch') else "俯仰角: N/A",
        f"偏航角: {math.degrees(car_state.yaw):.1f}°" if hasattr(car_state, 'yaw') else "偏航角: N/A"
      ]

      # Draw each line with 50% transparent black text
      start_y = self._rect.y + self._rect.height - 180 + 20  # Same position as before
      for i, line in enumerate(info_lines):
        rl.draw_text_ex(
          self._font_medium,
          line,
          rl.Vector2(self._rect.x + 20, start_y + i * 22),
          20,  # font size
          0,
          rl.Color(0, 0, 0, 128)  # 50% transparency black
        )

  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_height: float) -> int:
    """Get the index corresponding to the given path height"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_height)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, allow_invert: bool = True) -> np.ndarray:
    """Convert 3D line to 2D polygon for rendering."""
    if line.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Slice points and filter non-negative x-coordinates
    points = line[:max_idx + 1]
    points = points[points[:, 0] >= 0]
    if points.shape[0] == 0:
      return np.empty((0, 2), dtype=np.float32)

    # Create left and right 3D points in one array
    n_points = points.shape[0]
    points_3d = self._temp_points_3d[:n_points * 2]
    points_3d[:n_points, 0] = points_3d[n_points:, 0] = points[:, 0]
    points_3d[:n_points, 1] = points[:, 1] - y_off
    points_3d[n_points:, 1] = points[:, 1] + y_off
    points_3d[:n_points, 2] = points_3d[n_points:, 2] = points[:, 2] + z_off

    # Single matrix multiplication for projections
    proj = np.ascontiguousarray(self._temp_proj[:, :n_points * 2])  # Slice the pre-allocated array
    np.dot(self._car_space_transform, points_3d.T, out=proj)
    valid_z = np.abs(proj[2]) > 1e-6
    if not np.any(valid_z):
      return np.empty((0, 2), dtype=np.float32)

    # Compute screen coordinates
    screen = proj[:2, valid_z] / proj[2, valid_z][None, :]
    left_screen = screen[:, :n_points].T
    right_screen = screen[:, n_points:].T

    # Ensure consistent shapes by re-aligning valid points
    valid_points = np.minimum(left_screen.shape[0], right_screen.shape[0])
    if valid_points == 0:
      return np.empty((0, 2), dtype=np.float32)
    left_screen = left_screen[:valid_points]
    right_screen = right_screen[:valid_points]

    if self._clip_region:
      clip = self._clip_region
      bounds_mask = (
        (left_screen[:, 0] >= clip.x) & (left_screen[:, 0] <= clip.x + clip.width) &
        (left_screen[:, 1] >= clip.y) & (left_screen[:, 1] <= clip.y + clip.height) &
        (right_screen[:, 0] >= clip.x) & (right_screen[:, 0] <= clip.x + clip.width) &
        (right_screen[:, 1] >= clip.y) & (right_screen[:, 1] <= clip.y + clip.height)
      )
      if not np.any(bounds_mask):
        return np.empty((0, 2), dtype=np.float32)
      left_screen = left_screen[bounds_mask]
      right_screen = right_screen[bounds_mask]

    if not allow_invert and left_screen.shape[0] > 1:
      keep = np.concatenate(([True], np.diff(left_screen[:, 1]) < 0))
      left_screen = left_screen[keep]
      right_screen = right_screen[keep]
      if left_screen.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    return np.vstack((left_screen, right_screen[::-1])).astype(np.float32)

  @staticmethod
  def _map_val(x, x0, x1, y0, y1):
    x = np.clip(x, x0, x1)
    ra = x1 - x0
    rb = y1 - y0
    return (x - x0) * rb / ra + y0 if ra != 0 else y0

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255),
        int(a * 255)
    )

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]
