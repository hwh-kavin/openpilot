"""
BluePilot Debug Panel Color Constants
Dedicated colors for the onroad debug panel, graphs, tabs, and data cards.
"""

import pyray as rl


class DebugColors:
  # Panel backgrounds
  PANEL_BG = rl.Color(30, 30, 30, 230)
  PANEL_BG_DARKER = rl.Color(20, 20, 20, 230)
  CARD_BG = rl.Color(40, 42, 46, 240)

  # Graph container
  GRAPH_CONTAINER_TOP = rl.Color(44, 62, 80, 220)
  GRAPH_CONTAINER_MID = rl.Color(32, 33, 35, 220)
  GRAPH_CONTAINER_BOTTOM = rl.Color(26, 37, 47, 220)

  # Graph area background
  GRAPH_BG = rl.Color(20, 25, 30, 200)
  GRAPH_BG_DARK = rl.Color(15, 20, 25, 200)
  GRAPH_BORDER = rl.Color(0, 0, 0, 100)
  GRAPH_BORDER_INNER = rl.Color(255, 255, 255, 20)

  # Graph data lines
  DESIRED_GREEN = rl.Color(0, 255, 0, 200)
  ACTUAL_YELLOW = rl.Color(255, 255, 0, 200)
  DESIRED_CURV_CYAN = rl.Color(0, 255, 255, 200)
  ACTUAL_CURV_MAGENTA = rl.Color(255, 100, 255, 200)
  TRAJECTORY_ORANGE = rl.Color(255, 150, 0, 200)
  GAS_GREEN = rl.Color(0, 255, 0, 200)
  BRAKE_RED = rl.Color(255, 0, 0, 200)

  # Grid and scale
  GRID_LINE = rl.Color(189, 195, 199, 80)
  ZERO_LINE = rl.Color(236, 240, 241, 255)
  ZERO_LINE_DIM = rl.Color(236, 240, 241, 100)
  SCALE_TEXT = rl.Color(236, 240, 241, 230)
  TIME_MARKER_PRIMARY = rl.Color(180, 180, 180, 140)
  TIME_MARKER_SECONDARY = rl.Color(160, 160, 160, 110)

  # Tab styling
  TAB_ACTIVE = rl.Color(33, 150, 243, 230)
  TAB_INACTIVE = rl.Color(54, 54, 54, 230)
  TAB_BORDER = rl.Color(60, 60, 60, 150)
  TAB_TEXT = rl.Color(255, 255, 255, 255)
  TAB_TEXT_DIM = rl.Color(180, 180, 180, 255)

  # Close button
  CLOSE_BG = rl.Color(60, 60, 60, 230)
  CLOSE_BORDER = rl.Color(170, 170, 170, 255)

  # Card accents (for OtherDebugPanel groups)
  ACCENT_DYNAMICS = rl.Color(33, 150, 243, 255)
  ACCENT_STEERING = rl.Color(100, 149, 237, 255)
  ACCENT_PEDALS = rl.Color(46, 204, 113, 255)
  ACCENT_SYSTEMS = rl.Color(241, 196, 15, 255)
  ACCENT_CRUISE = rl.Color(155, 89, 182, 255)
  ACCENT_SAFETY = rl.Color(231, 76, 60, 255)
  ACCENT_RADAR = rl.Color(52, 152, 219, 255)
  ACCENT_TUNING = rl.Color(230, 126, 34, 255)
  ACCENT_FIRMWARE = rl.Color(26, 188, 156, 255)
  ACCENT_DEVICE = rl.Color(149, 165, 166, 255)

  # Card separator
  SEPARATOR = rl.Color(255, 255, 255, 30)

  # Legend
  LEGEND_TEXT = rl.Color(255, 255, 255, 255)
  LEGEND_LABEL = rl.Color(180, 180, 180, 255)
