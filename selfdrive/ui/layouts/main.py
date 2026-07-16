import pyray as rl
from enum import IntEnum
import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import UnknownKeyName
from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.selfdrive.ui.layouts.settings.settings import SettingsLayout, PanelType
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView, draw_onroad_border
from openpilot.selfdrive.ui.ui_state import device, ui_state
from openpilot.selfdrive.ui.layouts.onboarding import OnboardingWindow
from openpilot.selfdrive.ui.body.layouts.onroad import BodyLayout
from bluepilot.ui.widgets.install_status import InstallStatusTracker, draw_install_banner, banner_height

if gui_app.sunnypilot_ui():
  from openpilot.selfdrive.ui.sunnypilot.layouts.settings.settings import SettingsLayoutSP as SettingsLayout
  from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import DeveloperUiRenderer


class MainState(IntEnum):
  HOME = 0
  SETTINGS = 1
  ONROAD = 2


class MainLayout(Widget):
  def __init__(self):
    super().__init__()

    self._pm = messaging.PubMaster(['bookmarkButton'])

    self._sidebar = Sidebar()
    self._install_status = InstallStatusTracker()
    self._current_mode = MainState.HOME
    self._prev_onroad = False

    # Initialize layouts
    self._home_layout = HomeLayout()
    self._home_body_layout = BodyLayout()
    self._road_view = AugmentedRoadView()
    self._layouts = {MainState.HOME: self._home_layout, MainState.SETTINGS: SettingsLayout(), MainState.ONROAD: self._road_view}

    self._sidebar_rect = rl.Rectangle(0, 0, 0, 0)
    self._content_rect = rl.Rectangle(0, 0, 0, 0)
    self._banner_height = 0.0

    # Split-screen state
    self._split_screen = False
    self._amap_view = None
    self._bottom_dev_ui = DeveloperUiRenderer() if gui_app.sunnypilot_ui() else None

    # Set callbacks
    self._setup_callbacks()

    gui_app.push_widget(self)

    # Start onboarding if terms or training not completed, make sure to push after self
    self._onboarding_window = OnboardingWindow()
    if not self._onboarding_window.completed:
      gui_app.push_widget(self._onboarding_window)

  def _render(self, _):
    self._handle_onroad_transition()
    self._render_main_content()

  def _setup_callbacks(self):
    self._sidebar.set_callbacks(on_settings=self._on_settings_clicked,
                                on_flag=self._on_bookmark_clicked,
                                open_settings=lambda: self.open_settings(PanelType.TOGGLES))
    self._layouts[MainState.HOME]._setup_widget.set_open_settings_callback(lambda: self.open_settings(PanelType.FIREHOSE))
    self._layouts[MainState.HOME].set_settings_callback(lambda: self.open_settings(PanelType.TOGGLES))
    self._layouts[MainState.SETTINGS].set_callbacks(on_close=self._set_mode_for_state)

    for layout in (self._road_view, self._home_body_layout):
      layout.set_click_callback(self._on_onroad_clicked)

    device.add_interactive_timeout_callback(self._set_mode_for_state)
    ui_state.add_on_body_changed_callbacks(self._on_body_changed)

  def _update_layout_rects(self):
    base_y = self._rect.y + self._banner_height
    base_h = max(0.0, self._rect.height - self._banner_height)

    self._sidebar_rect = rl.Rectangle(self._rect.x, base_y, SIDEBAR_WIDTH, base_h)

    x_offset = SIDEBAR_WIDTH if self._sidebar.is_visible else 0
    self._content_rect = rl.Rectangle(self._rect.x + x_offset, base_y, self._rect.width - x_offset, base_h)

  def _handle_onroad_transition(self):
    if ui_state.started != self._prev_onroad:
      self._prev_onroad = ui_state.started

      self._set_mode_for_state()

  def _set_mode_for_state(self):
    # Don't go onroad if body, home is onroad
    if ui_state.is_body:
      self._set_current_layout(MainState.HOME)
      self._sidebar.set_visible(not ui_state.ignition)
      return

    if ui_state.started:
      # Don't hide sidebar from interactive timeout
      if self._current_mode != MainState.ONROAD:
        self._sidebar.set_visible(False)
      self._set_current_layout(MainState.ONROAD)
    else:
      self._set_current_layout(MainState.HOME)
      self._sidebar.set_visible(True)

  def _set_current_layout(self, layout: MainState):
    if layout != self._current_mode:
      self._layouts[self._current_mode].hide_event()
      self._current_mode = layout
      self._layouts[self._current_mode].show_event()
      if self._road_view is not None:
        self._road_view.set_draw_border(True)
    # Exit split-screen / stop map worker request when leaving onroad
    if layout != MainState.ONROAD:
      if self._split_screen:
        self._split_screen = False
      if self._amap_view is not None:
        self._amap_view.set_enabled(False)

  def open_settings(self, panel_type: PanelType):
    self._layouts[MainState.SETTINGS].set_current_panel(panel_type)
    self._set_current_layout(MainState.SETTINGS)
    self._sidebar.set_visible(False)

  def _on_settings_clicked(self):
    self.open_settings(PanelType.DEVICE)

  def _on_bookmark_clicked(self):
    user_bookmark = messaging.new_message('bookmarkButton')
    user_bookmark.valid = True
    self._pm.send('bookmarkButton', user_bookmark)

  def _on_onroad_clicked(self):
    if self._split_screen:
      # Exit split-screen → show sidebar
      self._split_screen = False
      self._sidebar.set_visible(True)
    elif not self._sidebar.is_visible:
      # Sidebar hidden → enter split-screen when map is ready, otherwise show sidebar
      if self._can_enter_split_screen():
        self._split_screen = True
      else:
        self._sidebar.set_visible(True)
    else:
      # Sidebar visible → hide sidebar
      self._sidebar.set_visible(False)

  def _has_amap_prerequisites(self) -> bool:
    """Check WiFi and JS API 2.0 credentials — map feature available but may still be loading."""
    sm = ui_state.sm
    is_wifi = (sm.valid.get('deviceState') and
               sm['deviceState'].networkType == log.DeviceState.NetworkType.wifi)
    try:
      has_key = bool(ui_state.params.get("AmapApiKey"))
      has_security = bool(ui_state.params.get("AmapSecurityJsCode"))
    except UnknownKeyName:
      return False
    return is_wifi and has_key and has_security

  def _ensure_amap_view(self) -> None:
    if self._amap_view is None:
      from bluepilot.ui.onroad.amap_view import AmapView
      self._amap_view = AmapView()

  def _prepare_amap(self) -> None:
    if not self._has_amap_prerequisites():
      if self._amap_view is not None:
        self._amap_view.set_enabled(False)
      return
    self._ensure_amap_view()
    self._amap_view.prepare()

  def _can_enter_split_screen(self) -> bool:
    """Map must be fully prepared (amapd shared frame ready) before split-screen."""
    return self._has_amap_prerequisites() and self._amap_view is not None and self._amap_view.is_ready()

  def _on_body_changed(self):
    self._layouts[MainState.HOME] = self._home_body_layout if ui_state.is_body else self._home_layout
    self._set_mode_for_state()

  def _render_main_content(self):
    self._install_status.update()
    self._banner_height = banner_height(self._install_status)
    self._update_layout_rects()

    if self._banner_height > 0:
      draw_install_banner(
        rl.Rectangle(self._rect.x, self._rect.y, self._rect.width, self._banner_height),
        self._install_status,
      )

    if self._sidebar.is_visible:
      self._sidebar.render(self._sidebar_rect)

    if self._sidebar.is_visible:
      content_rect = self._content_rect
    else:
      content_rect = rl.Rectangle(
        self._rect.x,
        self._rect.y + self._banner_height,
        self._rect.width,
        max(0.0, self._rect.height - self._banner_height),
      )

    if self._current_mode == MainState.ONROAD:
      self._prepare_amap()

    if self._current_mode == MainState.ONROAD and self._split_screen:
      self._render_split_screen(content_rect)
    else:
      if self._current_mode == MainState.ONROAD:
        self._road_view.set_draw_border(True)
      self._layouts[self._current_mode].render(content_rect)

    if self._current_mode == MainState.ONROAD:
      self._render_bottom_dev_ui(content_rect)

  def _render_bottom_dev_ui(self, rect: rl.Rectangle) -> None:
    if self._bottom_dev_ui is None:
      return

    from openpilot.selfdrive.ui import UI_BORDER_SIZE

    inner_rect = rl.Rectangle(
      rect.x + UI_BORDER_SIZE,
      rect.y + UI_BORDER_SIZE,
      rect.width - 2 * UI_BORDER_SIZE,
      rect.height - 2 * UI_BORDER_SIZE,
    )
    self._bottom_dev_ui.render_bottom(inner_rect)

  def _render_split_screen(self, rect: rl.Rectangle):
    """Render split-screen: driving view (left) + Amap (right), with unified border."""
    from openpilot.selfdrive.ui import UI_BORDER_SIZE

    # Draw unified border around the full area
    draw_onroad_border(rect)

    # Inner content area (inside the colored border)
    inner_x = rect.x + UI_BORDER_SIZE
    inner_y = rect.y + UI_BORDER_SIZE
    inner_w = rect.width - 2 * UI_BORDER_SIZE
    inner_h = rect.height - 2 * UI_BORDER_SIZE

    half_w = int(inner_w / 2)

    left_rect = rl.Rectangle(inner_x, inner_y, half_w, inner_h)
    right_rect = rl.Rectangle(inner_x + half_w, inner_y, inner_w - half_w, inner_h)

    # Left: driving view without its own border
    self._road_view.set_draw_border(False)
    self._road_view.render(left_rect)

    # Right: Amap map view
    self._ensure_amap_view()
    self._amap_view.render(right_rect)

    # Divider between driving view and map
    divider_x = int(inner_x + half_w)
    rl.draw_line(divider_x, int(inner_y), divider_x, int(inner_y + inner_h), rl.Color(0, 0, 0, 220))
    rl.draw_line(divider_x + 1, int(inner_y), divider_x + 1, int(inner_y + inner_h), rl.Color(90, 90, 90, 180))
