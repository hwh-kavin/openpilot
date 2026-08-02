from enum import IntEnum

import cereal.messaging as messaging
import pyray as rl

from openpilot.system.ui.lib.application import gui_app
from openpilot.system.ui.widgets import Widget
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar, SIDEBAR_WIDTH
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.selfdrive.ui.layouts.settings.settings import SettingsLayout, PanelType
from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
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

    # Split-screen state (CarLife companion map mirror)
    self._split_screen = False
    self._carlife_map_view = None
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
    # Exit split-screen / stop map viewer when leaving onroad
    if layout != MainState.ONROAD:
      if self._split_screen:
        self._split_screen = False
      if self._carlife_map_view is not None:
        self._carlife_map_view.set_enabled(False)

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

  def _carlife_mirror_enabled(self) -> bool:
    return ui_state.params.get_bool("CarLifeMapMirrorEnabled")

  def _on_onroad_clicked(self):
    if self._split_screen:
      # Exit split-screen → show sidebar
      self._split_screen = False
      self._sidebar.set_visible(True)
      if self._carlife_map_view is not None:
        self._carlife_map_view.set_enabled(False)
    elif not self._sidebar.is_visible:
      # Sidebar hidden → enter CarLife map split only when mirror is enabled.
      if self._carlife_mirror_enabled():
        self._ensure_carlife_map_view()
        self._carlife_map_view.prepare()
        self._split_screen = True
      else:
        self._sidebar.set_visible(True)
    else:
      # Sidebar visible → hide sidebar
      self._sidebar.set_visible(False)

  def _ensure_carlife_map_view(self) -> None:
    if self._carlife_map_view is None:
      from bluepilot.ui.onroad.carlife_map_view import CarLifeMapView
      self._carlife_map_view = CarLifeMapView()

  def _prepare_carlife_map(self) -> None:
    # Warm shm only when mirror is enabled.
    if not self._carlife_mirror_enabled():
      if self._split_screen:
        self._split_screen = False
      if self._carlife_map_view is not None:
        self._carlife_map_view.set_enabled(False)
      return
    self._ensure_carlife_map_view()
    self._carlife_map_view.prepare()

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
      self._prepare_carlife_map()

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
    """Waiting: 50/50. Ready: map height-fit (keep aspect), leftover width → driving."""
    ox, oy = float(rect.x), float(rect.y)
    iw, ih = float(rect.width), float(rect.height)
    if iw <= 1 or ih <= 1:
      return

    self._ensure_carlife_map_view()
    self._carlife_map_view.prepare()

    # Both panes always span full content height so the join has no vertical black bars.
    map_h = ih
    map_y = oy
    min_drive_w = 360.0
    max_map_w = max(1.0, iw - min_drive_w)

    if self._carlife_map_view.is_ready():
      # Height-fit phone frame aspect → map width; leftover width goes to driving.
      frame = self._carlife_map_view.frame_size()
      if frame is not None:
        fw, fh = float(frame[0]), float(frame[1])
        map_w = min(max_map_w, ih * (fw / fh)) if fh > 0 else max_map_w * 0.5
      else:
        map_w, _ = self._carlife_map_view.displayed_size(max_map_w, ih)
      map_w = max(1.0, min(max_map_w, map_w))
    else:
      # Waiting placeholder: fixed half / half split.
      map_w = iw * 0.5

    map_x = ox + iw - map_w
    drive_w = map_x - ox

    left_rect = rl.Rectangle(ox, oy, drive_w, ih)
    right_rect = rl.Rectangle(map_x, map_y, map_w, map_h)

    self._road_view.set_draw_border(False)
    self._road_view.render(left_rect)

    try:
      # Pane is aspect-correct (or half-width while waiting) and full height — fill edge-to-edge.
      self._carlife_map_view.render(right_rect)
    except Exception:
      pass

    divider_x = int(round(map_x))
    rl.draw_line(divider_x, int(oy), divider_x, int(oy + ih), rl.Color(0, 0, 0, 220))
    rl.draw_line(divider_x + 1, int(oy), divider_x + 1, int(oy + ih), rl.Color(90, 90, 90, 180))
