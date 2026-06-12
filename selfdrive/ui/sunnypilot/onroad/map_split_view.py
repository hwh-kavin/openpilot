import pyray as rl

from openpilot.selfdrive.ui.onroad.augmented_road_view import AugmentedRoadView
from openpilot.selfdrive.ui.sunnypilot.onroad.amap_tile_view import AmapTileView
from openpilot.system.ui.widgets import Widget

MAP_PANEL_RATIO = 0.5


class MapSplitOnroadView(Widget):
  def __init__(self, road_view: AugmentedRoadView, map_view: AmapTileView):
    super().__init__()
    self._road = road_view
    self._map = map_view
    self._road.set_draw_border(False)

  def _render(self, rect: rl.Rectangle):
    road_width = rect.width * (1.0 - MAP_PANEL_RATIO)
    road_rect = rl.Rectangle(rect.x, rect.y, road_width, rect.height)
    map_rect = rl.Rectangle(rect.x + road_width, rect.y, rect.width - road_width, rect.height)

    self._road.render(road_rect)
    self._map.render(map_rect)
