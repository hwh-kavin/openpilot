"""
BluePilot Recent Changes Dialog
Fullscreen modal overlay showing categorized changes from BP_CHANGES.json.
Auto-shown when BP version changes (after update). Ported from Qt BPRecentChangesDialog.
"""

import json
import os
import pyray as rl

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.ui.lib.application import gui_app, FontWeight, FONT_SCALE
from openpilot.system.ui.lib.scroll_panel import GuiScrollPanel
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.lib.application import MousePos
from bluepilot.ui.lib.colors import BPColors

# Layout constants (matching Qt)
HEADER_HEIGHT = 130
CONTENT_MARGIN = 40
SECTION_SPACING = 30
FRAME_PADDING = 15
FRAME_RADIUS = 15
BADGE_RADIUS = 20
BADGE_PAD_X = 24
BADGE_PAD_Y = 12

# Font sizes
TITLE_FONT_SIZE = 42
CATEGORY_TITLE_SIZE = 34
ITEM_FONT_SIZE = 30
BADGE_FONT_SIZE = 38
AGNOS_FONT_SIZE = 32
CLOSE_FONT_SIZE = 36

# Category definitions: (json_key, display_title, color)
CATEGORIES = [
  ("changes", "New Features & Changes", BPColors.CHANGES_GREEN),
  ("fixes", "Bug Fixes", BPColors.CHANGES_BLUE),
  ("removals", "Removed Features", BPColors.CHANGES_ORANGE),
  ("known_issues", "Known Issues", BPColors.CHANGES_RED),
]

_BP_ROOT = os.path.join(os.path.dirname(__file__), '../../..')

CLOSE_CIRCLE_RADIUS = 32
CLOSE_X_SIZE = 14
CLOSE_X_WIDTH = 4


class _CloseButton(Widget):
  """Close button: white filled circle with a dark X cut-out."""

  def __init__(self, callback):
    super().__init__()
    self._callback = callback

  def _handle_mouse_release(self, mouse_pos: MousePos) -> bool:
    self._callback()
    return True

  def _render(self, rect: rl.Rectangle):
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)
    circle_color = rl.Color(180, 180, 180, 255) if self.is_pressed else rl.WHITE

    # White filled circle
    rl.draw_circle(cx, cy, CLOSE_CIRCLE_RADIUS, circle_color)

    # Dark X lines on top (matches header bg so it looks transparent/cut-out)
    half = CLOSE_X_SIZE
    x_color = BPColors.CHANGES_HEADER_BG
    rl.draw_line_ex(rl.Vector2(cx - half, cy - half), rl.Vector2(cx + half, cy + half), CLOSE_X_WIDTH, x_color)
    rl.draw_line_ex(rl.Vector2(cx + half, cy - half), rl.Vector2(cx - half, cy + half), CLOSE_X_WIDTH, x_color)


def _read_bp_version() -> str:
  try:
    with open(os.path.join(_BP_ROOT, 'BPVERSION')) as f:
      return f.readline().strip()
  except Exception:
    return ""


def _load_changes_json() -> dict:
  try:
    with open(os.path.join(_BP_ROOT, 'BP_CHANGES.json')) as f:
      return json.load(f)
  except Exception as e:
    cloudlog.error(f"Failed to load BP_CHANGES.json: {e}")
    return {}


class RecentChangesDialog(Widget):
  """Fullscreen modal overlay displaying recent changes for a given BP version."""

  def __init__(self, version: str, dismiss_callback=None):
    super().__init__()
    self._version = version
    self._dismiss_callback = dismiss_callback
    self._result = DialogResult.NO_ACTION
    self._scroll_panel = GuiScrollPanel()
    self._close_btn = _CloseButton(self._on_close)

    # Load version data
    data = _load_changes_json()
    versions = data.get("versions", {})
    self._version_data = versions.get(version, {})
    self._agnos_version = self._version_data.get("agnos_version", "")

    # Pre-build section data: list of (title, color, items)
    self._sections: list[tuple[str, rl.Color, list[str]]] = []
    for key, title, color in CATEGORIES:
      items = self._version_data.get(key, [])
      if items:
        self._sections.append((title, color, items))

    # Cached content height (computed on first render with known width)
    self._content_height: float = 0
    self._cached_width: int = 0

  def _on_close(self):
    self._result = DialogResult.CONFIRM
    if self._dismiss_callback:
      self._dismiss_callback(self._result)
    gui_app.pop_widget()

  def show_event(self):
    super().show_event()
    self._scroll_panel.set_offset(0)

  def _compute_content_height(self, content_width: int) -> float:
    """Calculate total scrollable content height."""
    if self._cached_width == content_width and self._content_height > 0:
      return self._content_height

    font = gui_app.font(FontWeight.SEMI_BOLD)
    item_font = gui_app.font(FontWeight.NORMAL)
    line_height = FONT_SCALE
    y = 0.0

    # Version badges row
    y += BADGE_PAD_Y * 2 + BADGE_FONT_SIZE * line_height + SECTION_SPACING

    # Each category section
    for title, _color, items in self._sections:
      # Frame top padding
      y += FRAME_PADDING
      # Category title
      y += CATEGORY_TITLE_SIZE * line_height + 8
      # Items
      usable_width = content_width - 2 * FRAME_PADDING - 28  # bullet + spacing
      for item_text in items:
        lines = wrap_text(item_font, item_text, ITEM_FONT_SIZE, int(usable_width))
        y += max(len(lines), 1) * ITEM_FONT_SIZE * line_height + 4
      # Frame bottom padding + section spacing
      y += FRAME_PADDING + SECTION_SPACING

    self._content_height = y
    self._cached_width = content_width
    return y

  def _render(self, rect: rl.Rectangle):
    # Background
    rl.draw_rectangle_rec(rect, rl.BLACK)

    # Header
    header_rect = rl.Rectangle(rect.x, rect.y, rect.width, HEADER_HEIGHT)
    rl.draw_rectangle_rec(header_rect, BPColors.CHANGES_HEADER_BG)

    # Close button (white circle with X, left side of header)
    close_tap = CLOSE_CIRCLE_RADIUS * 2 + 30  # tap area larger than visual
    close_rect = rl.Rectangle(
      header_rect.x + 15,
      header_rect.y + (HEADER_HEIGHT - close_tap) / 2,
      close_tap, close_tap
    )
    self._close_btn.render(close_rect)

    # Title (centered)
    title_font = gui_app.font(FontWeight.SEMI_BOLD)
    title_text = "Recent Changes"
    title_size = measure_text_cached(title_font, title_text, TITLE_FONT_SIZE)
    title_x = rect.x + (rect.width - title_size.x) / 2
    title_y = header_rect.y + (HEADER_HEIGHT - title_size.y) / 2
    rl.draw_text_ex(title_font, title_text, rl.Vector2(int(title_x), int(title_y)),
                    TITLE_FONT_SIZE, 0, rl.WHITE)

    # Scrollable content area
    content_x = rect.x + CONTENT_MARGIN
    content_y = rect.y + HEADER_HEIGHT
    content_w = rect.width - 2 * CONTENT_MARGIN
    content_h = rect.height - HEADER_HEIGHT
    scroll_rect = rl.Rectangle(content_x, content_y, content_w, content_h)

    total_height = self._compute_content_height(int(content_w))
    scroll_content = rl.Rectangle(content_x, content_y, content_w, total_height)
    scroll_offset = self._scroll_panel.update(scroll_rect, scroll_content)

    rl.begin_scissor_mode(int(content_x), int(content_y), int(content_w), int(content_h))
    self._draw_content(content_x, content_y + scroll_offset + CONTENT_MARGIN, content_w)
    rl.end_scissor_mode()

    return self._result

  def _draw_content(self, x, y, width):
    """Draw all content sections at the given position."""
    cur_y = y
    font_bold = gui_app.font(FontWeight.SEMI_BOLD)
    font_normal = gui_app.font(FontWeight.NORMAL)
    line_height = FONT_SCALE

    # Version badges
    cur_y = self._draw_version_badges(x, cur_y, font_bold)
    cur_y += SECTION_SPACING

    # Category sections
    for title, color, items in self._sections:
      cur_y = self._draw_category_section(x, cur_y, width, title, color, items,
                                          font_bold, font_normal, line_height)
      cur_y += SECTION_SPACING

  def _draw_version_badges(self, x, y, font) -> float:
    """Draw version and AGNOS badges. Returns y after badges."""
    line_height = FONT_SCALE
    badge_h = int(BADGE_FONT_SIZE * line_height + BADGE_PAD_Y * 2)
    cur_x = x

    # Version badge (blue)
    version_text = f"Version {self._version}"
    version_w = int(measure_text_cached(font, version_text, BADGE_FONT_SIZE).x + BADGE_PAD_X * 2)
    version_rect = rl.Rectangle(cur_x, y, version_w, badge_h)
    roundness = BADGE_RADIUS / (badge_h / 2) if badge_h > 0 else 0.5
    rl.draw_rectangle_rounded(version_rect, roundness, 10, BPColors.CHANGES_BLUE)
    text_y = y + (badge_h - BADGE_FONT_SIZE * line_height) / 2
    rl.draw_text_ex(font, version_text, rl.Vector2(int(cur_x + BADGE_PAD_X), int(text_y)),
                    BADGE_FONT_SIZE, 0, rl.WHITE)
    cur_x += version_w + 15

    # AGNOS badge (purple)
    if self._agnos_version:
      agnos_text = f"AGNOS {self._agnos_version}"
      agnos_w = int(measure_text_cached(font, agnos_text, AGNOS_FONT_SIZE).x + 20 * 2)
      agnos_h = int(AGNOS_FONT_SIZE * line_height + 10 * 2)
      agnos_rect = rl.Rectangle(cur_x, y + (badge_h - agnos_h) / 2, agnos_w, agnos_h)
      agnos_roundness = BADGE_RADIUS / (agnos_h / 2) if agnos_h > 0 else 0.5
      rl.draw_rectangle_rounded(agnos_rect, agnos_roundness, 10, BPColors.CHANGES_AGNOS)
      agnos_text_y = y + (badge_h - AGNOS_FONT_SIZE * line_height) / 2
      rl.draw_text_ex(font, agnos_text, rl.Vector2(int(cur_x + 20), int(agnos_text_y)),
                      AGNOS_FONT_SIZE, 0, rl.WHITE)

    return y + badge_h

  def _draw_category_section(self, x, y, width, title, color, items, font_bold, font_normal, line_height) -> float:
    """Draw a category frame with title and bullet items. Returns y after section."""
    # Pre-calculate frame height
    frame_inner_w = width - 2 * FRAME_PADDING
    bullet_offset = 28  # bullet width + spacing
    usable_text_w = int(frame_inner_w - bullet_offset)

    frame_h = FRAME_PADDING  # top padding
    frame_h += CATEGORY_TITLE_SIZE * line_height + 8  # title + gap
    for item_text in items:
      lines = wrap_text(font_normal, item_text, ITEM_FONT_SIZE, usable_text_w)
      frame_h += max(len(lines), 1) * ITEM_FONT_SIZE * line_height + 4
    frame_h += FRAME_PADDING  # bottom padding

    # Draw frame background
    frame_rect = rl.Rectangle(x, y, width, frame_h)
    roundness = FRAME_RADIUS / (min(width, frame_h) / 2) if min(width, frame_h) > 0 else 0.1
    rl.draw_rectangle_rounded(frame_rect, roundness, 10, BPColors.CHANGES_FRAME_BG)

    cur_y = y + FRAME_PADDING

    # Category title
    rl.draw_text_ex(font_bold, title, rl.Vector2(int(x + FRAME_PADDING), int(cur_y)),
                    CATEGORY_TITLE_SIZE, 0, color)
    cur_y += CATEGORY_TITLE_SIZE * line_height + 8

    # Items
    for item_text in items:
      # Bullet
      rl.draw_text_ex(font_bold, "\u2022", rl.Vector2(int(x + FRAME_PADDING), int(cur_y)),
                      ITEM_FONT_SIZE, 0, color)

      # Wrapped text
      lines = wrap_text(font_normal, item_text, ITEM_FONT_SIZE, usable_text_w)
      for line in lines:
        rl.draw_text_ex(font_normal, line,
                        rl.Vector2(int(x + FRAME_PADDING + bullet_offset), int(cur_y)),
                        ITEM_FONT_SIZE, 0, BPColors.CHANGES_TEXT)
        cur_y += ITEM_FONT_SIZE * line_height
      cur_y += 4  # gap between items

    return y + frame_h


class RecentChangesManager:
  """Manages auto-showing the recent changes dialog when BP version changes."""

  def __init__(self):
    self._shown = False
    self._bp_version = _read_bp_version()
    self._params = Params()
    print(f"[RecentChanges] Initialized — BP version: '{self._bp_version}'")

  def should_show(self) -> bool:
    if self._shown:
      print("[RecentChanges] should_show=False (already shown this session)")
      return False
    if not self._bp_version:
      print("[RecentChanges] should_show=False (no BP version found)")
      return False
    stored = self._params.get("BPLastSeenVersion") or ""
    result = self._bp_version != stored
    print(f"[RecentChanges] should_show={result} — current='{self._bp_version}' stored='{stored}' (type={type(stored).__name__})")
    return result

  def show_if_needed(self):
    if not self.should_show():
      return
    # Don't interrupt another overlay (e.g. onboarding) - use nav stack depth
    if hasattr(gui_app, "_nav_stack") and len(gui_app._nav_stack) > 1:
      print("[RecentChanges] Blocked — another overlay is active")
      return
    print(f"[RecentChanges] Showing dialog for version {self._bp_version}")
    dialog = RecentChangesDialog(self._bp_version, dismiss_callback=self._on_dismissed)
    gui_app.push_widget(dialog)
    self._shown = True

  def _on_dismissed(self, _result):
    print(f"[RecentChanges] Dismissed — saving BPLastSeenVersion='{self._bp_version}'")
    self._params.put("BPLastSeenVersion", self._bp_version)
