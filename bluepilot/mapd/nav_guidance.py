"""Next-maneuver guidance from Amap route steps + current GCJ-02 position."""

from __future__ import annotations

import math
from typing import Any


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
  r = 6371000.0
  p1, p2 = math.radians(lat1), math.radians(lat2)
  dphi = math.radians(lat2 - lat1)
  dlmb = math.radians(lon2 - lon1)
  a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
  return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def _as_text(val: Any) -> str:
  if val is None:
    return ""
  if isinstance(val, list):
    return ""
  return str(val).strip()


def classify_action(action: str, assistant: str = "", instruction: str = "") -> str | None:
  """Return icon key: left|right|uturn|arrive|slight_left|slight_right, or None if not a turn cue."""
  blob = f"{action} {assistant} {instruction}"
  if any(k in blob for k in ("到达目的地", "到达终点", "到达途经")):
    return "arrive"
  if "掉头" in blob:
    return "uturn"
  if any(k in action for k in ("向左前方", "靠左")) or "向左前方" in blob:
    return "slight_left"
  if any(k in action for k in ("向右前方", "靠右")) or "向右前方" in blob:
    return "slight_right"
  if "左转" in action or (not action and "左转" in instruction):
    return "left"
  if "右转" in action or (not action and "右转" in instruction):
    return "right"
  return None


def action_label(icon: str, action: str) -> str:
  if action:
    return action
  return {
    "left": "左转",
    "right": "右转",
    "slight_left": "向左前方",
    "slight_right": "向右前方",
    "uturn": "掉头",
    "arrive": "到达",
  }.get(icon, "")


def format_distance_m(distance_m: float) -> str:
  d = max(0.0, float(distance_m))
  if d < 1000:
    return f"{int(round(d))}米"
  return f"{d / 1000.0:.1f}公里"


def _step_coords(step: dict[str, Any]) -> list[tuple[float, float]]:
  coords = step.get("coordinates") or []
  out: list[tuple[float, float]] = []
  for pt in coords:
    try:
      out.append((float(pt["latitude"]), float(pt["longitude"])))
    except (KeyError, TypeError, ValueError):
      continue
  return out


def compute_next_guidance(route: dict[str, Any] | None, lat: float, lon: float) -> dict[str, Any] | None:
  """Pick the next turn after the closest point on the route to (lat, lon) GCJ-02."""
  if not route:
    return None
  steps = route.get("steps") or []
  if not steps:
    return None

  # Build cumulative path: list of (lat, lon, step_idx, dist_from_start)
  path: list[tuple[float, float, int, float]] = []
  cum = 0.0
  for si, step in enumerate(steps):
    pts = _step_coords(step)
    if not pts:
      # Fall back to distance-only steps without geometry
      continue
    for i, (plat, plon) in enumerate(pts):
      if path:
        plat0, plon0, _, _ = path[-1]
        cum += _haversine_m(plat0, plon0, plat, plon)
      path.append((plat, plon, si, cum))

  if len(path) < 2:
    # No polyline detail: use remaining step distances from the first turn action
    remain = 0.0
    for step in steps:
      action = _as_text(step.get("action"))
      assistant = _as_text(step.get("assistant_action"))
      instruction = _as_text(step.get("instruction"))
      icon = classify_action(action, assistant, instruction)
      dist = float(step.get("distance_m") or 0)
      if icon:
        return {
          "icon": icon,
          "action": action_label(icon, action),
          "distance_m": remain + dist * 0.5,
          "distance_text": format_distance_m(remain + dist * 0.5),
          "road": _as_text(step.get("road")),
          "instruction": instruction,
        }
      remain += dist
    return None

  # Closest point on path
  best_i = 0
  best_d = 1e18
  for i, (plat, plon, _, _) in enumerate(path):
    d = _haversine_m(lat, lon, plat, plon)
    if d < best_d:
      best_d = d
      best_i = i

  # If far from route, still show next turn but mark off-route lightly via large distance
  cur_dist = path[best_i][3]
  cur_step = path[best_i][2]

  # Find next maneuver at or after current step
  for si in range(cur_step, len(steps)):
    step = steps[si]
    action = _as_text(step.get("action"))
    assistant = _as_text(step.get("assistant_action"))
    instruction = _as_text(step.get("instruction"))
    icon = classify_action(action, assistant, instruction)
    if not icon:
      continue
    # Maneuver distance: along-route distance to end of this step (typical turn point)
    end_dist = cur_dist
    for plat, plon, sidx, cdist in path:
      if sidx == si:
        end_dist = cdist
    remain = max(0.0, end_dist - cur_dist)
    # If we're already past most of this step, look further
    step_len = float(step.get("distance_m") or 0)
    if si == cur_step and step_len > 0 and remain < min(25.0, step_len * 0.15):
      continue
    return {
      "icon": icon,
      "action": action_label(icon, action),
      "distance_m": remain,
      "distance_text": format_distance_m(remain),
      "road": _as_text(step.get("road")),
      "instruction": instruction,
      "off_route": best_d > 80.0,
    }

  # Destination fallback
  dest = route.get("destination") or {}
  try:
    dlat, dlon = float(dest["latitude"]), float(dest["longitude"])
    remain = _haversine_m(lat, lon, dlat, dlon)
    return {
      "icon": "arrive",
      "action": "到达",
      "distance_m": remain,
      "distance_text": format_distance_m(remain),
      "road": _as_text(dest.get("place_name")),
      "instruction": "",
      "off_route": best_d > 80.0,
    }
  except (KeyError, TypeError, ValueError):
    return None
