#!/usr/bin/env python3
"""Convert sunnypilot settings_ui.json into BluePilot web panel JSON."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_UI_PATH = REPO_ROOT / "sunnypilot" / "sunnylink" / "settings_ui.json"
PANEL_DIR = REPO_ROOT / "selfdrive" / "ui" / "bluepilot" / "menus"

# settings_ui panel id -> bp_*_panel.json stem (matches bp_portal panel_order)
PANEL_ID_MAP = {
  "device": "bp_device_panel",
  "toggles": "bp_toggles_panel",
  "steering": "bp_steering_panel",
  "cruise": "bp_cruise_panel",
  "visuals": "bp_visuals_panel",
  "display": "bp_display_panel",
  "models": "bp_models_panel",
  "developer": "bp_developer_panel",
}

PANEL_ORDER = [
  "bp_device_panel",
  "bp_toggles_panel",
  "bp_steering_panel",
  "bp_cruise_panel",
  "bp_visuals_panel",
  "bp_display_panel",
  "bp_models_panel",
  "bp_vehicle_panel",
  "bp_developer_panel",
]

ICON_BY_PANEL = {
  "bp_device_panel": "devices",
  "bp_toggles_panel": "toggle_on",
  "bp_steering_panel": "trip_origin",
  "bp_cruise_panel": "speed",
  "bp_visuals_panel": "visibility",
  "bp_display_panel": "monitor",
  "bp_models_panel": "model_training",
  "bp_vehicle_panel": "directions_car",
  "bp_developer_panel": "code",
}

STRING_PARAM_ACTIONS: dict[str, str] = {}


def _resolve_unit(unit: Any) -> str | None:
  if unit is None:
    return None
  if isinstance(unit, str):
    return unit
  if isinstance(unit, dict):
    return unit.get("metric") or unit.get("imperial")
  return None


def _convert_rule(rule: dict[str, Any]) -> dict[str, Any] | None:
  if not isinstance(rule, dict):
    return None

  rule_type = rule.get("type")
  if rule_type == "offroad_only":
    return {"isOffroad": True}
  if rule_type == "not_engaged":
    return {"isEngaged": False}
  if rule_type == "param":
    key = rule.get("key")
    if key is None:
      return None
    if rule.get("equals") is True:
      return {"paramIsTrue": key}
    if rule.get("equals") is False:
      return {"paramIsFalse": key}
    if "equals" in rule:
      return {"paramValueEquals": {key: rule["equals"]}}
  if rule_type == "param_compare":
    key = rule.get("key")
    op = rule.get("op")
    value = rule.get("value")
    if key is None or op is None:
      return None
    if op == ">":
      return {"paramValueGreaterThan": {key: value}}
    if op == "<":
      return {"paramValueLessThan": {key: value}}
  if rule_type == "capability":
    field = rule.get("field")
    equals = rule.get("equals")
    if field is not None:
      return {"capabilityEquals": {"field": field, "equals": equals}}
  if rule_type == "not":
    inner = _convert_rule(rule.get("condition", {}))
    return {"notCondition": inner} if inner else None
  if rule_type == "any":
    conditions = [_convert_rule(c) for c in rule.get("conditions", [])]
    conditions = [c for c in conditions if c]
    return {"anyConditionsTrue": conditions} if conditions else None
  if rule_type == "all":
    conditions = [_convert_rule(c) for c in rule.get("conditions", [])]
    conditions = [c for c in conditions if c]
    return {"allConditionsTrue": conditions} if conditions else None
  return None


def _convert_rules(rules: list[dict[str, Any]] | None) -> dict[str, Any] | None:
  if not rules:
    return None
  converted = [_convert_rule(r) for r in rules]
  converted = [r for r in converted if r]
  if not converted:
    return None
  if len(converted) == 1:
    return converted[0]
  return {"allConditionsTrue": converted}


def _option_control_type(item: dict[str, Any]) -> str:
  step = item.get("step", 1)
  if isinstance(step, float) and step != int(step):
    return "float"
  for key in ("min", "max", "step"):
    val = item.get(key)
    if isinstance(val, float) and val != int(val):
      return "float"
  return "integer"


def _convert_item(item: dict[str, Any], group_prefix: str, index: int) -> list[dict[str, Any]]:
  key = item.get("key")
  widget = item.get("widget")
  if not key or not widget:
    return []

  title = item.get("title") or key
  desc = item.get("description") or ""
  controls: list[dict[str, Any]] = []

  if widget == "toggle":
    control: dict[str, Any] = {
      "type": "toggle",
      "param": key,
      "title": title,
      "desc": desc,
    }
    if item.get("needs_onroad_cycle"):
      control["needsOnroadCycle"] = True
    enablement = _convert_rules(item.get("enablement"))
    if enablement:
      control["enableConditions"] = enablement
    visibility = _convert_rules(item.get("visibility"))
    if visibility:
      control["visibleConditions"] = visibility
    controls.append(control)

  elif widget == "option":
    ctype = _option_control_type(item)
    control = {
      "type": ctype,
      "param": key,
      "title": title,
      "desc": desc,
      "min": item.get("min", 0),
      "max": item.get("max", 100),
      "increment": item.get("step", 1),
    }
    unit = _resolve_unit(item.get("unit"))
    if unit:
      control["unit"] = unit
    enablement = _convert_rules(item.get("enablement"))
    if enablement:
      control["enableConditions"] = enablement
      control["visibleConditions"] = enablement
    controls.append(control)

  elif widget == "multiple_button":
    options = []
    for opt in item.get("options", []):
      option: dict[str, Any] = {
        "name": opt.get("label", str(opt.get("value"))),
        "value": opt.get("value"),
      }
      opt_enablement = _convert_rules(opt.get("enablement"))
      if opt_enablement:
        option["enableConditions"] = opt_enablement
      options.append(option)
    control = {
      "type": "segmented_control",
      "param": key,
      "title": title,
      "desc": desc,
      "options": options,
    }
    enablement = _convert_rules(item.get("enablement"))
    if enablement:
      control["enableConditions"] = enablement
    controls.append(control)

  elif widget == "button" or key in STRING_PARAM_ACTIONS:
    action = item.get("action") or STRING_PARAM_ACTIONS.get(key)
    control = {
      "type": "command_button",
      "param": key,
      "title": title,
      "desc": desc,
      "button_text": "EDIT",
      "action": action or "set_param",
    }
    controls.append(control)

  else:
    logger.debug("Skipping unsupported settings_ui widget %s for key %s", widget, key)

  for sub_index, sub_item in enumerate(item.get("sub_items") or []):
    sub_controls = _convert_item(sub_item, group_prefix, sub_index)
    parent_enablement = _convert_rules([{"type": "param", "key": key, "equals": True}])
    for sub_control in sub_controls:
      if parent_enablement:
        sub_control["visibleConditions"] = parent_enablement
    controls.extend(sub_controls)

  return controls


def _convert_section(section: dict[str, Any]) -> dict[str, Any] | None:
  section_id = section.get("id") or "section"
  items = section.get("items") or []
  controls: list[dict[str, Any]] = []

  for index, item in enumerate(items):
    controls.extend(_convert_item(item, section_id, index))

  for sub_panel in section.get("sub_panels") or []:
    for index, item in enumerate(sub_panel.get("items") or []):
      controls.extend(_convert_item(item, sub_panel.get("id") or section_id, index))

  if not controls:
    return None

  return {
    "groupName": section_id,
    "title": section.get("title") or section_id,
    "controls": controls,
  }


def _convert_panel(panel: dict[str, Any]) -> dict[str, Any] | None:
  panel_id = panel.get("id")
  bp_id = PANEL_ID_MAP.get(panel_id)
  if not bp_id:
    return None

  groups = []
  for section in panel.get("sections") or []:
    group = _convert_section(section)
    if group:
      groups.append(group)

  if not groups:
    return None

  return {
    "menuName": panel.get("label") or panel_id,
    "menuDescription": panel.get("description") or "",
    "menuIcon": panel.get("icon") or ICON_BY_PANEL.get(bp_id, ""),
    "groups": groups,
  }


@lru_cache(maxsize=1)
def get_onroad_cycle_params() -> frozenset[str]:
  """Return param keys that require an onroad cycle after modification."""
  data = _load_settings_ui()
  params: set[str] = set()

  def walk_items(items: list[dict[str, Any]] | None) -> None:
    if not items:
      return
    for item in items:
      if not isinstance(item, dict):
        continue
      key = item.get("key")
      if key and item.get("needs_onroad_cycle"):
        params.add(str(key))
      walk_items(item.get("sub_items"))

  for panel in data.get("panels", []):
    for section in panel.get("sections", []):
      walk_items(section.get("items"))
      for sub_panel in section.get("sub_panels", []):
        walk_items(sub_panel.get("items"))

  vehicle = data.get("vehicle_settings") or {}
  for section in vehicle.get("sections", []):
    walk_items(section.get("items"))

  return frozenset(params)


@lru_cache(maxsize=1)
def _load_settings_ui() -> dict[str, Any]:
  if not SETTINGS_UI_PATH.exists():
    return {}
  return json.loads(SETTINGS_UI_PATH.read_text(encoding="utf-8"))


def list_panels() -> list[dict[str, str]]:
  """Return panel metadata for /api/panels."""
  data = _load_settings_ui()
  panels_by_id: dict[str, dict[str, str]] = {}

  for panel in data.get("panels", []):
    ui_id = panel.get("id")
    bp_id = PANEL_ID_MAP.get(ui_id)
    if not bp_id:
      continue
    panels_by_id[bp_id] = {
      "id": bp_id,
      "name": panel.get("label") or bp_id,
      "description": panel.get("description") or "",
      "icon": panel.get("icon") or ICON_BY_PANEL.get(bp_id, ""),
    }

  # Include any hand-authored JSON panels not generated from settings_ui.
  if PANEL_DIR.exists():
    for json_file in sorted(PANEL_DIR.glob("bp_*_panel.json")):
      bp_id = json_file.stem
      if bp_id in panels_by_id:
        continue
      try:
        panel_data = json.loads(json_file.read_text(encoding="utf-8"))
        panels_by_id[bp_id] = {
          "id": bp_id,
          "name": panel_data.get("menuName", bp_id),
          "description": panel_data.get("menuDescription", ""),
          "icon": panel_data.get("menuIcon", ICON_BY_PANEL.get(bp_id, "")),
        }
      except Exception:
        logger.warning("Failed to read panel metadata from %s", json_file)

  panels = []
  for bp_id in PANEL_ORDER:
    if bp_id in panels_by_id:
      panels.append(panels_by_id[bp_id])
  for bp_id, info in panels_by_id.items():
    if bp_id not in PANEL_ORDER:
      panels.append(info)
  return panels


def get_panel(panel_id: str) -> dict[str, Any] | None:
  """Return full panel config for /api/panels/{id}."""
  json_path = PANEL_DIR / f"{panel_id}.json"
  if json_path.exists():
    try:
      return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
      logger.exception("Failed to load panel file %s", json_path)

  ui_id = {v: k for k, v in PANEL_ID_MAP.items()}.get(panel_id)
  if not ui_id:
    return None

  data = _load_settings_ui()
  for panel in data.get("panels", []):
    if panel.get("id") == ui_id:
      return _convert_panel(panel)
  return None
