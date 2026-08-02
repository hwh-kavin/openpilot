#!/usr/bin/env python3
"""
Localized parameter labels for BluePilot Portal.

Titles/descriptions are sourced from sunnypilot settings_ui.json (same as C3
settings items) and translated with the same app_zh-CHS.po catalog as the UI.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
SETTINGS_UI_PATH = REPO_ROOT / "sunnypilot" / "sunnylink" / "settings_ui.json"
PARAMS_METADATA_PATH = REPO_ROOT / "sunnypilot" / "sunnylink" / "params_metadata.json"
ZH_PO_PATH = REPO_ROOT / "selfdrive" / "ui" / "translations" / "app_zh-CHS.po"
ZH_DESCRIPTIONS_PATH = Path(__file__).resolve().parent / "param_descriptions_zh-CHS.json"

# settings_ui.json strings that differ from C3 tr() msgids in app_zh-CHS.po
DESCRIPTION_PO_ALIASES: dict[str, str] = {
    "When enabled, pressing the accelerator pedal will disengage longitudinal control.": (
        "When enabled, pressing the accelerator pedal will disengage sunnypilot."
    ),
    "Enable MADS. Disable toggle to revert back to stock sunnypilot engagement/disengagement.": (
        "Enable the beloved MADS feature. "
        "Disable toggle to revert back to stock sunnypilot engagement/disengagement."
    ),
    "Device will automatically shutdown after set time once the engine is turned off. 30h is the default.": (
        "Device will automatically shutdown after set time once the engine is turned off.\n(30h is the default)"
    ),
}

# Native C3 UI strings that differ from settings_ui.json (msgids for gettext).
NATIVE_TITLE_OVERRIDES: dict[str, dict[str, str]] = {
    "Mads": {
        "title": "Modular Assistive Driving System (MADS)",
        "description": (
            "Enable the beloved MADS feature. "
            "Disable toggle to revert back to stock sunnypilot engagement/disengagement."
        ),
        "panel": "Steering",
    },
    "LanguageSetting": {
        "title": "Change Language",
        "description": "",
        "panel": "Device",
    },
    "OpenpilotEnabledToggle": {
        "title": "Enable sunnypilot",
        "description": (
            "Use the sunnypilot system for adaptive cruise control and lane keep driver assistance. "
            "Your attention is required at all times to use this feature."
        ),
        "panel": "Toggles",
    },
    "IsMetric": {
        "title": "Use Metric System",
        "description": "Display speed in km/h instead of mph.",
        "panel": "Toggles",
    },
    "EnableCopyparty": {
        "title": "BluePilot Portal Service",
        "description": (
            "BluePilot Portal is a web-based interface for viewing routes, logs, system metrics, "
            "and device settings. Connect to your comma device locally via its IP address (default port 80)."
        ),
        "panel": "Sunnylink",
    },
    "CarLifeMapMirrorEnabled": {
        "title": "Enable Phone Map Mirror",
        "description": (
            "When on, tap the onroad view (sidebar hidden) to show the phone map split-screen."
        ),
        "panel": "OSM / Map",
    },
}


@lru_cache(maxsize=1)
def _load_po_translations() -> dict[str, str]:
    try:
        from openpilot.system.ui.lib.multilang import load_translations
        translations, _ = load_translations(ZH_PO_PATH)
        return translations
    except Exception:
        logger.exception("Failed to load zh-CHS translations for portal param labels")
        return {}


@lru_cache(maxsize=1)
def _settings_ui_labels() -> dict[str, dict[str, str]]:
  """Map param key -> English title/description/panel from settings_ui.json."""
  labels: dict[str, dict[str, str]] = {}
  if not SETTINGS_UI_PATH.exists():
    return labels

  try:
    data = json.loads(SETTINGS_UI_PATH.read_text(encoding="utf-8"))
  except Exception:
    logger.exception("Failed to read settings_ui.json for param labels")
    return labels

  def walk_items(items: list[dict[str, Any]] | None, panel: str, section: str) -> None:
    if not items:
      return
    for item in items:
      if not isinstance(item, dict):
        continue
      key = item.get("key")
      if key and item.get("widget"):
        labels[str(key)] = {
          "title": str(item.get("title") or ""),
          "description": str(item.get("description") or ""),
          "panel": panel,
          "section": section,
        }
      walk_items(item.get("sub_items"), panel, section)

  for panel in data.get("panels", []):
    panel_label = str(panel.get("label") or "")
    for section in panel.get("sections", []):
      section_title = str(section.get("title") or "")
      walk_items(section.get("items"), panel_label, section_title)
      for sub_panel in section.get("sub_panels", []):
        walk_items(sub_panel.get("items"), panel_label, str(sub_panel.get("label") or section_title))

  vehicle = data.get("vehicle_settings") or {}
  for section in vehicle.get("sections", []):
    section_title = str(section.get("title") or "Vehicle")
    walk_items(section.get("items"), "Vehicle", section_title)

  return labels


@lru_cache(maxsize=1)
def _metadata_labels() -> dict[str, dict[str, str]]:
  if not PARAMS_METADATA_PATH.exists():
    return {}
  try:
    data = json.loads(PARAMS_METADATA_PATH.read_text(encoding="utf-8"))
  except Exception:
    logger.exception("Failed to read params_metadata.json")
    return {}

  labels: dict[str, dict[str, str]] = {}
  for key, info in data.items():
    if not isinstance(info, dict):
      continue
    labels[key] = {
      "title": str(info.get("title") or key),
      "description": str(info.get("description") or ""),
      "panel": "",
      "section": "",
    }
  return labels


@lru_cache(maxsize=1)
def _description_zh_overrides() -> dict[str, str]:
  if not ZH_DESCRIPTIONS_PATH.exists():
    return {}
  try:
    data = json.loads(ZH_DESCRIPTIONS_PATH.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items() if v}
  except Exception:
    logger.exception("Failed to load param_descriptions_zh-CHS.json")
    return {}


def _translate(text: str, locale: str, translations: dict[str, str]) -> str:
  if not text or locale != "zh-CHS":
    return text
  if text in translations:
    return translations[text]
  alias = DESCRIPTION_PO_ALIASES.get(text)
  if alias and alias in translations:
    return translations[alias]
  return text


def get_param_label(key: str, locale: str = "en") -> dict[str, str]:
  """Return localized display fields for a parameter key."""
  settings = _settings_ui_labels()
  metadata = _metadata_labels()
  translations = _load_po_translations() if locale == "zh-CHS" else {}

  source = (
    NATIVE_TITLE_OVERRIDES.get(key)
    or settings.get(key)
    or metadata.get(key)
    or {
      "title": key,
      "description": "",
      "panel": "",
      "section": "",
    }
  )

  title_en = source.get("title") or key
  description_en = source.get("description") or ""
  panel_en = source.get("panel") or ""
  section_en = source.get("section") or ""

  title = _translate(title_en, locale, translations)
  description = _translate(description_en, locale, translations)
  if locale == "zh-CHS" and description_en and description == description_en:
    description = _description_zh_overrides().get(key, description)
  panel = _translate(panel_en, locale, translations) if panel_en else ""
  section = _translate(section_en, locale, translations) if section_en else ""

  category = panel or ("sunnypilot" if key in settings else "System")
  if locale == "zh-CHS":
    if category == "System":
      category = "系统"
    elif category == "BluePilot":
      category = "BluePilot"

  return {
    "title": title,
    "title_en": title_en,
    "description": description,
    "description_en": description_en,
    "panel": panel,
    "section": section,
    "category_label": category,
  }


def get_param_labels_map(locale: str = "en") -> dict[str, dict[str, str]]:
  """Return labels for all known settings/metadata params."""
  keys = set(_settings_ui_labels()) | set(_metadata_labels())
  return {key: get_param_label(key, locale) for key in sorted(keys)}
