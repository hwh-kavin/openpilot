#!/usr/bin/env python3
"""
Drive stats persistence with rotating file cache.

Local aggregate stats are stored in round-robin JSON slots under
/data/bluepilot/cache/aggregate_drive_stats/ to reduce flash wear from
repeated writes to a single params key.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from bluepilot.backend.cache.rotating_json_cache import RotatingJsonCache

logger = logging.getLogger(__name__)

PARAM_KEY = "ApiCache_DriveStats"
DEFAULT_CACHE_DIR = (
    "/data/bluepilot/cache/aggregate_drive_stats"
    if os.path.exists("/data")
    else os.path.expanduser("~/comma_data/bluepilot/cache/aggregate_drive_stats")
)
ROTATE_SLOTS = 8
DEFAULT_FRESH_SECONDS = 300

_rotating_cache = RotatingJsonCache(
    cache_dir=DEFAULT_CACHE_DIR,
    prefix="drive_stats",
    slots=ROTATE_SLOTS,
)


def build_drive_stats_payload(all_stats: Dict[str, Any], week_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Build ApiCache_DriveStats-compatible payload from aggregate counters."""
    return {
        'all': {
            'routes': int(all_stats.get('routes', 0)),
            'distance': float(all_stats.get('distance', 0)),
            'minutes': float(all_stats.get('duration', 0)) / 60.0,
        },
        'week': {
            'routes': int(week_stats.get('routes', 0)),
            'distance': float(week_stats.get('distance', 0)),
            'minutes': float(week_stats.get('duration', 0)) / 60.0,
        },
    }


def _normalize_param_payload(raw_value: Any) -> Optional[Dict[str, Any]]:
    if raw_value is None:
        return None
    if isinstance(raw_value, dict):
        return raw_value
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode('utf-8').strip()
    if isinstance(raw_value, str) and raw_value:
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return None
    return None


def _has_drive_stats_data(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    all_stats = payload.get('all', {})
    week_stats = payload.get('week', {})
    for stats in (all_stats, week_stats):
        if stats.get('routes', 0) > 0:
            return True
        if stats.get('distance', 0) > 0:
            return True
        if stats.get('minutes', 0) > 0:
            return True
    return False


def has_drive_stats_data(payload: Optional[Dict[str, Any]]) -> bool:
    return _has_drive_stats_data(payload)


def load_drive_stats(params=None, allow_empty_file_cache: bool = True) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Load drive stats from rotating file cache, falling back to params.

    Returns:
        Tuple[payload_or_none, source_label]
    """
    cached = _rotating_cache.read()
    if cached is not None and (allow_empty_file_cache or _has_drive_stats_data(cached)):
        return cached, 'file_cache'

    if params is None:
        return None, 'missing'

    try:
        param_payload = _normalize_param_payload(params.get(PARAM_KEY))
    except Exception as exc:
        logger.debug("Failed reading %s param: %s", PARAM_KEY, exc)
        param_payload = None

    if not _has_drive_stats_data(param_payload):
        return None, 'missing'

    # Migrate legacy param cache into rotating files once.
    _rotating_cache.write_if_changed(param_payload)
    return param_payload, 'param_cache'


def save_drive_stats(payload: Dict[str, Any]) -> bool:
    """Persist drive stats using rotating file slots (write only if changed)."""
    if not payload:
        return False
    return _rotating_cache.write_if_changed(payload)


def is_drive_stats_cache_fresh(max_age_seconds: int = DEFAULT_FRESH_SECONDS) -> bool:
    """Return True if rotating cache was updated recently."""
    latest_mtime = _rotating_cache.latest_mtime()
    if latest_mtime is None:
        return False
    return (time.time() - latest_mtime) <= max_age_seconds
