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
import tempfile
import time
from datetime import datetime, timezone
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

AGGREGATE_META_FILENAME = "aggregate_meta.json"


def _aggregate_meta_path() -> str:
    return os.path.join(DEFAULT_CACHE_DIR, AGGREGATE_META_FILENAME)


def _empty_aggregate_meta() -> Dict[str, Any]:
    return {
        'bootstrapped': False,
        'last_processed_date': None,
        'processed_routes': {},
        'cumulative_all': {'routes': 0, 'distance': 0.0, 'duration': 0.0},
        'cumulative_week': {'routes': 0, 'distance': 0.0, 'duration': 0.0},
        'last_incremental_boot_id': None,
    }


def load_aggregate_meta() -> Dict[str, Any]:
    """Load per-route aggregate bookkeeping used for incremental stats updates."""
    path = _aggregate_meta_path()
    if not os.path.exists(path):
        return _empty_aggregate_meta()
    try:
        with open(path, encoding='utf-8') as handle:
            meta = json.load(handle)
        if not isinstance(meta, dict):
            return _empty_aggregate_meta()
        meta.setdefault('bootstrapped', False)
        meta.setdefault('last_processed_date', None)
        meta.setdefault('processed_routes', {})
        meta.setdefault('cumulative_all', {'routes': 0, 'distance': 0.0, 'duration': 0.0})
        meta.setdefault('cumulative_week', {'routes': 0, 'distance': 0.0, 'duration': 0.0})
        meta.setdefault('last_incremental_boot_id', None)
        if not isinstance(meta['processed_routes'], dict):
            meta['processed_routes'] = {}
        return meta
    except Exception as exc:
        logger.debug("Failed reading aggregate meta: %s", exc)
        return _empty_aggregate_meta()


def save_aggregate_meta(meta: Dict[str, Any]) -> bool:
    """Persist aggregate meta atomically."""
    path = _aggregate_meta_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(path),
            prefix='.tmp_aggregate_meta_',
            suffix='.json',
        )
        try:
            payload = json.dumps(meta, separators=(',', ':')).encode('utf-8')
            os.write(temp_fd, payload)
            os.close(temp_fd)
            temp_fd = None
            os.replace(temp_path, path)
            return True
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    except Exception as exc:
        logger.warning("Failed writing aggregate meta: %s", exc)
        return False


def totals_from_aggregate_meta(
    meta: Dict[str, Any],
    week_cutoff: Optional[datetime] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Sum all-time and week totals from processed route entries."""
    all_time_stats = {'routes': 0, 'distance': 0.0, 'duration': 0.0}
    week_stats = {'routes': 0, 'distance': 0.0, 'duration': 0.0}

    for entry in meta.get('processed_routes', {}).values():
        distance = float(entry.get('distance', 0) or 0)
        duration = float(entry.get('duration', 0) or 0)
        all_time_stats['routes'] += 1
        all_time_stats['distance'] += distance
        all_time_stats['duration'] += duration

        if week_cutoff is None:
            continue

        timestamp = entry.get('timestamp')
        if not timestamp:
            continue
        try:
            route_dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
            if route_dt.tzinfo is None:
                route_dt = route_dt.replace(tzinfo=timezone.utc)
            if route_dt >= week_cutoff:
                week_stats['routes'] += 1
                week_stats['distance'] += distance
                week_stats['duration'] += duration
        except (ValueError, AttributeError, TypeError) as exc:
            logger.debug("Could not parse processed route timestamp %s: %s", timestamp, exc)

    return all_time_stats, week_stats


def get_system_boot_id() -> Optional[int]:
    """Return Unix timestamp of the current boot (stable until next power cycle)."""
    try:
        with open('/proc/stat', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('btime '):
                    return int(line.split()[1])
    except Exception as exc:
        logger.debug("Could not read /proc/stat btime: %s", exc)

    try:
        with open('/proc/uptime', encoding='utf-8') as handle:
            uptime_seconds = float(handle.read().split()[0])
        return int(time.time() - uptime_seconds)
    except Exception as exc:
        logger.debug("Could not derive boot id from /proc/uptime: %s", exc)

    return None


def boot_incremental_pending(meta: Optional[Dict[str, Any]] = None) -> bool:
    """True when this power cycle has not yet run latest-date stats aggregation."""
    if meta is None:
        meta = load_aggregate_meta()
    boot_id = get_system_boot_id()
    if boot_id is None:
        return False
    return meta.get('last_incremental_boot_id') != boot_id


def mark_boot_incremental_done(meta: Dict[str, Any], boot_id: Optional[int] = None) -> None:
    """Record that latest-date aggregation finished for this boot."""
    meta['last_incremental_boot_id'] = boot_id if boot_id is not None else get_system_boot_id()


def get_stored_cumulative_totals(meta: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return persisted cumulative totals from meta."""
    if meta is None:
        meta = load_aggregate_meta()
    all_stats = dict(meta.get('cumulative_all') or {'routes': 0, 'distance': 0.0, 'duration': 0.0})
    week_stats = dict(meta.get('cumulative_week') or {'routes': 0, 'distance': 0.0, 'duration': 0.0})
    return all_stats, week_stats


def sync_cumulative_totals(meta: Dict[str, Any], week_cutoff: Optional[datetime] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Recompute cumulative totals from processed routes and persist in meta."""
    all_stats, week_stats = totals_from_aggregate_meta(meta, week_cutoff)
    meta['cumulative_all'] = all_stats
    meta['cumulative_week'] = week_stats
    return all_stats, week_stats


def format_drive_stats_distance(distance_miles: float, is_metric: bool) -> str:
    """Format cached/API drive stats distance for UI (FrogPilot-compatible)."""
    from openpilot.common.constants import CV

    distance = float(distance_miles or 0)
    if is_metric:
        return str(int(distance * CV.MPH_TO_KPH))  # same factor as MILE_TO_KM
    return str(int(distance))


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


def get_drive_stats_cache_mtime() -> Optional[float]:
    """Return mtime of the newest drive stats cache slot, if any."""
    return _rotating_cache.latest_mtime()


def is_drive_stats_cache_fresh(max_age_seconds: int = DEFAULT_FRESH_SECONDS) -> bool:
    """Return True if rotating cache was updated recently."""
    latest_mtime = get_drive_stats_cache_mtime()
    if latest_mtime is None:
        return False
    return (time.time() - latest_mtime) <= max_age_seconds


def _aggregate_cache_missing_distance(payload: Optional[Dict[str, Any]]) -> bool:
    """True when cached totals have drives but no distance."""
    if not payload:
        return True
    for period in ('all', 'week'):
        block = payload.get(period, {})
        if int(block.get('routes', 0)) > 0 and float(block.get('distance', 0)) <= 0:
            return True
    return False


def needs_drive_stats_recalculate(params=None) -> bool:
    """Return True when aggregate stats need a boot-time or first-run update."""
    meta = load_aggregate_meta()
    if not meta.get('bootstrapped'):
        return True

    stats, _source = load_drive_stats(params)
    if _aggregate_cache_missing_distance(stats):
        return True

    return boot_incremental_pending(meta)


def reload_drive_stats(params=None, recalculate_if_stale: bool = False) -> Dict[str, Any]:
    """Load drive stats; optionally run one latest-date update per boot."""
    stats, _source = load_drive_stats(params)
    if recalculate_if_stale and needs_drive_stats_recalculate(params):
        try:
            from bluepilot.backend.routes.processing import recalculate_aggregate_drive_stats
            meta = load_aggregate_meta()
            force_full = not meta.get('bootstrapped') or _aggregate_cache_missing_distance(stats)
            recalculate_aggregate_drive_stats(force_full=force_full)
        except Exception as exc:
            logger.warning("Failed to recalculate aggregate drive stats: %s", exc)

    stats, _source = load_drive_stats(params)
    return stats or {}
