#!/usr/bin/env python3
"""Shared install/deploy progress for bootstrap and portal dependency setup."""
import json
import os
import time
from pathlib import Path
from typing import Any

STATUS_PATH = Path("/data/openpilot/.install_status")
MAX_DEPS_RESTARTS = 3
RESTART_COUNT_PATH = Path("/data/openpilot/.bp_deps_restart_count")


def write_status(
    phase: str,
    status: str,
    message: str,
    *,
    progress: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "phase": phase,
        "status": status,
        "message": message,
        "updated": int(time.time()),
    }
    if progress is not None:
        payload["progress"] = max(0, min(100, int(progress)))
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(STATUS_PATH)
    except OSError:
        pass


def read_status() -> dict[str, Any] | None:
    try:
        if not STATUS_PATH.is_file():
            return None
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def clear_status() -> None:
    try:
        STATUS_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def get_restart_count() -> int:
    try:
        if RESTART_COUNT_PATH.is_file():
            return int(RESTART_COUNT_PATH.read_text(encoding="utf-8").strip() or "0")
    except (OSError, ValueError):
        pass
    return 0


def increment_restart_count() -> int:
    count = get_restart_count() + 1
    try:
        RESTART_COUNT_PATH.write_text(str(count), encoding="utf-8")
    except OSError:
        pass
    return count


def reset_restart_count() -> None:
    try:
        RESTART_COUNT_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def should_allow_deps_restart() -> bool:
    return get_restart_count() < MAX_DEPS_RESTARTS
