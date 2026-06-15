#!/usr/bin/env python3
"""
Rotating JSON file cache.

Spreads flash wear by round-robin writing across multiple slot files instead
of rewriting the same path on every update.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RotatingJsonCache:
    """Round-robin JSON cache backed by numbered slot files."""

    def __init__(self, cache_dir: str, prefix: str, slots: int = 8):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.slots = max(2, int(slots))
        os.makedirs(self.cache_dir, exist_ok=True)

    def _slot_path(self, index: int) -> str:
        return os.path.join(self.cache_dir, f"{self.prefix}.{index:03d}.json")

    def read(self) -> Optional[Any]:
        """Return the newest valid JSON payload across all slots."""
        latest_data = None
        latest_mtime = -1.0

        for index in range(self.slots):
            path = self._slot_path(index)
            if not os.path.exists(path):
                continue
            try:
                mtime = os.path.getmtime(path)
                if mtime <= latest_mtime:
                    continue
                with open(path, encoding='utf-8') as handle:
                    latest_data = json.load(handle)
                latest_mtime = mtime
            except Exception as exc:
                logger.debug("Failed reading rotating cache slot %s: %s", path, exc)

        return latest_data

    def latest_mtime(self) -> Optional[float]:
        """Return mtime of the newest slot, if any."""
        latest_mtime = None
        for index in range(self.slots):
            path = self._slot_path(index)
            if not os.path.exists(path):
                continue
            mtime = os.path.getmtime(path)
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
        return latest_mtime

    def _next_slot(self) -> int:
        """Pick the oldest slot (or first missing slot) for the next write."""
        target = 0
        oldest_mtime = float('inf')
        for index in range(self.slots):
            path = self._slot_path(index)
            if not os.path.exists(path):
                return index
            mtime = os.path.getmtime(path)
            if mtime < oldest_mtime:
                oldest_mtime = mtime
                target = index
        return target

    def write(self, data: Any) -> bool:
        """Atomically write JSON to the next rotating slot."""
        target = self._next_slot()
        path = self._slot_path(target)

        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix='.tmp_',
                suffix=f'.{self.prefix}.json',
            )
            try:
                payload = json.dumps(data, separators=(',', ':')).encode('utf-8')
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
            logger.warning("Failed writing rotating cache slot %s: %s", path, exc)
            return False

    def write_if_changed(self, data: Any) -> bool:
        """Write only when payload differs from the newest cached value."""
        current = self.read()
        if current == data:
            return False
        return self.write(data)
