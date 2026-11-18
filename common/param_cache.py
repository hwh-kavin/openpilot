"""Small in-RAM cache in front of Params for realtime processes.

``Params.get()`` is a plain file read on every call (open/read/close against the
params directory, see ``Params::get`` in params.cc) - there is no cache layer in
the C++ side. Calling it from a realtime loop such as modeld's 50 ms budget costs
a handful of syscalls per call and, more importantly, can block for tens of
milliseconds whenever the storage device is busy with write-back (qlog rotation,
uploads), which is exactly when the loop can least afford it.

``ParamCache`` keeps the last read value in RAM and refreshes it on a slow TTL,
so a UI toggle is still picked up within ``ttl`` seconds without touching the
filesystem on every iteration.

The API mirrors the subset of ``Params`` used on these paths, including
``get(..., return_default=True)`` and ``get_bool()``, so it can be dropped in
anywhere a read-only ``Params`` is used.
"""
import time

from openpilot.common.params import Params


class ParamCache:
  def __init__(self, params: Params | None = None, ttl: float = 30.0):
    if ttl <= 0:
      raise ValueError("ttl must be positive")
    self._params = params if params is not None else Params()
    self._ttl = ttl
    self._values: dict[tuple[str, bool], object] = {}
    self._timestamps: dict[tuple[str, bool], float] = {}

  def _fresh(self, cache_key: tuple[str, bool]) -> bool:
    ts = self._timestamps.get(cache_key)
    return ts is not None and (time.monotonic() - ts) < self._ttl

  def get(self, key: str, block: bool = False, return_default: bool = False):
    """Cached equivalent of ``Params.get(key, block, return_default)``."""
    cache_key = (key, return_default)
    if not self._fresh(cache_key):
      self._values[cache_key] = self._params.get(key, block=block, return_default=return_default)
      self._timestamps[cache_key] = time.monotonic()
    return self._values[cache_key]

  def get_bool(self, key: str, block: bool = False) -> bool:
    """Cached equivalent of ``Params.get_bool(key, block)``.

    Params stores booleans as "1"/"0" and ``getBool`` is ``get(key) == "1"``, so an
    unset key yields False.
    """
    value = self.get(key, block=block)
    if value is None:
      return False
    return bool(value)

  def invalidate(self) -> None:
    """Drop every cached value, forcing the next read to hit storage."""
    self._values.clear()
    self._timestamps.clear()
