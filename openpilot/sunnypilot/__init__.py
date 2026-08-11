"""Compatibility shim for the legacy openpilot.sunnypilot import path.

This repo keeps the Sunnypilot sources under the top-level ``sunnypilot`` package.
Older code imports them as ``openpilot.sunnypilot.*``. Re-export the real module's
public API so imports continue to resolve without copying the package tree.
"""

from pathlib import Path

_real_root = Path(__file__).resolve().parents[2] / "sunnypilot"
if _real_root.exists():
  __path__ = [str(_real_root)]

import sunnypilot as _real_sunnypilot
for _name in getattr(_real_sunnypilot, "__all__", []):
  globals()[_name] = getattr(_real_sunnypilot, _name)
for _name in [
  "PARAMS_UPDATE_PERIOD",
  "IntEnumBase",
  "get_sanitize_int_param",
  "get_file_hash",
]:
  if hasattr(_real_sunnypilot, _name):
    globals()[_name] = getattr(_real_sunnypilot, _name)
