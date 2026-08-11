"""Append lines to Developer → Error Log (error.log)."""
from __future__ import annotations

import html
import os
from datetime import datetime

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths

# Cap error.log growth when file logging is enabled.
_ERROR_LOG_MAX_BYTES = 512 * 1024


def is_error_log_enabled(params: Params | None = None) -> bool:
  try:
    p = params or Params()
    return bool(p.get_bool("UiAlertLogEnable"))
  except Exception:
    return False


def append_error_log(line: str, *, check_enable: bool = True, params: Params | None = None) -> None:
  """Append text for the Developer → Error Log viewer."""
  if check_enable and not is_error_log_enabled(params):
    return

  try:
    log_dir = Paths.crash_log_root()
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "error.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] {html.escape(line)}<br>\n"

    with open(path, "a", encoding="utf-8") as f:
      f.write(entry)

    # Keep file bounded: drop oldest half if oversized.
    # Find a safe truncation point (after a <br>\n) to avoid breaking HTML.
    try:
      if os.path.getsize(path) > _ERROR_LOG_MAX_BYTES:
        with open(path, encoding="utf-8") as f:
          data = f.read()
        # Truncate at a safe boundary: find the first <br>\n after the midpoint
        mid = len(data) // 2
        safe_pos = data.find("<br>\n", mid)
        if safe_pos != -1:
          data = data[safe_pos + len("<br>\n"):]
        else:
          data = data[mid:]
        with open(path, "w", encoding="utf-8") as f:
          f.write(data)
    except OSError:
      pass
  except Exception:
    cloudlog.exception("failed to append to error.log")
