#!/usr/bin/env python3
"""Run setup_dependencies.sh with on-screen spinner progress on AGNOS."""
import os
import re
import subprocess
import sys

from openpilot.common.basedir import BASEDIR
from openpilot.common.spinner import Spinner

try:
    from bluepilot.backend.core import install_status
except ImportError:
    install_status = None

SETUP_SCRIPT = os.path.join(BASEDIR, "tools", "setup_dependencies.sh")
BOOTSTRAP_RE = re.compile(r"\[bootstrap\]\s*(.+)")


def _write_bootstrap_status(message: str, *, progress: int | None = None, status: str = "installing") -> None:
    if install_status is None:
        return
    install_status.write_status("bootstrap", status, message, progress=progress)


def run_with_spinner() -> int:
  spinner = Spinner()
  initial = (
    "Installing dependencies\n"
    "First install may take 30+ minutes.\n"
    "Please keep the device powered on."
  )
  spinner.update(initial)
  _write_bootstrap_status("Starting dependency setup...", progress=0)

  proc = subprocess.Popen(
    [SETUP_SCRIPT],
    cwd=BASEDIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
  )
  assert proc.stdout is not None

  for line in proc.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    stripped = line.strip()
    if not stripped:
      continue

    match = BOOTSTRAP_RE.search(stripped)
    if match:
      msg = match.group(1)
      pct_match = re.search(r"^(\d+)%", msg)
      if pct_match:
        pct = int(pct_match.group(1))
        spinner.update(pct_match.group(1))
        _write_bootstrap_status(msg, progress=pct)
      else:
        spinner.update(msg)
        _write_bootstrap_status(msg)
    elif stripped.startswith("  ->"):
      detail = stripped.replace("  -> ", "Installing ")
      spinner.update(detail)
      _write_bootstrap_status(detail)

  rc = proc.wait()
  spinner.close()
  if rc == 0:
    _write_bootstrap_status("Dependencies installed", progress=100, status="ready")
  else:
    _write_bootstrap_status("Dependency setup failed", status="failed")
  return rc


if __name__ == "__main__":
  sys.exit(run_with_spinner())
