#!/usr/bin/env python3
"""
USB/panda jitter monitor — evidence capture for the "Communication Issue Between
Processes" (commIssue) investigation.

The commIssue trigger is ``deviceState`` (hardwared) going not-alive, which is
suspected to be a USB hub / panda / system stall. dmesg is wiped on reboot, so the
only way to catch the next occurrence is to tail the kernel log live and persist it.

This script:
  1. tails ``dmesg -w`` and appends USB-related kernel events to a log file, and
  2. snapshots the panda USB device / device nodes / pandad liveness every few
     seconds (logged on change), so a panda drop is recorded even if the kernel
     log is silent about it.

Log: /data/community/crashes/usb_jitter.log  (bounded to ~256 KB, survives reboot)

Run:
  nohup python3 /data/openpilot/tools/usb_jitter_monitor.py >/dev/null 2>&1 &

Auto-start (optional): add it to process_config.py as an always_run process, or a
systemd unit.
"""
import os
import re
import select
import subprocess
import time
from datetime import datetime

LOG_DIR = "/data/community/crashes"
LOG_PATH = os.path.join(LOG_DIR, "usb_jitter.log")
MAX_BYTES = 256 * 1024
STATUS_INTERVAL_S = 5.0

# Kernel messages that indicate USB/hub/panda activity.
USB_RE = re.compile(
  r"usb|panda|disconnect|reset|unregister|enumerat|xhci|dwc3|hub|over-current|not running at top speed",
  re.IGNORECASE,
)

PANDA_VIDPID = "3801:ddcc"


def boot_id() -> str:
  try:
    with open("/proc/sys/kernel/random/boot_id") as f:
      return f.read().strip()
  except Exception:
    return "unknown"


def append(line: str) -> None:
  ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  entry = f"[{ts}] {line}\n"
  try:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
      f.write(entry)
    # Keep the file bounded: drop the oldest half.
    if os.path.getsize(LOG_PATH) > MAX_BYTES:
      with open(LOG_PATH, encoding="utf-8") as f:
        data = f.read()
      mid = len(data) // 2
      nl = data.find("\n", mid)
      if nl != -1:
        data = data[nl + 1:]
      with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(data)
  except OSError:
    pass


def panda_status() -> tuple[bool, list[str], bool | None]:
  """(panda on USB bus, panda device nodes, pandad alive)."""
  usb_ok = False
  try:
    out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5).stdout
    usb_ok = PANDA_VIDPID in out
  except Exception:
    pass

  nodes = [p for p in ("/dev/panda", "/dev/panda1", "/dev/panda2") if os.path.exists(p)]

  pandad = None
  try:
    pandad = subprocess.run(["pgrep", "-f", "pandad"], capture_output=True, timeout=5).returncode == 0
  except Exception:
    pass

  return usb_ok, nodes, pandad


def start_dmesg() -> subprocess.Popen:
  return subprocess.Popen(["dmesg", "-w"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)


def main() -> None:
  append(f"=== monitor started (boot_id={boot_id()}) ===")
  proc = start_dmesg()
  last_status = None
  last_check = 0.0

  while True:
    # Drain pending kernel-log lines without blocking for long.
    try:
      ready, _, _ = select.select([proc.stdout], [], [], 0.5)
      while ready:
        line = proc.stdout.readline()
        if not line:
          break
        line = line.strip()
        if USB_RE.search(line):
          append(f"dmesg: {line}")
        ready, _, _ = select.select([proc.stdout], [], [], 0.0)
    except (OSError, ValueError):
      pass

    # Periodic panda snapshot, log only on change.
    now = time.time()
    if now - last_check >= STATUS_INTERVAL_S:
      last_check = now
      usb_ok, nodes, pandad = panda_status()
      status = (usb_ok, tuple(nodes), pandad)
      if status != last_status:
        state = f"usb={'ok' if usb_ok else 'MISSING'} nodes={nodes} pandad={'alive' if pandad else ('dead' if pandad is False else '?')}"
        append(f"panda_status: {state}")
        last_status = status

    # Restart the tail if the kernel-log reader died.
    if proc.poll() is not None:
      append("dmesg -w exited; restarting")
      proc = start_dmesg()


if __name__ == "__main__":
  main()
