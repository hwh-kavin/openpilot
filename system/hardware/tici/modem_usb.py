"""Defer Quectel modem USB configuration until panda is stable on the shared hub.

C3 puts EC25 (500mA) and DOS panda on the same USB hub. Modem enumeration
trips hub power-budget warnings and can knock panda offline. Strategy:

  1. LTE module may already be powered by AGNOS (OK).
  2. Keep modem USB unauthorized so it does not configure / claim hub current.
  3. Let panda enumerate and pandad connect.
  4. Then authorize modem USB so LTE enumerates.

Flag file: /dev/shm/modem_usb_allowed
"""
from __future__ import annotations

import glob
import os
import subprocess
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.utils import sudo_write

MODEM_USB_ALLOW_PATH = "/dev/shm/modem_usb_allowed"
MODEM_VID = "2c7c"
# /etc is read-only on AGNOS; /run/udev/rules.d is writable and loaded by udevd
UDEV_RULE_PATH = "/run/udev/rules.d/99-openpilot-modem-defer.rules"
UDEV_RULE = f'''# Defer Quectel modem USB config until panda is stable (shared hub on C3)
ACTION=="add", SUBSYSTEM=="usb", ATTR{{idVendor}}=="{MODEM_VID}", TEST!="{MODEM_USB_ALLOW_PATH}", ATTR{{authorized}}="0"
'''


def modem_usb_allowed() -> bool:
  return os.path.exists(MODEM_USB_ALLOW_PATH)


def _modem_usb_dirs() -> list[str]:
  out = []
  for path in glob.glob("/sys/bus/usb/devices/*"):
    vid_path = os.path.join(path, "idVendor")
    try:
      if os.path.isfile(vid_path) and open(vid_path).read().strip() == MODEM_VID:
        out.append(path)
    except OSError:
      pass
  return out


def install_modem_defer_udev() -> None:
  """Install udev rule so new modem devices stay unauthorized until allowed."""
  try:
    os.makedirs(os.path.dirname(UDEV_RULE_PATH), exist_ok=True)
    if os.path.isfile(UDEV_RULE_PATH):
      with open(UDEV_RULE_PATH) as f:
        if MODEM_USB_ALLOW_PATH in f.read():
          return
    try:
      with open(UDEV_RULE_PATH, "w") as f:
        f.write(UDEV_RULE)
    except PermissionError:
      subprocess.run(
        ["sudo", "tee", UDEV_RULE_PATH],
        input=UDEV_RULE.encode(),
        stdout=subprocess.DEVNULL,
        check=False,
      )
    subprocess.run(["sudo", "udevadm", "control", "--reload"], check=False)
    cloudlog.info("installed modem USB defer udev rule")
  except Exception:
    cloudlog.exception("failed to install modem USB defer udev rule")


def defer_modem_usb() -> None:
  """Block modem USB configuration (LTE powered OK, but not enumerated/configured)."""
  try:
    if os.path.exists(MODEM_USB_ALLOW_PATH):
      os.remove(MODEM_USB_ALLOW_PATH)
  except OSError:
    pass

  install_modem_defer_udev()

  for path in _modem_usb_dirs():
    auth = os.path.join(path, "authorized")
    if not os.path.exists(auth):
      continue
    try:
      if open(auth).read().strip() != "0":
        sudo_write("0", auth)
        cloudlog.event("modem.usb_deferred", path=path)
    except Exception:
      cloudlog.exception(f"failed to deauthorize modem at {path}")


def allow_modem_usb() -> None:
  """Allow modem USB enumeration after panda is stable."""
  try:
    with open(MODEM_USB_ALLOW_PATH, "w") as f:
      f.write(str(time.time()))
  except OSError:
    cloudlog.exception("failed to create modem USB allow flag")
    return

  # Authorize any already-present deferred modem devices
  found = False
  for path in _modem_usb_dirs():
    auth = os.path.join(path, "authorized")
    if not os.path.exists(auth):
      continue
    try:
      sudo_write("1", auth)
      cloudlog.event("modem.usb_allowed", path=path)
      found = True
    except Exception:
      cloudlog.exception(f"failed to authorize modem at {path}")

  if not found:
    # Modem may have been removed from the bus; trigger a rescan of the hub children
    cloudlog.event("modem.usb_allowed_waiting_for_device")
    for hub in ("/sys/bus/usb/devices/1-1", "/sys/bus/usb/devices/usb1"):
      uevent = os.path.join(hub, "uevent")
      if os.path.exists(uevent):
        try:
          sudo_write("add", uevent)
        except Exception:
          pass
