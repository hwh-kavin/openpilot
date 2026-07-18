#!/usr/bin/env python3
# simple pandad wrapper that updates the panda first
import os
import usb1
import time
import signal
import subprocess

from panda import Panda, PandaDFU, PandaProtocolMismatch, McuType, FW_PATH
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.system.hardware import HARDWARE
from openpilot.common.swaglog import cloudlog

# C3/DOS internal panda is USB-only (often behind a hub). Prefer connecting immediately
# when serial is already readable; only GPIO-reset + settle when list() is empty.
NO_PANDA_RESET_SETTLE_S = 1.0
NO_PANDA_POLL_S = 0.2
NO_PANDA_RESET_EVERY_S = 5.0
PANDA_STABLE_BEFORE_MODEM_S = 2.0

# Avoid immediate GPIO reset on first empty list(); wait one reset interval.
_last_panda_reset_at = time.monotonic()


def reset_and_wait_for_usb() -> None:
  """GPIO-reset internal panda, then wait for USB re-enumeration before list/serial."""
  global _last_panda_reset_at
  if not HARDWARE.has_internal_panda():
    return
  cloudlog.event("pandad.reset_internal")
  HARDWARE.reset_internal_panda()
  _last_panda_reset_at = time.monotonic()
  time.sleep(NO_PANDA_RESET_SETTLE_S)


def list_pandas_for_connect() -> list[str]:
  """List pandas; if USB not ready yet, poll, and only reset every few seconds."""
  serials = Panda.list()
  if serials:
    return serials
  now = time.monotonic()
  if now - _last_panda_reset_at >= NO_PANDA_RESET_EVERY_S:
    reset_and_wait_for_usb()
  else:
    time.sleep(NO_PANDA_POLL_S)
  return Panda.list()


def enable_modem_usb_after_panda() -> None:
  """Panda is up — allow LTE USB enumeration on the shared hub."""
  try:
    time.sleep(PANDA_STABLE_BEFORE_MODEM_S)
    HARDWARE.allow_modem_usb()
    cloudlog.event("pandad.modem_usb_enabled")
  except Exception:
    cloudlog.exception("pandad.modem_usb_enable_failed")


def get_expected_signature(panda: Panda) -> bytes:
  hw_type = panda.get_type()
  mcu_type = McuType.F4 if hw_type == Panda.HW_TYPE_DOS else McuType.H7
  fn = os.path.join(FW_PATH, mcu_type.config.app_fn)
  return Panda.get_signature_from_firmware(fn)

def flash_panda(panda_serial: str):
  panda = Panda(panda_serial)

  # skip flashing for unsupported external pandas; internal DOS/F4 pandas are still allowed
  hw_type = panda.get_type()
  if hw_type not in Panda.SUPPORTED_DEVICES and not panda.is_internal():
    cloudlog.warning(f"Panda {panda_serial} is not supported (hw_type: {hw_type}), skipping flash...")
    panda.close()
    return

  fw_signature = get_expected_signature(panda)
  internal_panda = panda.is_internal()

  panda_version = "bootstub" if panda.bootstub else panda.get_version()
  panda_signature = b"" if panda.bootstub else panda.get_signature()
  cloudlog.warning(f"Panda {panda_serial} connected, version: {panda_version}, signature {panda_signature.hex()[:16]}, expected {fw_signature.hex()[:16]}")

  if panda.bootstub or panda_signature != fw_signature:
    cloudlog.info("Panda firmware out of date, update required")
    panda.flash()
    cloudlog.info("Done flashing")

  if panda.bootstub:
    bootstub_version = panda.get_version()
    cloudlog.info(f"Flashed firmware not booting, flashing development bootloader. {bootstub_version=}, {internal_panda=}")
    if internal_panda:
      HARDWARE.recover_internal_panda()
    panda.recover(reset=(not internal_panda))
    cloudlog.info("Done flashing bootstub")

  if panda.bootstub:
    cloudlog.info("Panda still not booting, exiting")
    raise AssertionError

  panda_signature = panda.get_signature()
  if panda_signature != fw_signature:
    cloudlog.info("Version mismatch after flashing, exiting")
    raise AssertionError

  panda.close()


def check_panda_support(panda_serials: list[str]) -> list[str]:
  spi_serials = set(Panda.spi_list())
  for serial in panda_serials:
    if serial in spi_serials:
      return [serial]

  for serial in panda_serials:
    panda = Panda(serial)
    is_internal = panda.is_internal()
    panda.close()
    if is_internal:
      return [serial]

  return []


def main() -> None:
  # signal pandad to close the relay and exit
  def signal_handler(signum, frame):
    cloudlog.info(f"Caught signal {signum}, exiting")
    nonlocal do_exit
    do_exit = True
    if process is not None:
      process.send_signal(signal.SIGINT)

  process = None
  do_exit = False
  signal.signal(signal.SIGINT, signal_handler)

  count = 0
  heartbeat_checked = False
  while not do_exit:
    try:
      cloudlog.event("pandad.flash_and_connect", count=count)
      count += 1

      # Fast path: connect if USB/serial already ready; else reset → wait 1s → list
      panda_serials = list_pandas_for_connect()
      if not panda_serials:
        cloudlog.event("pandad.no_panda_after_reset", count=count)
        continue

      # check health for lost heartbeat (once, on a clean USB connection)
      if not heartbeat_checked:
        heartbeat_checked = True
        try:
          for s in panda_serials:
            with Panda(s) as p:
              try:
                health = p.health()
              except RuntimeError:
                continue
              if p.is_internal() and health["heartbeat_lost"]:
                Params().put_bool("PandaHeartbeatLost", True, block=True)
                cloudlog.event("heartbeat lost", deviceState=health)
        except Exception:
          cloudlog.exception("pandad.uncaught_exception")

      # Flash all Pandas in DFU mode
      for serial in PandaDFU.list():
        cloudlog.info(f"Panda in DFU mode found, flashing recovery {serial}")
        PandaDFU(serial).recover()
        time.sleep(1)

      panda_serials = Panda.list()
      if len(panda_serials):
        # find the internal supported panda (e.g. skip external Black Panda)
        panda_serials = check_panda_support(panda_serials)

        assert len(panda_serials) == 1
        cloudlog.info(f"{len(panda_serials)} panda found, connecting - {panda_serials}")
        flash_panda(panda_serials[0])

        # run real pandad
        os.environ['MANAGER_DAEMON'] = 'pandad'
        process = subprocess.Popen(["./pandad"], cwd=os.path.join(BASEDIR, "selfdrive/pandad"))
        # After panda is connected, allow LTE USB enum (deferred to protect shared hub)
        enable_modem_usb_after_panda()
        process.wait()
    # TODO: wrap all panda exceptions in a base panda exception
    except (usb1.USBErrorNoDevice, usb1.USBErrorPipe):
      # a panda was disconnected while setting everything up. let's try again
      cloudlog.exception("Panda USB exception while setting up")
    except PandaProtocolMismatch:
      cloudlog.exception("pandad.protocol_mismatch")
    except Exception:
      cloudlog.exception("pandad.uncaught_exception")


if __name__ == "__main__":
  main()
