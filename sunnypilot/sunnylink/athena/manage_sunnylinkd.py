#!/usr/bin/env python3
import time
from multiprocessing import Process

from openpilot.common.params import Params
from openpilot.system.manager.process import launcher
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import HARDWARE
from openpilot.system.version import get_build_metadata

SUNNYLINKD_MODULE = 'sunnypilot.sunnylink.athena.sunnylinkd'
SUNNYLINKD_PID_PARAM = "SunnylinkdPid"


def manage_sunnylinkd():
  params = Params()
  dongle_id = params.get("SunnylinkDongleId")
  build_metadata = get_build_metadata()

  cloudlog.bind_global(dongle_id=dongle_id,
                       version=build_metadata.openpilot.version,
                       origin=build_metadata.openpilot.git_normalized_origin,
                       branch=build_metadata.channel,
                       commit=build_metadata.openpilot.git_commit,
                       dirty=build_metadata.openpilot.is_dirty,
                       device=HARDWARE.get_device_type())

  try:
    while True:
      if not params.get_bool("SunnylinkEnabled"):
        cloudlog.info("Sunnylink disabled, not starting sunnylinkd")
        time.sleep(5)
        continue

      cloudlog.info("starting sunnylinkd daemon")
      proc = Process(name='sunnylinkd', target=launcher, args=(SUNNYLINKD_MODULE, 'sunnylinkd'))
      proc.start()
      proc.join()
      cloudlog.event("sunnylinkd exited", exitcode=proc.exitcode)
      time.sleep(5)
  except Exception:
    cloudlog.exception("manage_sunnylinkd.exception")
  finally:
    params.remove(SUNNYLINKD_PID_PARAM)


def main():
  manage_sunnylinkd()


if __name__ == '__main__':
  main()
