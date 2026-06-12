#!/usr/bin/env python3
import datetime
import time
from typing import NoReturn

import cereal.messaging as messaging
from cereal import log
from openpilot.common.time_helpers import min_date, MAX_DATE, system_time_valid, set_system_time, sync_time_from_network
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.gps import get_gps_location_service
from openpilot.system.hardware import HARDWARE

NetworkType = log.DeviceState.NetworkType
_network_synced = False


def maybe_network_time_sync() -> None:
  global _network_synced
  if _network_synced:
    return
  if HARDWARE.get_network_type() == NetworkType.none:
    return
  if sync_time_from_network():
    _network_synced = True
    cloudlog.info("Network time sync completed")


def main() -> NoReturn:
  """
    timed has two responsibilities:
    - getting the current time from GPS
    - publishing the time in the logs

    AGNOS will also use NTP to update the time.
  """

  params = Params()
  gps_location_service = get_gps_location_service(params)

  pm = messaging.PubMaster(['clocks'])
  sm = messaging.SubMaster([gps_location_service])
  while True:
    sm.update(1000)
    maybe_network_time_sync()

    msg = messaging.new_message('clocks')
    msg.valid = system_time_valid()
    msg.clocks.wallTimeNanos = time.time_ns()
    pm.send('clocks', msg)

    gps = sm[gps_location_service]
    gps_time = datetime.datetime.fromtimestamp(gps.unixTimestampMillis / 1000.)
    if not sm.updated[gps_location_service] or (time.monotonic() - sm.logMonoTime[gps_location_service] / 1e9) > 2.0:
      continue
    if not gps.hasFix:
      continue
    if gps_time < min_date() or gps_time > MAX_DATE:
      continue

    set_system_time(gps_time)
    time.sleep(10)

if __name__ == "__main__":
  main()
