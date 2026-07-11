#!/usr/bin/env python3
import time
import cereal.messaging as messaging
from cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_DMON
from openpilot.selfdrive.monitoring.policy import DriverMonitoring

AlertLevel = log.DriverMonitoringState.AlertLevel
MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy


def get_disabled_dm_packet():
  dat = messaging.new_message('driverMonitoringState', valid=True)
  dm = dat.driverMonitoringState
  dm.alertLevel = AlertLevel.none
  dm.activePolicy = MonitoringPolicy.wheeltouch
  dm.lockout = False
  dm.alwaysOnLockout = False
  dm.visionPolicyState.awarenessPercent = 100
  dm.wheeltouchPolicyState.awarenessPercent = 100
  return dat


def dmonitoringd_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  params = Params()
  pm = messaging.PubMaster(['driverMonitoringState'])
  sm = messaging.SubMaster(['driverStateV2', 'liveCalibration', 'carState', 'selfdriveState', 'modelV2',
                            'carControl'], poll='driverStateV2')

  DM = DriverMonitoring(rhd_saved=params.get_bool("IsRhdDetected"), always_on=params.get_bool("AlwaysOnDM"))
  demo_mode = False
  driver_model_enabled = params.get_bool("DriverModelEnable")

  # 20Hz <- dmonitoringmodeld
  while True:
    if driver_model_enabled:
      sm.update(0)
      dat = get_disabled_dm_packet()
      pm.send('driverMonitoringState', dat)
      if sm.frame % int(1 / DT_DMON) == 0:
        driver_model_enabled = params.get_bool("DriverModelEnable")
        demo_mode = params.get_bool("IsDriverViewEnabled")
      time.sleep(DT_DMON)
      continue

    sm.update()
    if not sm.updated['driverStateV2']:
      # iterate when model has new output
      continue

    valid = sm.all_checks()
    if demo_mode and sm.valid['driverStateV2']:
      DM.run_step(sm, demo=True)
    elif valid:
      DM.run_step(sm, demo=demo_mode)

    # publish
    dat = DM.get_state_packet(valid=valid)
    pm.send('driverMonitoringState', dat)

    # load live always-on toggle
    if sm['driverStateV2'].frameId % 40 == 1:
      DM.always_on = params.get_bool("AlwaysOnDM")
      demo_mode = params.get_bool("IsDriverViewEnabled")
      driver_model_enabled = params.get_bool("DriverModelEnable")

    # save rhd virtual toggle every 5 mins
    if (sm['driverStateV2'].frameId % 6000 == 0 and not demo_mode and
     DM.wheelpos_offsetter.filtered_stat.n > DM.settings._WHEELPOS_FILTER_MIN_COUNT and
     DM.wheel_on_right == (DM.wheelpos_offsetter.filtered_stat.M > DM.settings._WHEELPOS_THRESHOLD)):
      params.put_bool("IsRhdDetected", DM.wheel_on_right)

def main():
  dmonitoringd_thread()


if __name__ == '__main__':
  main()
