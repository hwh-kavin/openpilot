#!/usr/bin/env python3
from cereal import car, custom
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.radar_fallback import highway_radar_fallback_enabled
import cereal.messaging as messaging


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  cloudlog.info("plannerd is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("plannerd got CarParamsSP")

  gps_location_service = get_gps_location_service(params)

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP, CP_SP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  # Map/GPS are optional (speed-limit assist). Until gpsOK, liveMapDataSP stays invalid and
  # must not make longitudinalPlan*/driverAssistance invalid → "Communication Issue Between Processes".
  ignore = ['liveMapDataSP', gps_location_service]
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'liveMapDataSP', 'carStateSP', gps_location_service],
                           ignore_alive=ignore, ignore_avg_freq=ignore, ignore_valid=ignore,
                           poll='carState')

  while True:
    sm.update()
    longitudinal_planner.sla.update_car_state(sm['carState'])
    if sm.updated['modelV2']:
      longitudinal_planner.update(sm)
      longitudinal_planner.publish(sm, pm)

      ldw.update(sm.frame, sm['modelV2'], sm['carState'], sm['carControl'])
      radar_fallback = highway_radar_fallback_enabled(CP, sm['carState'].vEgo) and \
        sm.all_checks(['modelV2', 'carState', 'controlsState', 'selfdriveState'])
      valid_services = ['carControl', 'carState', 'controlsState', 'liveParameters',
                        'modelV2', 'selfdriveState', 'carStateSP']
      if not radar_fallback:
        valid_services.append('radarState')
      msg = messaging.new_message('driverAssistance')
      msg.valid = sm.all_checks(service_list=valid_services)
      msg.driverAssistance.leftLaneDeparture = ldw.left
      msg.driverAssistance.rightLaneDeparture = ldw.right
      pm.send('driverAssistance', msg)


if __name__ == "__main__":
  main()
