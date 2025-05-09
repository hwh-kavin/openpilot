# from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.selfdrived.events import ET
from opendbc.car import DT_CTRL

def hysteresis(current_value, old_value, target: float, delta: float):
  if target < current_value < min(target + delta, 0):
    result = old_value
  elif current_value <= target:
    result = 1
  elif current_value >= min(target + delta, 0):
    result = 0

  return result


def actuators_calc(cc_self, brake): # cc_self is the CarController object (self)
  ts = cc_self.frame * DT_CTRL

  brake_actuate = hysteresis(brake, cc_self.brake_actuate_last, cc_self.brake_actuator_activate, cc_self.brake_actuator_release_delta)
  cc_self.brake_actuate_last = brake_actuate

  precharge_actuate = hysteresis(
    brake, cc_self.precharge_actuate_last, (cc_self.brake_actuator_activate+cc_self.precharge_actuator_target_delta), (cc_self.brake_actuator_release_delta-cc_self.precharge_actuator_target_delta)
  )

  if precharge_actuate and not cc_self.precharge_actuate_last:
    cc_self.precharge_actuate_ts = ts
  elif not precharge_actuate:
    cc_self.precharge_actuate_ts = 0

  if (
    precharge_actuate
    and not brake_actuate
    and cc_self.precharge_actuate_ts > 0
    and brake > cc_self.brake_actuator_activate
    and (ts - cc_self.precharge_actuate_ts) > (200 * DT_CTRL)
  ):
    precharge_actuate = False

  cc_self.precharge_actuate_last = precharge_actuate
  logDebug(f"actuators_calc: {brake}\t{precharge_actuate}\t{brake_actuate}")

  return precharge_actuate, brake_actuate

def get_dm_driver_state(d_state):
  if d_state == "preDriverDistracted/permanent":
    return "preDistracted"
  elif d_state == "promptDriverDistracted/permanent":
    return "promptDistracted"
  elif d_state == "driverDistracted/permanent":
    return "distracted"
  elif d_state == "preDriverUnresponsive/permanent":
    return "preUnresponsive"
  elif d_state == "promptDriverUnresponsive/permanent":
    return "promptUnresponsive"
  elif d_state == "driverUnresponsive/permanent":
    return "unresponsive"
  else:
    return "none"

def get_dm_disable_state(d_state):
  # print(f"d_state: {d_state}")
  if d_state == ET.USER_DISABLE:
    return "userDisable"
  elif d_state == ET.SOFT_DISABLE:
    return "softDisable"
  elif d_state == ET.IMMEDIATE_DISABLE:
    return "immediateDisable"
  elif d_state == ET.NO_ENTRY:
    return "noEntry"
  else:
    return "none"


def compute_dm_msg_values(hud_control, send_hands_free_cluster_msg):
    tja_msg = 0
    tja_warn = 0

    disableState = get_dm_disable_state(hud_control.alertType)
    driverState = get_dm_driver_state(hud_control.eventType)

    if send_hands_free_cluster_msg:
      if disableState == "noEntry":
        tja_msg = 4  # BlueCruise not available
      elif (driverState in ("distracted", "unresponsive") or disableState in ("softDisable", "immediateDisable")):
        tja_warn = 3  # Resume Control
      elif disableState == "userDisable":
        tja_warn = 1  # Cancelled
      elif driverState == "preDistracted":
        tja_warn = 6  # Watch The Road (no chime)
      elif driverState == "promptDistracted":
        tja_warn = 7  # Watch The Road (chime)
      elif hud_control.leftLaneDepart:
        tja_warn = 5  # Left Lane Departure (chime)
      elif hud_control.rightLaneDepart:
        tja_warn = 4  # Right Lane Departure (chime)
      else:
        tja_warn = 0
    else:
      if disableState == "noEntry":
        tja_msg = 1  # Lane Centering Assist not available
      elif (driverState in ("distracted", "unresponsive") or disableState in ("softDisable", "immediateDisable")):
        tja_warn = 3  # Resume Control
      elif disableState == "userDisable":
        tja_warn = 1  # Cancelled
      else:
        tja_warn = 0

    return tja_msg, tja_warn

def get_hev_power_flow_text(mode_value):
  # PwrFlowTxt_D_Dsply 15 "NotUsed7" 14 "NotUsed6" 13 "NotUsed5" 12 "NotUsed4" 11 "Disply_Rgen_Chrg_Txt" 10 "Disp_Fast_Charge_Txt" 9 "Disp_Fast_Charge_Cmplt_Txt" 8 "Disp_Charge_Cmplt_Txt" 7 "Disp_Remote_Start_Txt" 6 "Disp_Eng_Drv_Txt" 5 "Disp_Elec_Drv_Txt" 4 "Disp_Idle_with_Chrg_Txt" 3 "Disp_Idle_Txt" 2 "Disp_Charg_HV_Batt_Txt" 1 "Disp_Hyb_Drive_Txt" 0 "No_Text";
  power_flow_modes = {
    0: "",
    1: "Hybrid Drive",
    2: "Charging HV Battery",
    3: "Idle",
    4: "Idle with Charging",
    5: "Electric Drive",
    6: "Engine Drive",
    7: "Remote Start",
    8: "Charge Complete",
    9: "Fast Charge Complete",
    10: "Fast Charging",
    11: "Regenerative Charging",
    12: "Not Used",
    13: "Not Used",
    14: "Not Used",
    15: "Not Used",
  }
  return power_flow_modes.get(int(mode_value), "Unknown")


def get_hev_engine_on_reason_text(reason_value):
  # EngOnMsg1_D_Dsply 14 "_0xE_to_1F_NotUsed" 12 "Battery_Temperature" 11 "Hill_Decent_Control" 10 "Fuel_Maintenance" 9 "Oil_Maintenance" 8 "Normal_Operation" 7 "Low_Gear" 6 "Batt_Charging" 5 "Engine_Cold" 4 "Neutral_Gear" 3 "Heater_Setting" 2 "High_Speed" 1 "Acceleration" 0 "No_Display" 13 "Drive_Mode_Selection";
  engine_on_reasons = {
    0: "",
    1: "Acceleration",
    2: "High Speed",
    3: "Heater Setting",
    4: "Neutral Gear",
    5: "Engine Cold",
    6: "Battery Charging",
    7: "Low Gear",
    8: "Normal Operation",
    9: "Oil Maintenance",
    10: "Fuel Maintenance",
    11: "Hill Descent Control",
    12: "Battery Temperature",
    13: "Drive Mode",
  }
  return engine_on_reasons.get(int(reason_value), "Unknown")
