"""Regression tests for BluePilot's Ford ICBM set-speed follow (ported from BluePilotDev/bluepilot#243).

With ICBM on a Ford the set-speed target follows the truck's cluster set speed at engage and after a driver
Set+/Set- press, so ICBM never presses toward a speed the truck isn't showing. Other brands keep sunnypilot's
behavior. These run against VCruiseHelper directly, one call per card frame (100 Hz).
"""
import pytest

from cereal import car, custom
from openpilot.common.bluepilot import is_bluepilot
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import VCruiseHelper
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import (
  IntelligentCruiseButtonManagement,
)

pytestmark = pytest.mark.skipif(not is_bluepilot(), reason="BluePilot-only behavior")

ButtonType = car.CarState.ButtonEvent.Type


class Truck:
  """The truck's cluster set speed and openpilot's target, frame by frame (10 ms)."""

  def __init__(self, brand: str = "ford", icbm: bool = True, is_metric: bool = False):
    self.helper = VCruiseHelper(car.CarParams(brand=brand, pcmCruise=True),
                                custom.CarParamsSP(pcmCruiseSpeed=not icbm))
    self.is_metric = is_metric
    self.unit = CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS
    self.cluster = 0  # display units, 0 = no set speed shown
    self.enabled = False

  def frame(self, *events) -> None:
    speed = self.cluster * self.unit
    CS = car.CarState(cruiseState={"available": True, "enabled": self.enabled, "speed": speed, "speedCluster": speed},
                      buttonEvents=[{"type": t, "pressed": p} for t, p in events])
    self.helper.update_v_cruise(CS, self.enabled, self.is_metric)

  def frames(self, n: int) -> None:
    for _ in range(n):
      self.frame()

  @property
  def target(self) -> int:
    return round(self.helper.v_cruise_kph * (1 if self.is_metric else CV.KPH_TO_MPH))

  def engage_from_main_off(self, set_speed: int) -> None:
    """Main then Set: the truck reports enabled about 40 ms before it shows the set speed."""
    self.frames(5)
    self.enabled = True
    self.frames(4)
    self.cluster = set_speed
    self.frames(50)

  def press(self, button, steps, release_after: int) -> None:
    self.frame((button, True))
    for i in range(1, release_after):
      self.cluster = dict(steps).get(i, self.cluster)
      self.frame()
    self.frame((button, False))

  def hold(self, button, release_after: int = 70) -> None:
    """A held press: 1 unit about 40 ms after the press, the next multiple of 5 at about 0.5 s."""
    sign = 1 if button == ButtonType.accelCruise else -1
    first = self.cluster + sign
    fifth = (first // 5 + (sign > 0)) * 5 if first % 5 else first
    self.press(button, [(4, first), (50, fifth)], release_after)

  def tap(self, button) -> None:
    sign = 1 if button == ButtonType.accelCruise else -1
    self.press(button, [(4, self.cluster + sign)], 12)


def test_engage_from_main_off_takes_the_cluster_set_speed():
  truck = Truck()
  truck.engage_from_main_off(35)
  assert truck.cluster == 35 and truck.target == 35


def test_held_set_minus_ends_on_the_truck_speed():
  truck = Truck()
  truck.engage_from_main_off(35)
  truck.hold(ButtonType.decelCruise)
  truck.frames(100)
  assert (truck.cluster, truck.target) == (30, 30)


def test_held_set_plus_ends_on_the_truck_speed():
  truck = Truck()
  truck.engage_from_main_off(30)
  truck.hold(ButtonType.accelCruise)
  truck.frames(100)
  assert (truck.cluster, truck.target) == (35, 35)


def test_tap_follows_the_truck_speed():
  truck = Truck()
  truck.engage_from_main_off(35)
  truck.tap(ButtonType.accelCruise)
  truck.frames(100)
  assert (truck.cluster, truck.target) == (36, 36)


def test_other_brand_with_icbm_keeps_sunnypilot_behavior():
  truck = Truck(brand="toyota")
  assert truck.helper.bp_cluster_set_speed is None
  truck.engage_from_main_off(35)
  assert truck.target == 158  # unset


def test_icbm_unset_or_zero_target_holds_the_cluster():
  for target_kph, cluster, is_metric, expected in (
    (255.0, 35, False, 35),
    (145.0, 35, False, 35),
    (0.0, 35, False, 35),
    (0.0, 60, True, 60),
  ):
    icbm = IntelligentCruiseButtonManagement(car.CarParams(brand="ford", pcmCruise=True),
                                             custom.CarParamsSP(pcmCruiseSpeed=False))
    unit = CV.KPH_TO_MS if is_metric else CV.MPH_TO_MS
    CS = car.CarState(cruiseState={"available": True, "enabled": True, "speedCluster": cluster * unit})
    icbm.run(CS, car.CarControl(enabled=True), custom.LongitudinalPlanSP(vTarget=target_kph * CV.KPH_TO_MS), is_metric)
    assert icbm.v_target == expected, (target_kph, cluster, is_metric, icbm.v_target)


def test_icbm_valid_target_passes_through():
  icbm = IntelligentCruiseButtonManagement(car.CarParams(brand="ford", pcmCruise=True),
                                           custom.CarParamsSP(pcmCruiseSpeed=False))
  CS = car.CarState(cruiseState={"available": True, "enabled": True, "speedCluster": 35 * CV.MPH_TO_MS})
  icbm.run(CS, car.CarControl(enabled=True), custom.LongitudinalPlanSP(vTarget=50 * CV.MPH_TO_MS), False)
  assert icbm.v_target == 50
