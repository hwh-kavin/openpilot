import pytest

from cereal import custom
from opendbc.car import structs
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource


class MockMpc:
  crash_cnt = 0


class MockSelfDriveState:
  def __init__(self, experimental_mode: bool):
    self.experimentalMode = experimental_mode


@pytest.fixture
def planner():
  return LongitudinalPlannerSP(structs.CarParams(), structs.CarParamsSP(), MockMpc())


def _sm(experimental_mode: bool):
  return {'selfdriveState': MockSelfDriveState(experimental_mode)}


def test_is_e2e_false_when_scc_vision_active(planner):
  planner.scc.vision.is_active = True
  assert not planner.is_e2e(_sm(experimental_mode=True))


def test_is_e2e_false_when_scc_vision_source_selected(planner):
  planner.source = LongitudinalPlanSource.sccVision
  assert not planner.is_e2e(_sm(experimental_mode=True))


def test_is_e2e_true_in_experimental_mode(planner):
  assert planner.is_e2e(_sm(experimental_mode=True))
