from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.radar_lead_filter import RadarLeadFilter, lead_from_vision
from openpilot.selfdrive.controls.radard import RADAR_TO_CAMERA


class _NoRadarErrors:
  def to_dict(self):
    return {}


def make_lead(status=True, radar=True, dRel=120.0, vLead=0.0, vRel=-25.0,
              aLeadK=0.0, aLeadTau=1.5, modelProb=0.95, yRel=0.0, radarTrackId=1):
  return SimpleNamespace(status=status, radar=radar, dRel=dRel, vLead=vLead, vRel=vRel,
                         aLeadK=aLeadK, aLeadTau=aLeadTau, modelProb=modelProb,
                         yRel=yRel, radarTrackId=radarTrackId)


def make_radar_state(lead_one, lead_two=None):
  return SimpleNamespace(leadOne=lead_one,
                         leadTwo=lead_two if lead_two is not None else make_lead(status=False),
                         radarErrors=_NoRadarErrors())


def make_vision_lead(prob=0.95, x=120.0, v=0.0, a=0.0, y=0.0):
  return SimpleNamespace(prob=prob, x=[x], y=[y], v=[v], a=[a])


class FakeSubMaster:
  def __init__(self, radar_state, v_ego, vision_lead=None, radar_valid=True):
    self.valid = {'radarState': radar_valid, 'modelV2': True}
    self.data = {
      'radarState': radar_state,
      'carState': SimpleNamespace(vEgo=v_ego),
      'modelV2': SimpleNamespace(leadsV3=[vision_lead] if vision_lead is not None else []),
    }

  def __getitem__(self, key):
    return self.data[key]


class PlannerStub:
  """Minimal collaborator for LongitudinalPlanner._mpc_radar_state (real method under test)."""
  _is_ford = False
  _fusion_params = None
  _fusion_enabled_cached = False
  _fusion_checked_t = 0.0

  def __init__(self, radar_lead_active):
    self._radar_lead_active = radar_lead_active
    self._radar_lead_filter = RadarLeadFilter()
    self._mpc_lead_status = False
    self._mpc_lead_d_rel = float('inf')


def select_lead_source(stub, sm):
  return LongitudinalPlanner._mpc_radar_state(stub, sm)


# --------------------------------------------------------------------------
# P1: vision lead must carry radard-equivalent semantics
# --------------------------------------------------------------------------

def test_lead_from_vision_reports_radar_frame_distance():
  out = lead_from_vision(make_vision_lead(x=120.0), v_ego=25.0)
  assert out.dRel == pytest.approx(120.0 - RADAR_TO_CAMERA)


def test_lead_from_vision_reports_relative_speed():
  # Previously vRel fell back to the dataclass default of 0.0, which silently disabled
  # every relative-speed / TTC consumer on the vision path.
  out = lead_from_vision(make_vision_lead(x=120.0, v=2.0), v_ego=25.0)
  assert out.vLead == pytest.approx(2.0)
  assert out.vRel == pytest.approx(2.0 - 25.0)


def test_lead_from_vision_uses_vision_accel_decay():
  # radard's get_RadarState_from_vision uses 0.3, not the radar track value of 1.5.
  out = lead_from_vision(make_vision_lead(a=-1.0), v_ego=10.0)
  assert out.aLeadTau == pytest.approx(0.3)
  assert out.aLeadK == pytest.approx(-1.0)


def test_lead_from_vision_without_lead_is_inactive():
  assert lead_from_vision(None, v_ego=10.0).status is False


# --------------------------------------------------------------------------
# P0: leadOne source selection by speed band
# --------------------------------------------------------------------------

def test_high_speed_prefers_radard_fused_lead():
  # Above the radar band, a radar-matched lead must come from radard (which did the
  # path-based association), not from the raw vision lead.
  lead_one = make_lead(radar=True, dRel=150.0, vLead=0.0, vRel=-25.0, aLeadTau=0.77)
  radar_state = make_radar_state(lead_one)
  sm = FakeSubMaster(radar_state, v_ego=25.0, vision_lead=make_vision_lead(x=150.0, v=0.0))
  stub = PlannerStub(radar_lead_active=False)

  out = select_lead_source(stub, sm)

  assert out is radar_state
  assert stub._mpc_lead_status is True
  assert stub._mpc_lead_d_rel == pytest.approx(150.0)


def test_high_speed_falls_back_to_vision_when_radar_not_matched():
  # radard reported a vision-only lead (no radar track associated).
  lead_one = make_lead(radar=False, dRel=118.48, vLead=0.0, vRel=-25.0)
  radar_state = make_radar_state(lead_one)
  sm = FakeSubMaster(radar_state, v_ego=25.0, vision_lead=make_vision_lead(x=120.0, v=0.0))
  stub = PlannerStub(radar_lead_active=False)

  out = select_lead_source(stub, sm)

  assert out.leadOne.radar is False
  assert out.leadOne.dRel == pytest.approx(120.0 - RADAR_TO_CAMERA)


def test_high_speed_falls_back_to_vision_when_radar_degraded():
  lead_one = make_lead(radar=True, dRel=150.0)
  radar_state = make_radar_state(lead_one)
  sm = FakeSubMaster(radar_state, v_ego=25.0, vision_lead=make_vision_lead(x=120.0), radar_valid=False)
  stub = PlannerStub(radar_lead_active=False)

  out = select_lead_source(stub, sm)

  assert out.leadOne.radar is False
  assert out.leadOne.dRel == pytest.approx(120.0 - RADAR_TO_CAMERA)


def test_low_speed_still_uses_radar_band_filter():
  # Inside the radar-primary band the lane-convergence filter still owns the lead.
  lead_one = make_lead(radar=True, dRel=40.0, vLead=0.0, vRel=-10.0)
  radar_state = make_radar_state(lead_one)
  sm = FakeSubMaster(radar_state, v_ego=10.0, vision_lead=make_vision_lead(x=41.52, v=0.0))
  stub = PlannerStub(radar_lead_active=True)
  stub._is_ford = True
  stub._fusion_enabled_cached = True

  out = select_lead_source(stub, sm)

  assert out.leadOne.radar is True
  assert out.leadOne.dRel == pytest.approx(40.0)
