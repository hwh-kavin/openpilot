from types import SimpleNamespace

from cereal import log

from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.radar_fallback import (
  HIGHWAY_RADAR_FALLBACK_SPEED_MS,
  highway_radar_fallback_enabled,
  radar_state_degraded,
)
from openpilot.selfdrive.selfdrived.events import Events, ET


def test_highway_radar_fallback_requires_ford_and_speed():
  CP = SimpleNamespace(brand="ford")
  at_threshold = HIGHWAY_RADAR_FALLBACK_SPEED_MS

  assert highway_radar_fallback_enabled(CP, at_threshold)
  assert not highway_radar_fallback_enabled(CP, at_threshold - 0.1)
  assert not highway_radar_fallback_enabled(None, at_threshold)
  assert not highway_radar_fallback_enabled(SimpleNamespace(brand="toyota"), at_threshold)


def test_radar_state_degraded_detects_missing_and_errors():
  assert radar_state_degraded(None)

  healthy = log.RadarState.new_message()
  assert not radar_state_degraded(healthy)

  degraded = log.RadarState.new_message()
  degraded.radarErrors.radarUnavailableTemporary = True
  assert radar_state_degraded(degraded)


def test_radar_degraded_event_is_warning_only():
  events = Events()
  events.add(log.OnroadEvent.EventName.radarDegraded)

  assert events.contains(ET.WARNING)
  assert not events.contains(ET.SOFT_DISABLE)
  assert not events.contains(ET.IMMEDIATE_DISABLE)
  assert not events.contains(ET.NO_ENTRY)
