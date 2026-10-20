"""BluePilot: Ford high-speed radar fallback policy.

Above the radar-primary speed band, longitudinal control is vision-first.  Radar
faults in that band should degrade to a warning instead of disengaging when the
vision path is healthy.  Low-speed/radar-primary behavior is unchanged.
"""

from openpilot.common.constants import CV

# Matches RADAR_LEAD_EXIT_V_MS in longitudinal_planner.py.  Keep the 60-62 km/h
# band as the existing hysteresis region; fallback only applies at/above this.
HIGHWAY_RADAR_FALLBACK_SPEED_MS = 62.0 * CV.KPH_TO_MS


def highway_radar_fallback_enabled(CP, v_ego: float) -> bool:
  """Return True when the Ford highway vision-first policy applies.

  This is intentionally hardcoded: above the radar-primary band, radar faults
  must not disengage vision-first longitudinal control.
  """
  if CP is None or getattr(CP, "brand", None) != "ford":
    return False

  try:
    return float(v_ego) >= HIGHWAY_RADAR_FALLBACK_SPEED_MS
  except (TypeError, ValueError):
    return False


def radar_state_degraded(radar_state) -> bool:
  """True when radar data is missing, invalid, or reports any error."""
  if radar_state is None:
    return True
  try:
    return any(radar_state.radarErrors.to_dict().values())
  except Exception:
    return True
