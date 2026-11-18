import pytest

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.selfdrived.selfdrived import MODELD_LAG_DROP_PERC_THRESHOLD


def frame_drop_percent_series(dropped_per_iteration):
  """Replay modeld's dropped-frame metric (modeld.py) over a drop sequence.

  modeld publishes frameDropPerc = 100 * frames_dropped / (1 + frames_dropped) where
  frames_dropped is a FirstOrderFilter with RC=10 model frames.
  """
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  out = []
  for dropped in dropped_per_iteration:
    frames_dropped = frame_dropped_filter.update(min(dropped, 10))
    out.append(100 * frames_dropped / (1 + frames_dropped))
  return out


def test_short_camera_backlog_stays_below_lag_threshold():
  # A one-off 3-frame camera backlog (150 ms stall at 20 Hz) is a harmless hiccup.
  # The metric lingers, so the threshold must sit above this burst's peak.
  series = frame_drop_percent_series([3] + [0] * 400)
  assert max(series) < MODELD_LAG_DROP_PERC_THRESHOLD


def test_sustained_frame_drops_exceed_lag_threshold():
  # modeld consistently missing frames is real lag and must still be flagged.
  series = frame_drop_percent_series([3] * 400)
  assert series[-1] > MODELD_LAG_DROP_PERC_THRESHOLD


def test_threshold_is_reachable_by_a_single_max_burst():
  # Sanity guard: modeld clamps each iteration's drop count to 10, so a threshold above
  # the maximum single-burst reading could never fire at all.
  series = frame_drop_percent_series([10] + [0] * 400)
  assert max(series) > MODELD_LAG_DROP_PERC_THRESHOLD


@pytest.mark.parametrize("dropped", [1, 2])
def test_small_single_bursts_never_flag(dropped):
  series = frame_drop_percent_series([dropped] + [0] * 400)
  assert max(series) < MODELD_LAG_DROP_PERC_THRESHOLD
