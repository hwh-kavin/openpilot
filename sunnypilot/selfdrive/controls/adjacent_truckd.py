#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

BluePilot: adjacent-lane large vehicle avoidance daemon.

This runs as its own low-priority process and is deliberately kept OUT of the controls realtime
path. It only *observes* `liveTracks` + `modelV2` + `carState` and publishes a suggestion into
Params; nothing in controlsd/modeld/radard depends on it, so a slow or crashed instance can never
affect vehicle control. If it does fail, it clears its own enable toggle and exits cleanly, so the
manager no longer expects it to run and openpilot engagement is never blocked by it.

Output is written to the `AdjacentTruckState` param (JSON) at 2 Hz. A future consumer (e.g. the
Ford lateral extension) can pick it up there; wiring the actuator is intentionally out of scope
for this change.

Enable with the "Avoid Large Vehicles in Adjacent Lanes" toggle on the Cruise settings page
(param `AdjacentTruckAvoid`). The manager only starts this process while that toggle is on.
"""

import numpy as np

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, set_core_affinity
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.adjacent_truck import (
  AdjacentTruckDetector, PHASE_IDLE, points_from_live_tracks,
)

PUBLISH_PERIOD = 0.5   # s, a Param write is a syscall - 2 Hz is plenty for a consumer
TOGGLE_PERIOD = 1.0    # s, re-read the enable flag; the manager also gates process start on it


def lane_arrays(model_v2):
  """Stack the left/right lane lines into a (3, N) array matching AdjacentTruckDetector."""
  lines = model_v2.laneLines
  if len(lines) <= 2:
    return None, None

  n = len(lines[0].x)
  if n < 2:
    return None, None

  arr = np.empty((3, n), dtype=float)
  arr[0] = np.asarray(lines[0].x, dtype=float)
  arr[1] = np.asarray(lines[1].y, dtype=float)
  arr[2] = np.asarray(lines[2].y, dtype=float)
  return arr, list(model_v2.laneLineProbs)


def main() -> None:
  """Run the detector; a failure here must never stop openpilot from engaging.

  Any unexpected error clears the enable toggle (the manager then stops this process instead of
  reporting it as "should be running") and exits cleanly. Without that, a crashed optional daemon
  stays in managerState as shouldBeRunning=True / running=False, which raises the "Process Not
  Running" soft-disable and makes openpilot un-engageable.
  """
  params: Params | None = None
  try:
    params = Params()
    try:
      # Observation-only daemon: pin it to a spare core but stay on the default scheduler, so it
      # can never compete with the controls realtime processes. Best effort - a missing/offline
      # core is not an error.
      set_core_affinity([4])
    except Exception:
      pass
    _run(params)
  except KeyboardInterrupt:
    raise
  except Exception:
    cloudlog.exception("adjacent_truckd failed; clearing AdjacentTruckAvoid so openpilot is unaffected")
    if params is not None:
      try:
        params.put_bool("AdjacentTruckAvoid", False, block=True)
      except Exception:
        cloudlog.exception("adjacent_truckd could not clear AdjacentTruckAvoid")
    raise SystemExit(0)


def _run(params: Params) -> None:
  detector = AdjacentTruckDetector()

  sm = messaging.SubMaster(['liveTracks', 'modelV2', 'carState', 'carControl'], poll='modelV2')

  enabled = params.get_bool("AdjacentTruckAvoid")
  toggle_t = 0.0
  publish_t = 0.0
  last_phase = PHASE_IDLE
  cloudlog.info("adjacent_truckd started")

  while True:
    sm.update()

    toggle_t += DT_MDL
    if toggle_t >= TOGGLE_PERIOD:
      toggle_t = 0.0
      enabled = params.get_bool("AdjacentTruckAvoid")

    points = points_from_live_tracks(sm['liveTracks']) if sm.valid['liveTracks'] else []
    lane_lines, lane_probs = lane_arrays(sm['modelV2'])
    engaged = bool(sm['carControl'].enabled and sm['carControl'].latActive)

    det = detector.update(DT_MDL, sm['carState'].vEgo, points, lane_lines, lane_probs, engaged, enabled)

    if det.phase != last_phase:
      if det.phase != PHASE_IDLE:
        cloudlog.info(f"adjacent_truckd: {det.phase} side={det.side:+d} span={det.span:.1f}m "
                      f"rcs={det.rcs:.1f}dBsm dRel={det.d_rel:.1f}m vTruck={det.v_truck:.1f}m/s")
      last_phase = det.phase

    publish_t += DT_MDL
    if publish_t >= PUBLISH_PERIOD:
      publish_t = 0.0
      params.put("AdjacentTruckState", {
        "active": bool(det.active),
        "phase": det.phase,
        "side": int(det.side),
        "offsetTarget": float(det.offset_target),
        "trackId": int(det.track_id),
        "dRel": float(det.d_rel),
        "yRel": float(det.y_rel),
        "lateral": float(det.lateral),
        "span": float(det.span),
        "rcs": float(det.rcs),
        "vTruck": float(det.v_truck),
        "holdRemaining": float(det.hold_remaining),
        "vEgo": float(sm['carState'].vEgo),
      })


if __name__ == "__main__":
  main()
