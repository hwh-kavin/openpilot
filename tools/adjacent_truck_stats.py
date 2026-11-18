#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

BluePilot: offline calibration for the adjacent-lane large vehicle detector.

Reduces a drive log to the three numbers the detector's thresholds hang on, so they can be measured
instead of guessed:

  * chain span  -> SPAN_BIG      and CHAIN_GAP / CHAIN_LATERAL
  * cluster RCS -> RCS_BIG       and RCS_RANGE
  * truck speed -> V_TRUCK_MIN   (and sanity of the pass timing)

Usage:
  tools/adjacent_truck_stats.py 00000176--e491d6e340--20
  tools/adjacent_truck_stats.py /data/media/0/realdata/00000176--e491d6e340--20/qlog.zst
  tools/adjacent_truck_stats.py <route> --csv out.csv --plot out.png

Note: the RCS columns only exist in logs recorded with a build that carries the cluster amplitude
into RadarPoint.yvRel (see opendbc/car/ford/radar_interface.py). On older logs every RCS value is
NaN and the script says so; the chain-span columns still work because they only need dRel.
"""

import argparse
import csv
import os
import sys

import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib import adjacent_truck as at
from openpilot.tools.lib.logreader import LogReader, ReadMode

# dRel buckets used for the RCS table; the detector only trusts RCS inside RCS_RANGE
RCS_BUCKETS = [(10.0, 15.0), (15.0, 25.0), (25.0, 35.0), (35.0, 45.0), (45.0, 55.0), (55.0, 70.0)]
CHAIN_GAP_SWEEP = (3.0, 4.0, 4.5, 5.0, 6.0)
PCTS = (50, 75, 90, 95, 99)


def resolve_source(src: str) -> str:
  """Accept a segment directory as well as a log path."""
  if not os.path.isdir(src):
    return src
  for name in ("qlog.zst", "qlog.bz2", "rlog.zst", "rlog.bz2"):
    path = os.path.join(src, name)
    if os.path.exists(path):
      return path
  raise FileNotFoundError(f"no qlog/rlog found in {src}")


def lane_arrays(model_v2):
  """(3, N) lane-line array + probs, matching AdjacentTruckDetector's expectations."""
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


def collect(sources, min_speed: float):
  """Walk the logs and return a list of per-frame adjacent-lane measurements."""
  rows = []
  frames = 0

  for src in sources:
    lr = LogReader(resolve_source(src), default_mode=ReadMode.AUTO)
    model = None
    v_ego = 0.0

    for msg in lr:
      which = msg.which()
      if which == "modelV2":
        m = msg.modelV2
        if m is not None:
          model = m
        continue
      if which == "carState":
        v_ego = msg.carState.vEgo
        continue
      if which != "liveTracks" or model is None:
        continue

      frames += 1
      if v_ego < min_speed:
        continue

      lanes, probs = lane_arrays(model)
      if lanes is None:
        continue

      points = [(p.trackId, p.dRel, p.yRel, p.vRel, p.yvRel) for p in msg.liveTracks.points]
      metrics = at.frame_metrics(points, lanes, probs, v_ego, gap_sweep=CHAIN_GAP_SWEEP)
      for side, m in metrics.items():
        rows.append({
          "side": side,
          "d_rel": m["d_rel"],
          "lateral": m["lateral"],
          "chain": m["chain"],
          "rcs": m["rcs"],
          "rcs_trusted": m["rcs_trusted"],
          "v_truck": m["v_truck"],
          "n_tracks": m["n_tracks"],
          "chain_gap_sweep": m["chain_sweep"],
        })

  return rows, frames


def pct_block(values, label: str):
  if not len(values):
    print(f"  {label:<22} (no samples)")
    return
  q = np.percentile(values, PCTS)
  body = "  ".join(f"p{p}={v:6.1f}" for p, v in zip(PCTS, q))
  print(f"  {label:<22} n={len(values):6d}  min={np.min(values):6.1f}  {body}  max={np.max(values):6.1f}")


def report(rows, frames: int):
  print()
  print(f"liveTracks frames inspected: {frames}, adjacent-lane frame-sides: {len(rows)}")
  if not rows:
    if frames == 0:
      print("No liveTracks message was seen at all. Logs from before the liveTracks service existed "
            "carry drivingModelData instead and cannot be used here - record a new drive.")
    else:
      print("liveTracks was present, but no track ever fell in the adjacent-lane band "
            f"({at.NEIGHBOR_LO:.2f}..{at.NEIGHBOR_HI:.2f} lane widths). Either the radar was "
            "unavailable or the drive had no adjacent traffic.")
    return

  chain = np.array([r["chain"] for r in rows])
  rcs_all = np.array([r["rcs"] for r in rows])
  rcs_ok = np.array([r["rcs_trusted"] for r in rows])
  v_truck = np.array([r["v_truck"] for r in rows])
  d_rel = np.array([r["d_rel"] for r in rows])
  n_tracks = np.array([r["n_tracks"] for r in rows])

  print()
  print("== chain span (m) ==")
  pct_block(chain, "all")
  pct_block(chain[d_rel <= at.D_FAR], "within D_FAR")
  for thr in (5.0, at.SPAN_BIG, 10.0):
    frac = float(np.mean(chain >= thr)) * 100.0
    print(f"  chain >= {thr:4.1f} m : {frac:5.2f}% of adjacent-lane frames")

  print()
  print(f"== tracks per frame-side (radar splits one object into several) ==")
  pct_block(n_tracks, "n_tracks")

  print()
  print("== CHAIN_GAP sensitivity (mean / p95 of chain span) ==")
  for g in CHAIN_GAP_SWEEP:
    vals = np.array([r["chain_gap_sweep"][g] for r in rows])
    print(f"  gap={g:4.1f} m : mean={vals.mean():5.2f}  p95={np.percentile(vals, 95):5.2f}  "
          f"frac>=7m={float(np.mean(vals >= at.SPAN_BIG)) * 100:5.2f}%")

  print()
  print("== RCS (dBsm) by dRel bucket ==")
  if float(np.max(np.abs(rcs_all))) < 1e-6:
    print("  All RCS values are 0/NaN - this log predates the cluster-amplitude carrier "
          "(RadarPoint.yvRel). Re-record with the current build to calibrate RCS_BIG.")
  else:
    for lo, hi in RCS_BUCKETS:
      mask = (d_rel >= lo) & (d_rel < hi)
      trusted = float(np.mean(mask & (rcs_ok > 0.0))) * 100.0 if np.any(mask) else 0.0
      print(f"  dRel [{lo:4.1f},{hi:5.1f}) m  n={int(mask.sum()):6d}  trusted={trusted:5.1f}%")
      pct_block(rcs_ok[mask & (rcs_ok > 0.0)], "rcs_trusted")

  print()
  print("== truck speed (m/s) ==")
  pct_block(v_truck, "all")

  print()
  print("== how the current threshold would split things ==")
  big = (chain >= at.SPAN_BIG) | (rcs_ok >= at.RCS_BIG)
  moving = np.abs(v_truck) > at.V_TRUCK_MIN
  print(f"  span-or-rcs evidence      : {np.sum(big):6d} frames ({np.mean(big) * 100:5.2f}%)")
  print(f"  ...and moving             : {np.sum(big & moving):6d} frames "
        f"({np.mean(big & moving) * 100:5.2f}%)")
  print(f"  static with evidence     : {np.sum(big & ~moving):6d} frames "
        f"(rejected by V_TRUCK_MIN - guardrails/walls)")

  frac_big = float(np.mean(big & moving)) * 100.0
  if frac_big > 25.0:
    print("  !! more than a quarter of adjacent-lane frames look like large vehicles - the "
          "thresholds are almost certainly too loose")
  elif frac_big < 0.5:
    print("  ~ very few hits: either you rarely passed a truck, or the thresholds are too tight")


def dump_csv(rows, path: str):
  keys = ["side", "d_rel", "lateral", "chain", "rcs", "rcs_trusted", "v_truck", "n_tracks"]
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
      w.writerow({k: r[k] for k in keys})
  print(f"csv written: {path}")


def dump_plot(rows, path: str):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  chain = np.array([r["chain"] for r in rows])
  rcs = np.array([r["rcs_trusted"] for r in rows])
  d_rel = np.array([r["d_rel"] for r in rows])
  v_truck = np.array([r["v_truck"] for r in rows])

  fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

  axes[0].hist(chain, bins=40)
  axes[0].axvline(at.SPAN_BIG, color="r", ls="--", label=f"SPAN_BIG={at.SPAN_BIG}")
  axes[0].set_title("chain span")
  axes[0].set_xlabel("m")
  axes[0].legend()

  m = rcs > 0.0
  if np.any(m):
    axes[1].scatter(d_rel[m], rcs[m], s=4, alpha=0.35)
    axes[1].axhline(at.RCS_BIG, color="r", ls="--", label=f"RCS_BIG={at.RCS_BIG}")
  axes[1].axvspan(*at.RCS_RANGE, color="g", alpha=0.08, label="RCS_RANGE")
  axes[1].set_title("trusted RCS vs range")
  axes[1].set_xlabel("dRel [m]")
  axes[1].set_ylabel("dBsm")
  axes[1].legend()

  axes[2].hist(v_truck, bins=40)
  axes[2].axvline(at.V_TRUCK_MIN, color="r", ls="--", label=f"V_TRUCK_MIN={at.V_TRUCK_MIN}")
  axes[2].axvline(-at.V_TRUCK_MIN, color="r", ls="--")
  axes[2].set_title("estimated truck speed")
  axes[2].set_xlabel("m/s")
  axes[2].legend()

  fig.tight_layout()
  fig.savefig(path, dpi=110)
  print(f"plot written: {path}")


def synthetic_rows(n: int = 6000, seed: int = 0):
  """Synthetic measurements shaped like the real thing, so the report and plot paths can be checked
  without a drive log (and in CI)."""
  rng = np.random.default_rng(seed)
  rows = []
  for _ in range(n):
    d = float(rng.uniform(12.0, 70.0))
    in_band = at.RCS_RANGE[0] <= d <= at.RCS_RANGE[1]
    big = rng.random() < 0.10
    static = rng.random() < 0.05
    chain = float(rng.uniform(8.0, 16.0) if big else rng.uniform(0.0, 6.0))
    rcs = float(rng.normal(30.0 if big else 14.0, 3.0))
    v_truck = 0.0 if static else float(rng.normal(22.0, 5.0))
    rows.append({
      "side": 1 if rng.random() < 0.5 else -1,
      "d_rel": d,
      "lateral": float(rng.uniform(2.2, 4.5)),
      "chain": chain,
      "rcs": rcs,
      "rcs_trusted": rcs if in_band else 0.0,
      "v_truck": v_truck,
      "n_tracks": int(rng.integers(1, 7)),
      "chain_gap_sweep": {g: chain * (1.0 - 0.05 * abs(g - at.CHAIN_GAP)) for g in CHAIN_GAP_SWEEP},
    })
  return rows, n


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[8] if __doc__ else "",
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("sources", nargs="*", help="route, segment, or qlog/rlog path")
  parser.add_argument("--min-speed", type=float, default=at.MIN_SPEED,
                      help="ignore frames below this ego speed (default: detector MIN_SPEED)")
  parser.add_argument("--csv", help="write per-frame rows to this CSV")
  parser.add_argument("--plot", help="write histograms to this PNG")
  parser.add_argument("--selftest", action="store_true",
                      help="run on synthetic data instead of a log (checks the report/plot paths)")
  args = parser.parse_args(argv)

  if args.selftest:
    rows, frames = synthetic_rows()
  else:
    if not args.sources:
      parser.error("at least one source is required unless --selftest is used")
    try:
      rows, frames = collect(args.sources, args.min_speed)
    except Exception as e:  # noqa: BLE001 - a bad path should not traceback at the user
      print(f"failed to read logs: {e}", file=sys.stderr)
      return 1

  report(rows, frames)
  if args.csv and rows:
    dump_csv(rows, args.csv)
  if args.plot and rows:
    dump_plot(rows, args.plot)
  return 0


if __name__ == "__main__":
  sys.exit(main())
