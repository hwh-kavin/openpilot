"""On-demand diagnostics snapshots for control/communication alerts.

The control loop must stay responsive, so the caller only copies lightweight
message state here.  Process/thread inspection and JSON writing happen in a
background thread.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from openpilot.common.error_log import append_error_log
from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware.hw import Paths

MAX_SNAPSHOTS = 30
CAPTURE_COOLDOWN_S = 3.0
MAX_THREADS_PER_PROCESS = 64

_CAPTURE_LOCK = threading.Lock()
_LAST_CAPTURE: dict[str, float] = {}

_TARGET_PROCESSES = (
  "selfdrived", "controlsd", "card", "manager", "hardwared",
  "camerad", "modeld", "plannerd", "radard", "pandad",
  "loggerd", "encoderd", "ui", "locationd", "paramsd",
  "torqued", "lagd", "calibrationd", "dmonitoringd",
  "dmonitoringmodeld", "athenad", "uploader",
)


def _diagnostics_dir() -> Path:
  override = os.getenv("CONTROL_DIAGNOSTICS_DIR")
  return Path(override) if override else Path(Paths.crash_log_root()) / "diagnostics"


def _read_text(path: str) -> str | None:
  try:
    return Path(path).read_text(errors="ignore").strip()
  except Exception:
    return None


def _read_status(pid: int) -> dict[str, str]:
  text = _read_text(f"/proc/{pid}/status")
  if text is None:
    return {}

  wanted = {
    "Name", "State", "Tgid", "Pid", "PPid", "Threads",
    "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches",
    "Cpus_allowed_list", "Mems_allowed_list",
  }
  status: dict[str, str] = {}
  for line in text.splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    if key in wanted:
      status[key] = value.strip()
  return status


def _read_schedstat(pid: int, tid: int | None = None) -> dict[str, int] | None:
  path = f"/proc/{pid}/schedstat" if tid is None else f"/proc/{pid}/task/{tid}/schedstat"
  text = _read_text(path)
  if text is None:
    return None
  try:
    runtime_ns, wait_ns, timeslices = (int(x) for x in text.split()[:3])
    return {"runtime_ns": runtime_ns, "wait_ns": wait_ns, "timeslices": timeslices}
  except Exception:
    return None


def _read_task_stat(pid: int, tid: int | None = None) -> dict[str, object]:
  path = f"/proc/{pid}/stat" if tid is None else f"/proc/{pid}/task/{tid}/stat"
  text = _read_text(path)
  if text is None:
    return {}
  try:
    # The comm field can contain spaces and parentheses; split after the last ')'.
    fields = text.rsplit(")", 1)[1].split()
    # fields[0] is state (field 3 in proc(5)); processor is field 39.
    processor = int(fields[36]) if len(fields) > 36 else None
    return {"state": fields[0], "processor": processor}
  except Exception:
    return {}


def _read_wchan(pid: int, tid: int | None = None) -> str | None:
  path = f"/proc/{pid}/wchan" if tid is None else f"/proc/{pid}/task/{tid}/wchan"
  return _read_text(path)


def _sched_policy_name(policy: int | None) -> str | None:
  if policy is None:
    return None
  names = {
    getattr(os, "SCHED_OTHER", -1): "other",
    getattr(os, "SCHED_FIFO", -1): "fifo",
    getattr(os, "SCHED_RR", -1): "rr",
    getattr(os, "SCHED_BATCH", -1): "batch",
    getattr(os, "SCHED_IDLE", -1): "idle",
  }
  return names.get(policy, str(policy))


def _process_snapshot(pid: int) -> dict[str, object]:
  import psutil

  out: dict[str, object] = {"pid": pid}
  try:
    p = psutil.Process(pid)
  except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"
    return out

  try:
    out["name"] = p.name()
    out["cmdline"] = " ".join(p.cmdline())[:500]
    out["status"] = p.status()
    out["create_time"] = p.create_time()
    out["num_threads"] = p.num_threads()
    out["cpu_num"] = p.cpu_num()
    out["nice"] = p.nice()
    out["ionice"] = str(p.ionice())
    out["memory_info"] = dict(p.memory_info()._asdict())
    out["cpu_times"] = dict(p.cpu_times()._asdict())
    out["ctx_switches"] = dict(p.num_ctx_switches()._asdict())
  except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"

  try:
    out["cpu_affinity"] = sorted(os.sched_getaffinity(pid))
  except Exception:
    out["cpu_affinity"] = None

  try:
    policy = os.sched_getscheduler(pid)
    out["sched_policy"] = _sched_policy_name(policy)
    out["sched_priority"] = os.sched_getparam(pid).sched_priority
  except Exception:
    out["sched_policy"] = None
    out["sched_priority"] = None

  out["schedstat"] = _read_schedstat(pid)
  out["wchan"] = _read_wchan(pid)
  out["task_stat"] = _read_task_stat(pid)
  out["status_fields"] = _read_status(pid)

  threads: list[dict[str, object]] = []
  try:
    task_dir = Path(f"/proc/{pid}/task")
    thread_ids = sorted((int(d.name) for d in task_dir.iterdir() if d.name.isdigit()))
    for tid in thread_ids[:MAX_THREADS_PER_PROCESS]:
      threads.append({
        "tid": tid,
        "comm": _read_text(f"/proc/{pid}/task/{tid}/comm"),
        "wchan": _read_wchan(pid, tid),
        "schedstat": _read_schedstat(pid, tid),
        "task_stat": _read_task_stat(pid, tid),
      })
  except Exception:
    pass
  out["threads"] = threads
  return out


def _find_process_pids(names: set[str], seed: dict[str, int]) -> dict[str, int]:
  import psutil

  found = {name: pid for name, pid in seed.items() if pid > 0}
  missing = names - set(found)
  if not missing:
    return found

  for proc in psutil.process_iter(["pid", "name", "cmdline"]):
    try:
      pid = proc.info["pid"]
      name = (proc.info["name"] or "").lower()
      cmdline = " ".join(proc.info["cmdline"] or []).lower()
      text = f"{name} {cmdline}"
    except Exception:
      continue

    for target in missing:
      if target in found:
        continue
      if target == "manager":
        matched = ("system.manager.manager" in text or "system/manager/manager" in text) and \
                  "models_manager" not in text and "mapd_manager" not in text
      else:
        matched = target.lower() in text
      if matched:
        found[target] = pid

    if len(found) >= len(names):
      break

  return found


def _base_snapshot(reason: str, sm, events, rk, params) -> dict[str, object]:
  now = time.monotonic()
  services: dict[str, dict[str, object]] = {}
  for service in sm.services:
    recv_time = sm.recv_time.get(service, 0.0)
    services[service] = {
      "alive": bool(sm.alive.get(service, False)),
      "valid": bool(sm.valid.get(service, False)),
      "freq_ok": bool(sm.freq_ok.get(service, False)),
      "recv_age_s": round(now - recv_time, 3) if recv_time else None,
      "recv_frame": sm.recv_frame.get(service),
      "log_mono_time": sm.logMonoTime.get(service),
    }

  sds = sm["selfdriveState"] if "selfdriveState" in sm.services else None
  device = sm["deviceState"] if "deviceState" in sm.services else None
  car = sm["carState"] if "carState" in sm.services else None
  manager = sm["managerState"] if "managerState" in sm.services else None

  snapshot: dict[str, object] = {
    "created": datetime.now(timezone.utc).isoformat(),
    "monotonic": now,
    "reason": reason,
    "pid": os.getpid(),
    "version": params.get("Version"),
    "git_commit": params.get("GitCommit"),
    "git_branch": params.get("GitBranch"),
    "route": params.get("CurrentRoute"),
    "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    "cpu_count": os.cpu_count(),
    "selfdrived_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
    "selfdrived_sched_policy": _sched_policy_name(os.sched_getscheduler(0)) if hasattr(os, "sched_getscheduler") else None,
    "selfdrived_sched_priority": os.sched_getparam(0).sched_priority if hasattr(os, "sched_getparam") else None,
    "sm_frame": sm.frame,
    "services": services,
    "events": [],
    "ratekeeper": None,
    "selfdrive_state": None,
    "device_state": None,
    "panda_states": [],
    "car_state": None,
    "manager_processes": [],
    "pressure": {},
  }

  try:
    snapshot["ratekeeper"] = {
      "avg_dt_ms": round(rk.avg_dt.get_average() * 1000., 3) if rk is not None else None,
      "lagging": bool(rk.lagging) if rk is not None else None,
      "frame": int(rk.frame) if rk is not None else None,
      "remaining_ms": round(rk.remaining * 1000., 3) if rk is not None else None,
    }
  except Exception:
    pass

  try:
    names = []
    for event in events.events:
      try:
        names.append(events.get_event_name(event))
      except Exception:
        names.append(str(event))
    snapshot["events"] = names
  except Exception:
    pass

  if sds is not None:
    snapshot["selfdrive_state"] = {
      "state": str(sds.state),
      "enabled": bool(sds.enabled),
      "active": bool(sds.active),
      "alert_text_1": str(sds.alertText1),
      "alert_text_2": str(sds.alertText2),
      "alert_status": str(sds.alertStatus),
      "alert_size": str(sds.alertSize),
    }

  if device is not None:
    snapshot["device_state"] = {
      "cpu_usage_percent": list(device.cpuUsagePercent),
      "cpu_temp_c": list(device.cpuTempC),
      "gpu_usage_percent": int(device.gpuUsagePercent),
      "gpu_temp_c": list(device.gpuTempC),
      "memory_usage_percent": int(device.memoryUsagePercent),
      "memory_temp_c": float(device.memoryTempC),
      "max_temp_c": float(device.maxTempC),
      "free_space_percent": float(device.freeSpacePercent),
      "thermal_status": str(device.thermalStatus),
      "fan_speed_percent_desired": int(device.fanSpeedPercentDesired),
      "power_draw_w": float(device.powerDrawW),
    }

  try:
    for ps in sm["pandaStates"]:
      snapshot["panda_states"].append({
        "controls_allowed": bool(ps.controlsAllowed),
        "controls_allowed_lateral": bool(ps.controlsAllowedLateral),
        "controls_allowed_longitudinal": bool(ps.controlsAllowedLongitudinal),
        "safety_rx_checks_invalid": bool(ps.safetyRxChecksInvalid),
        "safety_rx_invalid": int(ps.safetyRxInvalid),
        "safety_tx_blocked": int(ps.safetyTxBlocked),
        "rx_buffer_overflow": int(ps.rxBufferOverflow),
        "tx_buffer_overflow": int(ps.txBufferOverflow),
        "fault_status": str(ps.faultStatus),
        "faults": [str(f) for f in ps.faults],
        "safety_model": str(ps.safetyModel),
        "harness_status": str(ps.harnessStatus),
      })
  except Exception:
    pass

  if car is not None:
    snapshot["car_state"] = {
      "v_ego": float(car.vEgo),
      "can_valid": bool(car.canValid),
      "can_timeout": bool(car.canTimeout),
      "can_error_counter": int(car.canErrorCounter),
      "gas_pressed": bool(car.gasPressed),
      "brake_pressed": bool(car.brakePressed),
      "standstill": bool(car.standstill),
      "gear": str(car.gearShifter),
      "cruise_enabled": bool(car.cruiseState.enabled),
      "cruise_available": bool(car.cruiseState.available),
    }

  if manager is not None:
    snapshot["manager_processes"] = [
      {
        "name": str(p.name),
        "pid": int(p.pid),
        "running": bool(p.running),
        "should_be_running": bool(p.shouldBeRunning),
        "exit_code": int(p.exitCode),
      }
      for p in manager.processes
    ]

  for name in ("cpu", "io", "memory"):
    text = _read_text(f"/proc/pressure/{name}")
    if text is not None:
      snapshot["pressure"][name] = text

  return snapshot


def _write_snapshot(snapshot: dict[str, object], process_pids: dict[str, int]) -> None:
  try:
    snapshot["processes"] = {
      name: _process_snapshot(pid) if pid > 0 else {"pid": None, "missing": True}
      for name, pid in sorted(process_pids.items())
    }

    diag_dir = _diagnostics_dir()
    diag_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    reason = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(snapshot.get("reason", "alert")))[:40]
    path = diag_dir / f"{stamp}_{reason}.json"
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
      json.dump(snapshot, f, indent=2, default=str)
    os.replace(tmp_path, path)

    latest = diag_dir / "latest.json"
    try:
      if latest.exists() or latest.is_symlink():
        latest.unlink()
      latest.symlink_to(path.name)
    except OSError:
      pass

    files = sorted((p for p in diag_dir.glob("*.json") if p != latest), key=lambda p: p.stat().st_mtime)
    for old in files[:-MAX_SNAPSHOTS]:
      try:
        old.unlink()
      except OSError:
        pass

    cloudlog.event(
      "controlDiagnostics",
      reason=snapshot.get("reason"),
      path=str(path),
      loadavg=snapshot.get("loadavg"),
      device_cpu=(snapshot.get("device_state") or {}).get("cpu_usage_percent"),
      not_alive=[s for s, v in (snapshot.get("services") or {}).items() if not v.get("alive", False)],
      not_freq_ok=[s for s, v in (snapshot.get("services") or {}).items() if not v.get("freq_ok", False)],
      invalid=[s for s, v in (snapshot.get("services") or {}).items() if not v.get("valid", False)],
      error=True,
    )
    append_error_log(f"control diagnostics {snapshot.get('reason')}: {path}")
  except Exception:
    cloudlog.exception("controlDiagnostics failed")


def capture_diagnostics(reason: str, sm, events, rk, params) -> None:
  """Capture one asynchronous snapshot, rate-limited per reason."""
  now = time.monotonic()
  with _CAPTURE_LOCK:
    last = _LAST_CAPTURE.get(reason, 0.0)
    if now - last < CAPTURE_COOLDOWN_S:
      return
    _LAST_CAPTURE[reason] = now

  try:
    snapshot = _base_snapshot(reason, sm, events, rk, params)
    seed = {
      str(p.name): int(p.pid)
      for p in sm["managerState"].processes
      if "managerState" in sm.services and int(p.pid) > 0
    } if "managerState" in sm.services else {}
    seed["selfdrived"] = os.getpid()
    process_pids = _find_process_pids(set(_TARGET_PROCESSES), seed)
    threading.Thread(target=_write_snapshot, args=(snapshot, process_pids), daemon=True).start()
  except Exception:
    cloudlog.exception("controlDiagnostics capture failed")
