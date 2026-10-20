import json
import os
import time
from types import SimpleNamespace

from openpilot.selfdrive.selfdrived.diagnostics import (
  _diagnostics_dir,
  _process_snapshot,
  _read_task_stat,
  _write_snapshot,
  capture_diagnostics,
)


def test_process_snapshot_contains_scheduling_fields():
  snap = _process_snapshot(os.getpid())

  assert snap["pid"] == os.getpid()
  assert "cpu_affinity" in snap
  assert "sched_policy" in snap
  assert "sched_priority" in snap
  assert "schedstat" in snap
  assert "wchan" in snap
  assert snap["threads"]


def test_task_stat_can_report_processor():
  stat = _read_task_stat(os.getpid())
  assert stat.get("state")
  assert "processor" in stat


def test_write_snapshot_creates_json(tmp_path, monkeypatch):
  monkeypatch.setenv("CONTROL_DIAGNOSTICS_DIR", str(tmp_path))
  assert _diagnostics_dir() == tmp_path

  _write_snapshot(
    {
      "created": "2026-10-18T00:00:00+00:00",
      "reason": "unit_test",
      "services": {"deviceState": {"alive": False, "valid": False, "freq_ok": False}},
      "device_state": {"cpu_usage_percent": [1, 2, 3]},
    },
    {"selfdrived": os.getpid()},
  )

  files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
  assert len(files) == 1
  assert (tmp_path / "latest.json").exists()

  data = json.loads(files[0].read_text())
  assert data["reason"] == "unit_test"
  assert data["processes"]["selfdrived"]["pid"] == os.getpid()
  assert json.loads((tmp_path / "latest.json").read_text())["reason"] == "unit_test"


def test_capture_diagnostics_end_to_end(tmp_path, monkeypatch):
  monkeypatch.setenv("CONTROL_DIAGNOSTICS_DIR", str(tmp_path))

  class FakeSM:
    services = ["selfdriveState", "deviceState", "pandaStates", "carState", "managerState"]
    alive = {s: True for s in services}
    valid = {s: True for s in services}
    freq_ok = {s: True for s in services}
    recv_time = {s: time.monotonic() for s in services}
    recv_frame = {s: 1 for s in services}
    logMonoTime = {s: 1 for s in services}
    frame = 1

    def __init__(self):
      self._data = {
        "selfdriveState": SimpleNamespace(state="enabled", enabled=True, active=True,
                                           alertText1="", alertText2="", alertStatus="normal", alertSize="none"),
        "deviceState": SimpleNamespace(cpuUsagePercent=[1] * 8, cpuTempC=[40.] * 8,
                                        gpuUsagePercent=1, gpuTempC=[40.], memoryUsagePercent=20,
                                        memoryTempC=30., maxTempC=40., freeSpacePercent=50.,
                                        thermalStatus="ok", fanSpeedPercentDesired=10, powerDrawW=1.),
        "pandaStates": [SimpleNamespace(controlsAllowed=True, controlsAllowedLateral=True,
                                        controlsAllowedLongitudinal=True, safetyRxChecksInvalid=False,
                                        safetyRxInvalid=0, safetyTxBlocked=0, rxBufferOverflow=0,
                                        txBufferOverflow=0, faultStatus="none", faults=[],
                                        safetyModel="ford", harnessStatus="normal")],
        "carState": SimpleNamespace(vEgo=10., canValid=True, canTimeout=False, canErrorCounter=0,
                                    gasPressed=False, brakePressed=False, standstill=False,
                                    gearShifter="drive", cruiseState=SimpleNamespace(enabled=True, available=True)),
        "managerState": SimpleNamespace(processes=[SimpleNamespace(name="selfdrived", pid=os.getpid(),
                                                                    running=True, shouldBeRunning=True, exitCode=0)]),
      }

    def __getitem__(self, key):
      return self._data[key]

  events = SimpleNamespace(events=[], get_event_name=lambda e: str(e))
  rk = SimpleNamespace(avg_dt=SimpleNamespace(get_average=lambda: 0.01), lagging=False, frame=1, remaining=0.)
  params = SimpleNamespace(get=lambda key: None)

  capture_diagnostics("unit_capture", FakeSM(), events, rk, params)

  deadline = time.monotonic() + 5.0
  while time.monotonic() < deadline and not (tmp_path / "latest.json").exists():
    time.sleep(0.02)

  assert (tmp_path / "latest.json").exists()
  data = json.loads((tmp_path / "latest.json").read_text())
  assert data["reason"] == "unit_capture"
  assert data["services"]["deviceState"]["alive"] is True
  assert data["processes"]["selfdrived"]["pid"] == os.getpid()
