"""Fail-safe behavior of the adjacent-lane truck daemon.

A broken optional feature must never leave openpilot un-engageable: the daemon clears its own
enable toggle and exits cleanly instead of staying in managerState as shouldBeRunning=True /
running=False ("Process Not Running" soft-disable).
"""
import pytest

import sunnypilot.selfdrive.controls.adjacent_truckd as atd


class FakeParams:
  def __init__(self):
    self.bools = {}

  def put_bool(self, key, value, block=False):
    self.bools[key] = value


def test_crash_clears_toggle_and_exits_clean(monkeypatch):
  fake = FakeParams()
  monkeypatch.setattr(atd, "Params", lambda: fake)
  monkeypatch.setattr(atd, "set_core_affinity", lambda cores: None)

  def boom(params):
    raise RuntimeError("boom")

  monkeypatch.setattr(atd, "_run", boom)

  with pytest.raises(SystemExit) as exc:
    atd.main()

  assert exc.value.code == 0
  assert fake.bools.get("AdjacentTruckAvoid") is False


def test_affinity_failure_is_not_fatal(monkeypatch):
  fake = FakeParams()
  ran = []

  def bad_affinity(cores):
    raise OSError(22, "Invalid argument")

  monkeypatch.setattr(atd, "Params", lambda: fake)
  monkeypatch.setattr(atd, "set_core_affinity", bad_affinity)
  monkeypatch.setattr(atd, "_run", lambda params: ran.append(params))

  atd.main()

  assert ran, "detector loop must still start when pinning fails"
  assert not fake.bools, "a clean run must not touch the toggle"
