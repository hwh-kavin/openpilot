"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
from types import SimpleNamespace

from openpilot.selfdrive.ui.sunnypilot import model_info as mi
from openpilot.sunnypilot.models.model_name import DEFAULT_MODEL


class _Params:
  def __init__(self, values=None):
    self.values = values or {}

  def get(self, key, *args, **kwargs):
    return self.values.get(key)


def _fake_state(monkeypatch, values=None, bundles=None, cache=None):
  state = SimpleNamespace(params=_Params(values),
                          sm={"modelManagerSP": SimpleNamespace(availableBundles=bundles or [])})
  monkeypatch.setattr(mi, "ui_state", state)
  if cache is not None:
    monkeypatch.setattr(mi, "get_cached_bundles", lambda params, source: cache)
  return state


def test_chestnut_unavailable_on_this_fork(monkeypatch):
  _fake_state(monkeypatch)
  assert mi.chestnut_supported() is False
  assert mi.active_source() == "qcom"
  assert mi.big_model_state() is None
  assert mi.bundles_for_source("chestnut") == []


def test_carrying_model_falls_back_to_default(monkeypatch):
  _fake_state(monkeypatch)
  assert mi.carrying_model() == ("qcom", f"{DEFAULT_MODEL} (Default)", f"{DEFAULT_MODEL} (Default)")


def test_model_info_names(monkeypatch):
  _fake_state(monkeypatch)
  source, active, other = mi.model_info()
  assert source == "qcom"
  assert active == f"{DEFAULT_MODEL} (Default)"
  assert other  # the big slot default is still reported


def test_bundles_for_source_uses_live_manager_for_active_source(monkeypatch):
  bundles = [SimpleNamespace(ref="abc")]
  _fake_state(monkeypatch, bundles=bundles)
  assert mi.bundles_for_source("qcom") == bundles


def test_queued_name_returns_other_pending_bundle(monkeypatch):
  pending = SimpleNamespace(ref="pending", internalName="pending-model")
  _fake_state(monkeypatch, values={"ModelManager_DownloadRef": "pending"}, bundles=[pending])
  assert mi.queued_name("running") == "pending-model"
  # the running bundle is not reported as queued
  assert mi.queued_name("pending") is None


def test_refresh_in_progress_window(monkeypatch):
  _fake_state(monkeypatch, values={"ModelManager_LastSyncTime": 0})
  assert mi.refresh_in_progress(None) is False
  assert mi.refresh_in_progress(time.monotonic()) is True
  # a fetch that never restamps the sync key stops spinning after the timeout
  assert mi.refresh_in_progress(time.monotonic() - mi.MODEL_SYNC_TIMEOUT - 1.0) is False
