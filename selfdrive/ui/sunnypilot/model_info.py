"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import contextlib
import os
import time

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.modeld.helpers import chestnut_present
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.models.fetcher import get_cached_bundles
from openpilot.sunnypilot.models.helpers import get_selected_bundle, resolve_bundle_by_ref
from openpilot.sunnypilot.models.model_name import DEFAULT_BIG_MODEL, DEFAULT_MODEL


def model_cache_size_mb() -> float:
  """Bytes on disk under the model cache directory, in MB."""
  model_root = Paths.model_root()
  total = 0
  if os.path.isdir(model_root):
    for name in os.listdir(model_root):
      with contextlib.suppress(OSError):
        total += os.path.getsize(os.path.join(model_root, name))
  return total / (1024 ** 2)


def chestnut_supported() -> bool:
  """BluePilot: the big model slot needs chestnut hardware, which stock modeld on this fork cannot drive."""
  return chestnut_present()


def active_source() -> str:
  # only the qcom slot can produce the driving model on this fork
  return "qcom"


def bundles_for_source(source: str):
  if not chestnut_supported() and source == "chestnut":
    return []
  if source == active_source():
    return ui_state.sm["modelManagerSP"].availableBundles
  return get_cached_bundles(ui_state.params, source)


def default_model(source: str) -> str:
  return DEFAULT_BIG_MODEL if source == 'chestnut' else DEFAULT_MODEL


def default_model_name(source: str) -> str:
  return f"{default_model(source)} (Default)"


def big_model_state() -> str | None:
  """'failed' | 'loading' | None. Always None without chestnut hardware."""
  return None


def carrying_model() -> tuple[str | None, str | None, str | None]:
  """(source, internal name, display name) of what actually drives."""
  source = active_source()
  bundle = get_selected_bundle(ui_state.params, source)
  if bundle:
    return source, bundle.internalName, bundle.displayName
  name = default_model_name(source)
  return source, name, name


def queued_name(current_ref) -> str | None:
  ref = ui_state.params.get("ModelManager_DownloadRef")
  if ref and ref != current_ref:
    source_bundles = {source: bundles_for_source(source) for source in ("qcom", "chestnut")}
    if resolved := resolve_bundle_by_ref(ref, source_bundles):
      return resolved[0].internalName
  return None


def model_info() -> tuple[str, str, str]:
  """returns (active source, active model name, other model name)"""
  source = active_source()
  other = "qcom" if source == "chestnut" else "chestnut"
  active_bundle = get_selected_bundle(ui_state.params, source)
  other_bundle = get_selected_bundle(ui_state.params, other)

  active_name = active_bundle.displayName if active_bundle else default_model_name(source)
  other_name = other_bundle.displayName if other_bundle else default_model_name(other)
  return source, active_name, other_name


# mirrors the manager's ModelCache keys; the manager restamps them on a successful fetch
MODEL_SYNC_KEYS = ("ModelManager_LastSyncTime",)
MODEL_SYNC_TIMEOUT = 20.0


def refresh_model_list() -> None:
  # zeroing the sync keys makes the manager refetch each manifest on its next tick
  for key in MODEL_SYNC_KEYS:
    ui_state.params.put(key, 0)


def refresh_in_progress(started_at: float | None) -> bool:
  """Whether a user refresh is still outstanding. A failed fetch never restamps the
  sync keys, so the spinner is bounded by MODEL_SYNC_TIMEOUT rather than sticking."""
  if started_at is None or time.monotonic() - started_at > MODEL_SYNC_TIMEOUT:
    return False
  return not all(ui_state.params.get(key) for key in MODEL_SYNC_KEYS)
