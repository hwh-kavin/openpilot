#!/usr/bin/env bash
# Shared C3/AGNOS bootstrap helpers for launch scripts.
set -euo pipefail

c3_bootstrap_root() {
  if [[ -n "${C3_BOOTSTRAP_ROOT:-}" ]]; then
    echo "$C3_BOOTSTRAP_ROOT"
    return
  fi
  local dir
  dir="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
  echo "$dir"
}

c3_stage_firmware() {
  local root="$1"
  "$root/tools/stage_firmware.sh"
}

c3_sync_branch_params() {
  local root="$1"
  if [[ ! -d "$root/.git" ]]; then
    return 0
  fi
  local branch
  branch="$(git -C "$root" branch --show-current 2>/dev/null || true)"
  if [[ -z "$branch" ]]; then
    return 0
  fi
  if [[ -x "$root/.venv/bin/python3" ]]; then
    # shellcheck disable=SC1091
    source "$root/.venv/bin/activate"
    python3 - <<PY
from openpilot.common.params import Params
p = Params()
p.put("GitBranch", "$branch", block=True)
p.put("UpdaterTargetBranch", "$branch", block=True)
PY
  fi
}

c3_ensure_venv() {
  local root="$1"
  cd "$root"
  if [[ -d "$root/.venv" ]] && "$root/.venv/bin/python3" -c 'import acados' >/dev/null 2>&1; then
    return 0
  fi
  echo "Installing Python dependencies (local deps only)..."
  "$root/tools/setup_dependencies.sh"
}

c3_activate_venv() {
  local root="$1"
  if [[ -f "$root/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$root/.venv/bin/activate"
  fi
}

c3_ensure_build() {
  local root="$1"
  cd "$root/system/manager"
  if [[ ! -s "$root/prebuilt" ]]; then
    rm -f "$root/prebuilt"
    ./build.py
  elif [[ "$root/common/params_keys.h" -nt "$root/common/params_pyx.so" ]]; then
    echo "params_keys.h changed, rebuilding params..."
    cd "$root"
    scons common/params_pyx.so -j4
    cd "$root/system/manager"
  fi
}
