#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export FPS=20

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# ui runs at 20fps (50ms frame budget), with enough headroom to preempt model workloads.
export QCOM_PRIORITY=12

# Comma 3 (tici) and this fork target AGNOS 16.
if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="16"
fi

export STAGING_ROOT="/data/safe_staging"

# AGNOS 16's runtime python (/usr/local/venv) may be missing pure-python deps
# that this fork ships in deps/wheels (e.g. jeepney for wifi_manager), which
# crashes the UI import chain. Install them offline on boot if missing (one-shot).
if [[ -f /AGNOS ]] && [[ -x /usr/local/venv/bin/python3.12 ]] && \
   ! /usr/local/venv/bin/python3.12 -c "import jeepney" 2>/dev/null; then
  LAUNCH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
  JEEPNEY_WHEEL="$LAUNCH_DIR/deps/wheels/jeepney-0.9.0-py3-none-any.whl"
  if [[ -f "$JEEPNEY_WHEEL" ]]; then
    echo "[launch_env] jeepney missing from runtime python, installing from vendored wheel..."
    sudo /usr/local/venv/bin/python3.12 -m pip install --quiet --no-index --no-deps "$JEEPNEY_WHEEL" \
      || echo "[launch_env] jeepney install failed; WiFi panel will be unavailable"
  fi
fi
