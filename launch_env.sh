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

# C3 (comma tici) stays on AGNOS 16; C3X branches may target newer AGNOS.
export AGNOS_VERSION="16"

export STAGING_ROOT="/data/safe_staging"
