#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export FPS=20

export QCOM_PRIORITY=12

# Comma 3 (tici) targets AGNOS 16.
if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="16"
fi

export STAGING_ROOT="/data/safe_staging"
