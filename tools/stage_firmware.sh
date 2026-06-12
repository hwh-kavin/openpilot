#!/usr/bin/env bash
# Fallback: copy prebuilt panda firmware from firmware_out/ if not built yet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FW_SRC="$ROOT/firmware_out"
FW_DST="$ROOT/panda/board/obj"

if [[ ! -d "$FW_SRC" ]]; then
  exit 0
fi

mkdir -p "$FW_DST"

stage_file() {
  local name="$1"
  if [[ -f "$FW_DST/$name" ]]; then
    return 0
  fi
  if [[ -f "$FW_SRC/$name" ]]; then
    echo "stage_firmware: using prebuilt $name from firmware_out/"
    cp -f "$FW_SRC/$name" "$FW_DST/$name"
  fi
}

# C3 internal panda (STM32F4 / DOS) — prefer scons-built firmware
stage_file "panda.bin.signed"
stage_file "bootstub.panda.bin"

# H7 pandas (external / newer hardware)
stage_file "panda_h7.bin.signed"
stage_file "bootstub.panda_h7.bin"
