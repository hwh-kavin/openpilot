#!/usr/bin/env bash
export ATHENA_HOST='ws://athena.mr-one.cn'
export API_HOST='http://res.mr-one.cn'

C3_LAUNCH_SH="./sunnypilot/system/hardware/c3/launch_chffrplus.sh"
MODEL="$(tr -d '\0' < "/sys/firmware/devicetree/base/model" 2>/dev/null || true)"
export MODEL

if [ "$MODEL" = "comma tici" ] && [ -x "$C3_LAUNCH_SH" ]; then
  exec "$C3_LAUNCH_SH"
fi

exec ./launch_chffrplus.sh
