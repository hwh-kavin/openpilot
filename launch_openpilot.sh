#!/usr/bin/env bash
export API_HOST=https://api.konik.ai
export ATHENA_HOST=wss://athena.konik.ai
#export MAPS_HOST=https://api.konik.ai/maps
export MAPBOX_TOKEN='YOUR_MAPBOX_TOKEN'

exec ./launch_chffrplus.sh
