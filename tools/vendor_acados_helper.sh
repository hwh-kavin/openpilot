#!/usr/bin/env bash
# Helper: copy acados from .deps_cache to deps/ or vice versa to help vendoring.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OP_DEPS_CACHE="$ROOT/.deps_cache"

if [[ -d "$OP_DEPS_CACHE/acados/acados" ]]; then
  echo "Found acados in .deps_cache, copying to deps/acados..."
  mkdir -p "$ROOT/deps"
  rm -rf "$ROOT/deps/acados"
  cp -a "$OP_DEPS_CACHE/acados/acados" "$ROOT/deps/acados"
  echo "Copied. Commit deps/acados to vendor acados in repository if desired."
else
  echo "No acados found in .deps_cache. You can run ./tools/setup_dependencies.sh to fetch it first."
fi
