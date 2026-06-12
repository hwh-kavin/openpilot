#!/usr/bin/env bash
# Vendor build dependencies into deps/ for offline C3 installation.
# Run once on a machine with network access, then commit deps/ to the repo.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_DIR="$ROOT/deps"
CACHE_DIR="$ROOT/.deps_cache"

PACKAGES=(
  bootstrap-icons
  capnproto
  catch2
  eigen
  libjpeg
  zstd
  zeromq
  bzip2
  ffmpeg
  libyuv
  ncurses
  json11
  libusb
  git-lfs
  gcc-arm-none-eabi
  xvfb
  acados
)

mkdir -p "$DEPS_DIR"

echo "==> vendoring native packages into deps/"
for name in "${PACKAGES[@]}"; do
  src=""
  if [[ -d "$DEPS_DIR/$name" ]]; then
    echo "  skip $name (already in deps/)"
    continue
  elif [[ -d "$CACHE_DIR/$name/$name" ]]; then
    src="$CACHE_DIR/$name/$name"
  else
    echo "  ERROR: missing $name in deps/ and .deps_cache/" >&2
    echo "  Run ./tools/setup_dependencies.sh once with network to populate .deps_cache," >&2
    echo "  or copy packages manually into deps/<name>." >&2
    exit 1
  fi
  echo "  -> $name"
  cp -a "$src" "$DEPS_DIR/$name"
done

echo "==> exporting PyPI requirements"
if [[ -f "$ROOT/uv.lock" ]] && command -v uv >/dev/null 2>&1; then
  uv export --frozen --no-hashes --no-emit-package openpilot --format requirements.txt \
    | grep -E '^[a-zA-Z0-9]' | grep -v '@ git+' > "$DEPS_DIR/pypi-reqs.txt"
elif [[ -f "$CACHE_DIR/pypi-reqs.txt" ]]; then
  cp "$CACHE_DIR/pypi-reqs.txt" "$DEPS_DIR/pypi-reqs.txt"
else
  echo "  ERROR: no uv.lock export and no .deps_cache/pypi-reqs.txt" >&2
  exit 1
fi

echo "==> downloading PyPI wheels into deps/wheels/"
mkdir -p "$DEPS_DIR/wheels"
PY_BIN="/usr/local/venv/bin/python3.12"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3.12 || command -v python3)"
fi
# uv on AGNOS does not support `pip download`; use system pip directly.
"$PY_BIN" -m pip download -r "$DEPS_DIR/pypi-reqs.txt" -d "$DEPS_DIR/wheels"

echo "==> done. deps/ is ready for offline installation."
