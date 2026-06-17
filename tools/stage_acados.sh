#!/usr/bin/env bash
# Restore third_party/acados/larch64 binaries when git-lfs pointers are present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/acados/larch64"
SRC="$ROOT/deps/acados/acados/install"

is_lfs_pointer() {
  [[ -f "$1" ]] && head -1 "$1" 2>/dev/null | grep -q 'git-lfs.github.com/spec/v1'
}

needs_stage=0
for f in "$DEST/t_renderer" "$DEST/lib/libacados.so"; do
  if [[ ! -f "$f" ]] || is_lfs_pointer "$f"; then
    needs_stage=1
    break
  fi
done

if [[ "$needs_stage" -eq 0 ]]; then
  exit 0
fi

if [[ ! -x "$SRC/bin/t_renderer" ]]; then
  echo "ERROR: acados binaries missing at $SRC (git-lfs pull or deps/acados required)" >&2
  exit 1
fi

echo "Staging acados binaries from deps/acados into third_party/acados/larch64..."
mkdir -p "$DEST/lib"
install -m 755 "$SRC/bin/t_renderer" "$DEST/t_renderer"
cp -a "$SRC/lib/"*.so* "$DEST/lib/"
echo "Acados binaries staged."
