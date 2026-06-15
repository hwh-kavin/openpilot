#!/usr/bin/env bash
# Build (optional) and package scons/native artifacts for zero-compile deploy.
# Run on a dev machine or C3 device after a successful build, then commit and push.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT/prebuilt_artifacts"
FILES_DIR="$ARTIFACTS_DIR/files"
PATHS_FILE="$ROOT/tools/prebuilt_paths.txt"

RUN_BUILD=1
RUN_WHEELS=1
COLLECT_ONLY=0

usage() {
  cat <<EOF
Usage: $0 [options]

  --collect-only   Package existing build outputs without running scons
  --skip-build     Do not run scons/panda build
  --skip-wheels    Do not rebuild deps/native_wheels
  -h, --help       Show this help

After packaging:
  git add -f prebuilt prebuilt_artifacts firmware_out deps/native_wheels
  git commit && git push
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect-only) COLLECT_ONLY=1; RUN_BUILD=0 ;;
    --skip-build) RUN_BUILD=0 ;;
    --skip-wheels) RUN_WHEELS=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ "$COLLECT_ONLY" -eq 1 ]]; then
  RUN_BUILD=0
fi

cd "$ROOT"

if [[ "$RUN_BUILD" -eq 1 ]]; then
  echo "==> building openpilot (scons)"
  if [[ -f /AGNOS ]]; then
    cd "$ROOT/system/manager"
    ./build.py
    cd "$ROOT"
  else
    scons -j"$(nproc)"
  fi

  echo "==> building panda firmware"
  if [[ -z "${PANDA_DEBUG_BUILD:-}" ]]; then
    scons -j"$(nproc)" panda/ || scons -j"$(nproc)" panda/
  else
    scons -j"$(nproc)" panda/
  fi
fi

echo "==> staging panda firmware to firmware_out/"
mkdir -p "$ROOT/firmware_out"
for fw in panda.bin.signed bootstub.panda.bin panda_h7.bin.signed bootstub.panda_h7.bin; do
  if [[ -f "$ROOT/panda/board/obj/$fw" ]]; then
    cp -f "$ROOT/panda/board/obj/$fw" "$ROOT/firmware_out/$fw"
  fi
done
if [[ -f "$ROOT/panda/board/obj/panda.bin.signed" ]]; then
  git -C "$ROOT/panda" describe --tags --always 2>/dev/null > "$ROOT/firmware_out/panda_version.txt" || true
fi

if [[ "$RUN_WHEELS" -eq 1 ]]; then
  echo "==> building native python wheels into deps/native_wheels/"
  mkdir -p "$ROOT/deps/native_wheels"
  PY_WHEEL="/usr/local/venv/bin/python3.12"
  if [[ ! -x "$PY_WHEEL" ]]; then
    PY_WHEEL="$(command -v python3.12 || command -v python3)"
  fi
  for pkg_dir in "$ROOT/deps"/*/; do
    name="$(basename "$pkg_dir")"
    case "$name" in
      wheels|native_wheels|pypi-reqs.txt|README.md) continue ;;
    esac
    if [[ ! -f "$pkg_dir/setup.py" && ! -f "$pkg_dir/pyproject.toml" ]]; then
      continue
    fi
    echo "  wheel: $name"
    "$PY_WHEEL" -m pip wheel --no-deps -w "$ROOT/deps/native_wheels" "$pkg_dir" || {
      echo "  WARNING: failed to build wheel for $name" >&2
    }
  done
fi

echo "==> collecting prebuilt artifacts"
rm -rf "$FILES_DIR"
mkdir -p "$FILES_DIR"

missing=0
while IFS= read -r rel || [[ -n "$rel" ]]; do
  [[ -z "$rel" || "$rel" =~ ^# ]] && continue
  src="$ROOT/$rel"
  if [[ ! -e "$src" ]]; then
    echo "  MISSING: $rel" >&2
    missing=$((missing + 1))
    continue
  fi
  dst="$FILES_DIR/$rel"
  mkdir -p "$(dirname "$dst")"
  if [[ -d "$src" ]]; then
    cp -a "$src" "$dst"
  else
    cp -f "$src" "$dst"
  fi
  echo "  packaged: $rel"
done < "$PATHS_FILE"

if [[ -d "$ROOT/cereal/gen" ]]; then
  mkdir -p "$FILES_DIR/cereal"
  cp -a "$ROOT/cereal/gen" "$FILES_DIR/cereal/gen"
  echo "  packaged: cereal/gen/"
fi

if [[ "$missing" -gt 0 ]]; then
  echo "ERROR: $missing required artifact(s) missing. Build first or use a machine with a complete build." >&2
  exit 1
fi

python3 - <<'PY' "$ARTIFACTS_DIR/manifest.json" "$FILES_DIR"
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
files_dir = Path(sys.argv[2])
entries = []
for path in sorted(files_dir.rglob("*")):
    if not path.is_file():
        continue
    rel = path.relative_to(files_dir).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append({"path": rel, "sha256": digest, "size": path.stat().st_size})

manifest = {
    "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "files": entries,
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"  manifest: {len(entries)} files")
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ROOT/prebuilt"
echo "packaged $(wc -l < "$PATHS_FILE" | tr -d ' ') paths + cereal/gen -> prebuilt_artifacts/"
echo
echo "Next steps:"
echo "  git add -f prebuilt prebuilt_artifacts firmware_out deps/native_wheels"
echo "  git commit -m 'Update prebuilt deploy artifacts'"
echo "  git push"
