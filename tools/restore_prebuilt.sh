#!/usr/bin/env bash
# Restore scons/native artifacts from prebuilt_artifacts/ into the repo tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS_DIR="$ROOT/prebuilt_artifacts"
MANIFEST="$ARTIFACTS_DIR/manifest.json"

restore_prebuilt_artifacts() {
  local root="${1:-$ROOT}"

  if [[ ! -f "$root/prebuilt" ]] || [[ ! -s "$root/prebuilt" ]]; then
    return 1
  fi
  if [[ ! -d "$root/prebuilt_artifacts/files" ]]; then
    echo "restore_prebuilt: missing prebuilt_artifacts/files" >&2
    return 1
  fi

  local restored=0
  local missing=0
  while IFS= read -r rel || [[ -n "$rel" ]]; do
    [[ -z "$rel" || "$rel" =~ ^# ]] && continue
    local src="$root/prebuilt_artifacts/files/$rel"
    local dst="$root/$rel"
    if [[ ! -e "$src" ]]; then
      echo "restore_prebuilt: missing packaged file $rel" >&2
      missing=$((missing + 1))
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    if [[ -d "$src" ]]; then
      rm -rf "$dst"
      cp -a "$src" "$dst"
    else
      cp -f "$src" "$dst"
      if [[ -x "$src" ]]; then
        chmod +x "$dst"
      fi
    fi
    restored=$((restored + 1))
  done < "$root/tools/prebuilt_paths.txt"

  if [[ -d "$root/prebuilt_artifacts/files/cereal/gen" ]]; then
    rm -rf "$root/cereal/gen"
    cp -a "$root/prebuilt_artifacts/files/cereal/gen" "$root/cereal/gen"
    restored=$((restored + 1))
  fi

  echo "restore_prebuilt: restored $restored artifact(s)"
  if [[ "$missing" -gt 0 ]]; then
    echo "restore_prebuilt: $missing required artifact(s) missing from package" >&2
    return 1
  fi
  return 0
}

prebuilt_artifacts_valid() {
  local root="${1:-$ROOT}"
  local rel

  if [[ ! -f "$root/prebuilt" ]] || [[ ! -s "$root/prebuilt" ]]; then
    return 1
  fi

  while IFS= read -r rel || [[ -n "$rel" ]]; do
    [[ -z "$rel" || "$rel" =~ ^# ]] && continue
    if [[ ! -e "$root/$rel" ]]; then
      if [[ -e "$root/prebuilt_artifacts/files/$rel" ]]; then
        continue
      fi
      return 1
    fi
  done < "$root/tools/prebuilt_paths.txt"

  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  restore_prebuilt_artifacts "$ROOT"
fi
