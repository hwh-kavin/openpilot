#!/usr/bin/env bash
# Deprecated: use ./tools/vendor_deps.sh to vendor all packages into deps/.
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vendor_deps.sh" "$@"
