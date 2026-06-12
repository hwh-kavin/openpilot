# Offline build dependencies

This directory contains vendored packages for C3/AGNOS installation without
fetching from external networks at install time.

## Layout

- `acados/`, `capnproto/`, ... — native Python packages (from commaai/dependencies)
- `pypi-reqs.txt` — locked PyPI requirements for the openpilot venv
- `wheels/` — pre-downloaded PyPI wheels for offline `uv pip install`

## Populate / refresh

On a dev machine with network access:

```bash
./tools/vendor_deps.sh
```

Then commit `deps/` to the repository.

## Install on device

Launch scripts call `tools/setup_dependencies.sh`, which installs exclusively
from this directory on AGNOS devices.
