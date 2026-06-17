# Offline build dependencies

This directory contains vendored packages for local development setup without
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

## Install (development machines only)

Run manually when setting up a PC dev environment:

```bash
./tools/op.sh setup
```

AGNOS devices use the system Python environment; launch scripts do not run
dependency installation on boot.
