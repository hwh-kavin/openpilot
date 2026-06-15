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

## Zero-compile deploy (prebuilt)

For users who deploy without compiling on the device:

1. On a machine with a successful build, run:

```bash
./tools/package_prebuilt.sh
```

2. Commit and push the generated artifacts:

```bash
git add -f prebuilt prebuilt_artifacts firmware_out deps/native_wheels
git commit -m "Update prebuilt deploy artifacts"
git push
```

3. Fresh clones restore `prebuilt_artifacts/` on boot and skip `scons`.

Required tracked paths are listed in `tools/prebuilt_paths.txt`.
