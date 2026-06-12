Vendor acados into repository

Goal: allow projects to include `acados` source inside the repo under `deps/acados` so `tools/setup_dependencies.sh` can install from local files without long network builds.

Recommended ways to vendor:

1) Git subtree (preserves history in your branch):

   git remote add deps https://github.com/commaai/dependencies.git
   git subtree pull --prefix deps/acados deps release-acados --squash

2) Git submodule (keeps external repo reference):

   git submodule add -b release-acados https://github.com/commaai/dependencies.git deps/dependencies
   # Then copy deps/dependencies/acados -> deps/acados or adjust setup to use submodule path

Helper script:

- `tools/vendor_acados_helper.sh` will copy `~/.deps_cache/acados/acados` into `deps/acados` if available locally (useful for CI or offline workflows).

After vendoring:

- Commit `deps/acados` into your repo. `tools/setup_dependencies.sh` will detect `deps/acados` and prefer installing from it.

Notes:

- Vendoring large native dependencies increases repo size. Consider keeping them in a separate vendor repo or using release artifacts (wheels).
- Building `acados` can be resource-intensive; prefer building in CI and providing prebuilt wheels where possible.
