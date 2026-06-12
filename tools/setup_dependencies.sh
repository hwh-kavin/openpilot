#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$(cd "$DIR/../" && pwd)"
# Ensure critical directories exist so SCons/autogen and installers don't fail
mkdir -p "$ROOT/board/obj"
mkdir -p "$ROOT/.venv"
mkdir -p "/data/scons_cache" || true
mkdir -p "/data/tmp" || true
OP_DEPS_CACHE="$ROOT/.deps_cache"
DEPS_DIR="$ROOT/deps"
DEPS_REPO="https://github.com/commaai/dependencies.git"
if [[ -f "/AGNOS" ]]; then
  ALLOW_NETWORK_FETCH="${ALLOW_NETWORK_FETCH:-0}"
else
  ALLOW_NETWORK_FETCH="${ALLOW_NETWORK_FETCH:-1}"
fi

# Check write permissions for important directories and abort with clear message
function check_writable_or_die() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    mkdir -p "$path" 2>/dev/null || true
  fi
  if [[ ! -w "$path" ]]; then
    echo "ERROR: Path $path is not writable. Please ensure the user running this script has write permissions or run with sudo." >&2
    exit 1
  fi
}
check_writable_or_die "$ROOT/board/obj"
check_writable_or_die "$ROOT/.venv"
check_writable_or_die "/data/scons_cache"
check_writable_or_die "/data/tmp"

# name:git_branch pairs for commaai/dependencies native packages
VENDORED_PACKAGES=(
  "bootstrap-icons:release-bootstrap-icons"
  "capnproto:release-capnproto"
  "catch2:release-catch2"
  "eigen:release-eigen"
  "libjpeg:release-libjpeg"
  "zstd:release-zstd"
  "zeromq:release-zeromq"
  "bzip2:release-bzip2"
  "ffmpeg:release-ffmpeg"
  "libyuv:release-libyuv"
  "ncurses:release-ncurses"
  "json11:release-json11"
  "libusb:release-libusb"
  "git-lfs:release-git-lfs"
  "gcc-arm-none-eabi:release-gcc-arm-none-eabi"
  "xvfb:release-xvfb"
  "acados:release-acados"
  "raylib:release-raylib"
)

function retry() {
  local attempts=$1
  shift
  for i in $(seq 1 "$attempts"); do
    if "$@"; then
      return 0
    fi
    if [ "$i" -lt "$attempts" ]; then
      echo "  Attempt $i/$attempts failed, retrying in 5s..."
      sleep 5
    fi
  done
  return 1
}

function is_agnos_device() {
  [[ -f "/AGNOS" ]]
}

function agnos_version() {
  cat /VERSION 2>/dev/null || echo ""
}

function system_python() {
  if [[ -x "/usr/local/venv/bin/python3.12" ]]; then
    echo "/usr/local/venv/bin/python3.12"
  elif command -v python3.12 > /dev/null 2>&1; then
    command -v python3.12
  else
    command -v python3
  fi
}

function ensure_build_tools() {
  local shim_dir="$ROOT/.local-bin"
  mkdir -p "$shim_dir"

  if ! command -v ccache > /dev/null 2>&1; then
    cat > "$shim_dir/ccache" <<'EOF'
#!/bin/sh
exec cc "$@"
EOF
    chmod +x "$shim_dir/ccache"
  fi
  export PATH="$shim_dir:$PATH"

  if [[ -d "$OP_DEPS_CACHE" ]]; then
    find "$OP_DEPS_CACHE" -name .git -type d 2>/dev/null | while read -r gitdir; do
      git config --global --add safe.directory "$(dirname "$gitdir")" 2>/dev/null || true
    done
  fi
}

function local_package_path() {
  local name="$1"
  if [[ -d "$DEPS_DIR/$name" ]]; then
    echo "$DEPS_DIR/$name"
    return 0
  fi
  if [[ -d "$OP_DEPS_CACHE/$name/$name" ]]; then
    echo "$OP_DEPS_CACHE/$name/$name"
    return 0
  fi
  return 1
}

function fetch_vendored_package() {
  local name="$1"
  local branch="$2"
  local local_path

  if local_path="$(local_package_path "$name")"; then
    echo "  using local package at $local_path"
    mkdir -p "$OP_DEPS_CACHE/$name"
    rm -rf "$OP_DEPS_CACHE/$name/$name"
    cp -a "$local_path" "$OP_DEPS_CACHE/$name/$name"
    return 0
  fi

  if [[ "$ALLOW_NETWORK_FETCH" != "1" ]]; then
    echo "  ERROR: missing local package deps/$name" >&2
    echo "  Run ./tools/vendor_deps.sh on a dev machine and commit deps/." >&2
    return 1
  fi

  local dest="$OP_DEPS_CACHE/$name"
  mkdir -p "$OP_DEPS_CACHE"
  echo "  fetching $name ($branch) from $DEPS_REPO..."
  rm -rf "$dest"
  if ! retry 5 git clone --depth 1 --branch "$branch" "$DEPS_REPO" "$dest"; then
    echo "  failed to fetch $name from $DEPS_REPO"
    return 1
  fi
  git config --global --add safe.directory "$dest" 2>/dev/null || true
}

function install_vendored_python_packages() {
  local -a packages=("${VENDORED_PACKAGES[@]}")
  mkdir -p /data/tmp/uv-tmp
  export TMPDIR=/data/tmp/uv-tmp
  export UV_TMPDIR=/data/tmp/uv-tmp
  ensure_build_tools

  # AGNOS 16 ships pyray/raylib in /usr/local/venv for EGL UI.
  if is_agnos_device && [[ "$(agnos_version)" == "16" ]]; then
  local filtered=()
  for entry in "${packages[@]}"; do
    [[ "$entry" == raylib:* ]] && continue
    filtered+=("$entry")
  done
  packages=("${filtered[@]}")
  fi

  echo "installing vendored python packages from local cache..."
  local entry name branch
  for entry in "${packages[@]}"; do
    name="${entry%%:*}"
    branch="${entry#*:}"
  if ! fetch_vendored_package "$name" "$branch"; then
      return 1
    fi
    echo "  -> $name (this may take several minutes on device)"
    if ! retry 2 uv pip install --no-build-isolation "$OP_DEPS_CACHE/$name/$name"; then
      echo "  failed to install $name from local cache"
      return 1
    fi
  done
}

function install_linux_deps() {
  SUDO=""

  if [[ ! $(id -u) -eq 0 ]]; then
    if [[ -z $(which sudo) ]]; then
      echo "Please install sudo or run as root"
      exit 1
    fi
    SUDO="sudo"
  fi

  local missing_linux_deps=0
  for cmd in gcc g++ make curl curl-config git; do
    if ! command -v "$cmd" > /dev/null 2>&1; then
      missing_linux_deps=1
      break
    fi
  done

  if [[ "$missing_linux_deps" -eq 0 ]]; then
    echo "[ ] system packages already installed t=$SECONDS"
  elif command -v apt-get > /dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends ca-certificates build-essential curl libcurl4-openssl-dev locales git
  elif command -v dnf > /dev/null 2>&1; then
    $SUDO dnf install -y ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-langpack-en git
  elif command -v yum > /dev/null 2>&1; then
    $SUDO yum install -y ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-langpack-en git
  elif command -v pacman > /dev/null 2>&1; then
    $SUDO pacman -Syu --noconfirm --needed base-devel ca-certificates curl git
  elif command -v zypper > /dev/null 2>&1; then
    $SUDO zypper --non-interactive refresh
    $SUDO zypper --non-interactive install ca-certificates gcc gcc-c++ make curl libcurl-devel glibc-locale git
  elif command -v apk > /dev/null 2>&1; then
    $SUDO apk add --no-cache ca-certificates build-base curl curl-dev musl-locales git
  elif command -v xbps-install > /dev/null 2>&1; then
    $SUDO xbps-install -Syu base-devel ca-certificates curl git libcurl-devel glibc-locales
  elif is_agnos_device; then
    echo "[ ] AGNOS device: skipping distro package manager install"
  else
    echo "Unsupported Linux distribution. Supported package managers: apt-get, dnf, yum, pacman, zypper, apk, xbps-install."
    exit 1
  fi

  if [[ -d "/etc/udev/rules.d/" ]] && [[ -w "/etc/udev/rules.d/" ]]; then
    $SUDO tee /etc/udev/rules.d/11-openpilot.rules > /dev/null <<-EOF
	# Panda Jungle devices
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcf", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddef", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcf", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddef", MODE="0666"

	# Panda devices
	SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="df11", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddcc", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="3801", ATTRS{idProduct}=="ddee", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddcc", MODE="0666"
	SUBSYSTEM=="usb", ATTRS{idVendor}=="bbaa", ATTRS{idProduct}=="ddee", MODE="0666"

	# comma devices over ADB
	SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="1234", ENV{adb_user}="yes"
	EOF

    $SUDO rm -f /etc/udev/rules.d/11-panda.rules /etc/udev/rules.d/12-panda_jungle.rules /etc/udev/rules.d/50-comma-adb.rules
    $SUDO udevadm control --reload-rules && $SUDO udevadm trigger || true
  elif [[ -d "/etc/udev/rules.d/" ]]; then
    echo "[ ] read-only root filesystem: skipping udev rules"
  fi
}

function install_python_deps() {
  export PIP_DEFAULT_TIMEOUT=200
  cd "$ROOT"

  if ! command -v "uv" > /dev/null 2>&1; then
    if is_agnos_device; then
      echo "ERROR: uv not found on AGNOS (expected /usr/comma/shims/uv)" >&2
      return 1
    fi
    if [[ "$ALLOW_NETWORK_FETCH" != "1" ]]; then
      echo "ERROR: uv not found and ALLOW_NETWORK_FETCH=0" >&2
      return 1
    fi
    echo "installing uv..."
    retry 3 sh -c 'curl --retry 5 --retry-delay 5 --retry-all-errors -LsSf https://astral.sh/uv/install.sh | UV_GITHUB_TOKEN="${GITHUB_TOKEN:-}" sh'
    PATH="$HOME/.local/bin:$PATH"
  fi

  if [[ "$ALLOW_NETWORK_FETCH" == "1" ]]; then
    echo "updating uv..."
    uv self update || true
  fi

  local py_bin
  py_bin="$(system_python)"
  local -a uv_args=(sync --frozen --all-extras --python "$py_bin")

  if is_agnos_device; then
    export UV_PYTHON_PREFERENCE=only-system
    uv_args+=(--python-preference only-system)
  fi

  mkdir -p /data/tmp/uv-tmp
  export TMPDIR=/data/tmp/uv-tmp
  export UV_TMPDIR=/data/tmp/uv-tmp
  ensure_build_tools

  echo "creating venv with system python..."
  local venv_args=(--python "$py_bin" --clear)
  if is_agnos_device && [[ "$(agnos_version)" == "16" ]]; then
    # AGNOS ships pyray in /usr/local/venv; needed for font build during scons.
    venv_args+=(--system-site-packages)
  fi
  uv venv "${venv_args[@]}"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"

  echo "bootstrapping python build tooling..."
  if ! retry 5 uv pip install setuptools wheel Cython scons fonttools hatchling editables; then
    return 1
  fi

  if is_agnos_device; then
    if ! install_vendored_python_packages; then
      return 1
    fi
    echo "installing openpilot (editable, skip git-native rebuilds)..."
    if ! retry 5 uv pip install -e "$ROOT" --no-build-isolation --no-deps; then
      return 1
    fi
    echo "installing remaining PyPI packages..."
    local pypi_reqs="$DEPS_DIR/pypi-reqs.txt"
    if [[ ! -f "$pypi_reqs" && -f "$OP_DEPS_CACHE/pypi-reqs.txt" ]]; then
      pypi_reqs="$OP_DEPS_CACHE/pypi-reqs.txt"
    fi
    if [[ ! -f "$pypi_reqs" ]]; then
      uv export --frozen --no-hashes --no-emit-package openpilot --format requirements.txt 2>/dev/null \
        | grep -E '^[a-zA-Z0-9]' | grep -v '@ git+' > "$OP_DEPS_CACHE/pypi-reqs.txt"
      pypi_reqs="$OP_DEPS_CACHE/pypi-reqs.txt"
    fi
    if [[ -d "$DEPS_DIR/wheels" ]] && ls "$DEPS_DIR/wheels"/*.whl >/dev/null 2>&1; then
      if ! retry 5 uv pip install --no-index --find-links "$DEPS_DIR/wheels" -r "$pypi_reqs" --no-build-isolation; then
        return 1
      fi
    elif [[ "$ALLOW_NETWORK_FETCH" == "1" ]]; then
      if ! retry 5 uv pip install -r "$pypi_reqs" --no-build-isolation; then
        return 1
      fi
    else
      echo "ERROR: missing deps/wheels for offline PyPI install" >&2
      echo "Run ./tools/vendor_deps.sh and commit deps/wheels/." >&2
      return 1
    fi
  else
    echo "installing python packages..."
    if ! retry 5 uv "${uv_args[@]}"; then
      return 1
    fi
  fi

}

# --- Main ---

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  install_linux_deps
  echo "[ ] installed system dependencies t=$SECONDS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  if [[ $SHELL == "/bin/zsh" ]]; then
    RC_FILE="$HOME/.zshrc"
  elif [[ $SHELL == "/bin/bash" ]]; then
    RC_FILE="$HOME/.bash_profile"
  fi
fi

if [ -f "$ROOT/pyproject.toml" ]; then
  install_python_deps
  echo "[ ] installed python dependencies t=$SECONDS"
fi

if [[ "$OSTYPE" == "darwin"* ]] && [[ -n "${RC_FILE:-}" ]]; then
  echo
  echo "----   OPENPILOT SETUP DONE   ----"
  echo "Open a new shell or configure your active shell env by running:"
  echo "source $RC_FILE"
fi
