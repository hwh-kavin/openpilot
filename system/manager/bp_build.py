#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# NOTE: Do NOT import anything here that needs be built (e.g. params)
from openpilot.common.basedir import BASEDIR
from openpilot.system.hardware import HARDWARE, AGNOS
from openpilot.common.swaglog import cloudlog, add_file_handler
from openpilot.system.version import get_build_metadata

# Import our custom BPSpinner instead of the original Spinner
from openpilot.system.ui.bp_spinner import BPSpinner

MAX_CACHE_SIZE = 4e9 if "CI" in os.environ else 2e9
CACHE_DIR = Path("/data/scons_cache" if AGNOS else "/tmp/scons_cache")

TOTAL_SCONS_NODES = 3130
MAX_BUILD_PROGRESS = 100


def build(spinner: BPSpinner, dirty: bool = False, minimal: bool = False) -> None:
  env = os.environ.copy()
  env['SCONS_PROGRESS'] = "1"
  nproc = os.cpu_count()
  if nproc is None:
    nproc = 2

  print(f"nproc: {nproc}")

  extra_args = ["--minimal"] if minimal else []

  if AGNOS:
    HARDWARE.set_power_save(False)
    os.sched_setaffinity(0, range(8))  # ensure we can use the isolcpus cores

  # Track last status and progress
  last_status_text = ""
  last_progress = 0

  # building with all cores can result in using too
  # much memory, so retry with less parallelism
  compile_output: list[bytes] = []
  for n in (nproc, nproc / 2, 1):
    compile_output.clear()

    # Catch the retry and send a message to the spinner to reset the output modal text "BUILD_RETRY"
    if n > 1:
      print(f"BUILD_RETRY:{n}")
      spinner.build_retry()

    scons: subprocess.Popen = subprocess.Popen(["scons", f"-j{int(n)}", "--cache-populate", *extra_args], cwd=BASEDIR, env=env, stderr=subprocess.PIPE)
    assert scons.stderr is not None

    # Read progress from stderr and update spinner
    while scons.poll() is None:
      try:
        line = scons.stderr.readline()
        if line is None:
          continue
        line = line.rstrip()
        progressPrefix = b'progress: '
        show_detailed = True
        # Look for spinner detailed info
        if os.environ.get('SPINNER_DETAILED') or show_detailed:
          if line.startswith(progressPrefix):
            i = int(line[len(progressPrefix) :])
            last_progress = i

          elif len(line):
            line_str = line.decode('utf8', 'replace')
            last_status_text = line_str
            compile_output.append(line)
            print(line_str)

          # Update progress while preserving text
          spinner.update_progress_with_text(last_progress, TOTAL_SCONS_NODES, last_status_text)
        else:
          # Basic progress (original behavior)
          if line.startswith(progressPrefix):
            i = int(line[len(progressPrefix) :])
            spinner.update_progress(i, TOTAL_SCONS_NODES)
          elif len(line):
            compile_output.append(line)
            print(line.decode('utf8', 'replace'))
      except Exception as e:
        print(f"Error processing build output: {e}")

    if scons.returncode == 0:
      break

  if scons.returncode != 0:
    # Read remaining output
    if scons.stderr is not None:
      compile_output += scons.stderr.read().split(b'\n')

    # Build failed log errors
    error_s = b"\n".join(compile_output).decode('utf8', 'replace')
    add_file_handler(cloudlog)
    cloudlog.error("scons build failed\n" + error_s)

    # Signal build failed to spinner to display error modal
    print("sending BUILD_FAILED")
    spinner.build_failed()
    # Wait for the user to dismiss the modal
    spinner.wait_for_exit()  # This will block until interrupted
    exit(1)

  # enforce max cache size
  cache_files = [f for f in CACHE_DIR.rglob('*') if f.is_file()]
  cache_files.sort(key=lambda f: f.stat().st_mtime)
  cache_size = sum(f.stat().st_size for f in cache_files)
  for f in cache_files:
    if cache_size < MAX_CACHE_SIZE:
      break
    cache_size -= f.stat().st_size
    f.unlink()


if __name__ == "__main__":
  # Use our custom BPSpinner instead of the original Spinner
  spinner = BPSpinner()

  # Set environment variable to get detailed output
  os.environ['SPINNER_DETAILED'] = '1'

  # Initial progress update
  spinner.update_progress(0, 100)

  # Get build metadata
  build_metadata = get_build_metadata()

  # Start the build
  build(spinner, build_metadata.openpilot.is_dirty, minimal=AGNOS)
