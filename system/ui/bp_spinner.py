#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path
from openpilot.common.basedir import BASEDIR


class BPSpinner:
  def __init__(self):
    try:
      bp_spinner_path = os.path.join(BASEDIR, "selfdrive", "ui", "_bp_spinner")
      if not Path(bp_spinner_path).exists():
        bp_spinner_path = "./bp_spinner"

      self.spinner_proc = subprocess.Popen([bp_spinner_path], stdin=subprocess.PIPE, cwd=os.path.join(BASEDIR, "selfdrive", "ui"), close_fds=True)
      self.error_shown = False
    except OSError as e:
      print(f"Error initializing BPSpinner: {e}")
      self.spinner_proc = None

  def __enter__(self):
    return self

  def update(self, spinner_text: str):
    if self.spinner_proc is not None:
      try:
        self.spinner_proc.stdin.write(spinner_text.encode('utf8') + b"\n")
        self.spinner_proc.stdin.flush()
      except BrokenPipeError:
        pass

  def update_progress(self, cur: float, total: float):
    self.update(str(round(100 * cur / total)))

  def update_progress_with_text(self, cur: float, total: float, text: str):
    """Update both progress percentage and status text in a single call"""
    percentage = str(round(100 * cur / total))
    self.update(f"{percentage}|{text}")

  def build_retry(self):
    """Signal a build retry to trigger retry modal display"""
    self.error_shown = False
    self.update("BUILD_RETRY")

  def build_failed(self):
    """Signal a build failure to trigger error modal display"""
    self.error_shown = True
    self.update("BUILD_FAILED")

  def wait_for_exit(self):
    """Block indefinitely when an error modal is shown."""
    if self.error_shown:
      import time

      try:
        print("Waiting for user to close error modal...")
        while True:
          time.sleep(1)  # Sleep indefinitely until interrupted
      except KeyboardInterrupt:
        # Allow exiting with Ctrl+C
        print("Interrupted by user")

  def close(self):
    if self.spinner_proc is not None:
      self.spinner_proc.kill()
      try:
        self.spinner_proc.communicate(timeout=2.0)
      except subprocess.TimeoutExpired:
        print("WARNING: failed to kill bp_spinner")
      self.spinner_proc = None

  def __del__(self):
    self.close()

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()


if __name__ == "__main__":
  import time

  with BPSpinner() as s:
    # Demo of different update methods
    s.update("Simple status text")
    time.sleep(2.0)

    s.update_progress(25, 100)
    time.sleep(2.0)

    s.update_progress_with_text(50, 100, "Processing with status text")
    time.sleep(2.0)

    # Demo error handling
    # s.build_failed()
    # time.sleep(5.0)

  print("BPSpinner demo complete")
