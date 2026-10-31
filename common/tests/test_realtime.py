import time

from openpilot.common.realtime import Ratekeeper


class TestRatekeeper:
  def setup_method(self):
    self.interval = 0.1
    self.rk = Ratekeeper(1.0 / self.interval, print_delay_threshold=None)
    # initialize the schedule
    self.rk.monitor_time()

  def test_normal_frame_advances_schedule(self):
    # less than one frame behind: keep the existing schedule
    self.rk._next_frame_time = time.monotonic() + 0.5 * self.interval
    now = time.monotonic()
    self.rk.monitor_time()
    assert self.rk._next_frame_time > now + self.interval

  def test_stall_resyncs_instead_of_catch_up(self):
    # simulate a long stall (e.g. process descheduled for several frames)
    now = time.monotonic()
    self.rk._next_frame_time = now - 10 * self.interval
    self.rk.monitor_time()
    # the schedule is realigned to one interval from now, not replayed frame by frame
    remaining = self.rk._next_frame_time - time.monotonic()
    assert 0.0 < remaining <= self.interval

  def test_keep_time_does_not_run_back_to_back_after_stall(self):
    self.rk._next_frame_time = time.monotonic() - 10 * self.interval
    start = time.monotonic()
    self.rk.keep_time()
    # a resync sleeps for roughly one interval instead of draining a 1s backlog
    assert time.monotonic() - start < 5 * self.interval
