import os
import time
from enum import Enum, auto

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params


LOG_PATH = "/data/media/0/realdata/debug.log"
PARAM_REFRESH_SEC = 2.0
MIN_SPEED_MS = 0.1
ANGLE_ERROR_FILTER_RC = 0.25


def _log(message: str) -> None:
  try:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
      f.write(f"{time.time():.3f} {message}\n")
  except Exception:
    pass


class HTDState(Enum):
  INACTIVE = auto()
  PAUSED = auto()


class HumanTurnDetection:
  def __init__(self) -> None:
    self._params = Params()
    self._last_params_read = 0.0

    self._enabled = False
    self._angle_threshold_deg = 45.0
    self._resume_angle_diff_deg = 10.0
    self._resume_delay_sec = 0.3

    self._state: HTDState = HTDState.INACTIVE
    self._angle_error_filter = FirstOrderFilter(0.0, ANGLE_ERROR_FILTER_RC, 0.01)
    self._recovery_condition_met_since: float | None = None

    self._last_angle = 0.0
    self._last_pressed = False
    self._last_model_angle = 0.0
    self._last_filtered_error = 0.0

  def _read_params(self) -> None:
    now = time.monotonic()
    if now - self._last_params_read < PARAM_REFRESH_SEC:
      return
    self._last_params_read = now

    self._enabled = self._params.get_bool("dp_htd_enabled")
    self._angle_threshold_deg = self._get_float("dp_htd_turn_angle_threshold", 45.0)
    self._resume_angle_diff_deg = self._get_float("dp_htd_resume_angle_diff", 10.0)
    delay_ms = self._get_float("dp_htd_resume_delay_ms", 300.0)
    self._resume_delay_sec = max(0.0, delay_ms) / 1000.0

  def _transition(self, new_state: HTDState, reason: str) -> None:
    if new_state == self._state:
      return
    self._state = new_state
    self._recovery_condition_met_since = None
    _log(
      f"HTD {new_state.name} reason={reason} angle={self._last_angle:.1f} "
      f"model={self._last_model_angle:.1f} err={self._last_filtered_error:.1f} "
      f"pressed={self._last_pressed} delay={self._resume_delay_sec:.2f}"
    )

  def update(
    self,
    lat_active: bool,
    steering_angle_deg: float,
    v_ego: float,
    steering_pressed: bool,
    model_desired_angle_deg: float,
    dt: float = 0.01,
  ) -> tuple[bool, HTDState]:
    self._read_params()
    self._angle_error_filter.dt = dt

    self._last_angle = abs(steering_angle_deg)
    self._last_pressed = steering_pressed
    self._last_model_angle = model_desired_angle_deg

    # Recovery 1: master switch off -> immediate resume
    if not self._enabled:
      if self._state != HTDState.INACTIVE:
        self._transition(HTDState.INACTIVE, "disabled")
      return True, self._state

    if v_ego < MIN_SPEED_MS:
      if self._state != HTDState.INACTIVE:
        self._transition(HTDState.INACTIVE, "low_speed")
      return True, self._state

    if self._state == HTDState.INACTIVE:
      if lat_active and self._should_trigger():
        self._transition(HTDState.PAUSED, "trigger")
        return False, self._state
      return True, self._state

    # PAUSED: lateral control remains off until a recovery path completes
    recovery_ready = False
    if not steering_pressed:
      recovery_ready = self._recovery_delay_elapsed("hands_off")
    else:
      angle_error = abs(model_desired_angle_deg - steering_angle_deg)
      self._angle_error_filter.update(angle_error)
      self._last_filtered_error = self._angle_error_filter.x
      if self._last_filtered_error < self._resume_angle_diff_deg:
        recovery_ready = self._recovery_delay_elapsed("angle_aligned")
      else:
        self._recovery_condition_met_since = None

    if recovery_ready:
      self._transition(HTDState.INACTIVE, "resume")
      return True, self._state

    return False, self._state

  def _should_trigger(self) -> bool:
    return self._last_pressed and self._last_angle >= self._angle_threshold_deg

  def _recovery_delay_elapsed(self, reason: str) -> bool:
    if self._resume_delay_sec <= 0.0:
      return True

    now = time.monotonic()
    if self._recovery_condition_met_since is None:
      self._recovery_condition_met_since = now
      return False

    if now - self._recovery_condition_met_since >= self._resume_delay_sec:
      _log(f"HTD recovery delay done reason={reason}")
      return True
    return False

  def _get_float(self, key: str, default: float) -> float:
    try:
      val = self._params.get(key)
      if val is None:
        return default
      return float(val)
    except Exception:
      return default
