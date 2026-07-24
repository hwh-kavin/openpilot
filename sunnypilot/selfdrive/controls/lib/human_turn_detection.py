import os
import time
from enum import Enum, auto

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params


LOG_PATH = "/data/media/0/realdata/debug.log"
PARAM_REFRESH_SEC = 2.0
MIN_SPEED_MS = 0.1
ANGLE_ERROR_FILTER_RC = 0.25

# Curve-exit defaults (Ford Q3–friendly: difference-based, not absolute steer ≥25°)
DEFAULT_CURVE_EXIT_MODEL_DEG = 6.0       # model nearly straight
DEFAULT_CURVE_LATCH_DEG = 12.0           # confirm we were in a curve
DEFAULT_CURVE_EXIT_ERROR_DEG = 10.0      # |actual − model| to release
DEFAULT_CURVE_EXIT_RESUME_ERROR_DEG = 5.0  # |actual − model| to re-engage (hysteresis)
# Same role as human HTD delay: hold pause after align so desired κ snaps to
# current wheel before re-engage (avoids replaying the prior curve command).
DEFAULT_CURVE_EXIT_RESUME_DELAY_MS = 300.0


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


class PauseReason(Enum):
  NONE = auto()
  HUMAN_TURN = auto()
  CURVE_EXIT = auto()


class HumanTurnDetection:
  def __init__(self) -> None:
    self._params = Params()
    self._last_params_read = 0.0

    self._enabled = False
    self._angle_threshold_deg = 45.0
    self._resume_angle_diff_deg = 10.0
    self._resume_delay_sec = 0.3

    self._curve_exit_enabled = True
    self._curve_exit_model_deg = DEFAULT_CURVE_EXIT_MODEL_DEG
    self._curve_latch_deg = DEFAULT_CURVE_LATCH_DEG
    self._curve_exit_error_deg = DEFAULT_CURVE_EXIT_ERROR_DEG
    self._curve_exit_resume_error_deg = DEFAULT_CURVE_EXIT_RESUME_ERROR_DEG
    self._curve_exit_resume_delay_sec = DEFAULT_CURVE_EXIT_RESUME_DELAY_MS / 1000.0

    self._state: HTDState = HTDState.INACTIVE
    self._pause_reason = PauseReason.NONE
    self._curve_latched = False
    self._angle_error_filter = FirstOrderFilter(0.0, ANGLE_ERROR_FILTER_RC, 0.01)
    self._recovery_condition_met_since: float | None = None

    self._last_angle = 0.0
    self._last_steer_deg = 0.0
    self._last_pressed = False
    self._last_model_angle = 0.0
    self._last_filtered_error = 0.0
    self._lane_changing = False

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

    self._curve_exit_enabled = self._params.get_bool("dp_htd_curve_exit_enabled")
    self._curve_exit_model_deg = self._get_float(
      "dp_htd_curve_exit_model_angle", DEFAULT_CURVE_EXIT_MODEL_DEG)
    self._curve_latch_deg = self._get_float(
      "dp_htd_curve_latch_angle", DEFAULT_CURVE_LATCH_DEG)
    self._curve_exit_error_deg = self._get_float(
      "dp_htd_curve_exit_error", DEFAULT_CURVE_EXIT_ERROR_DEG)
    self._curve_exit_resume_error_deg = self._get_float(
      "dp_htd_curve_exit_resume_error", DEFAULT_CURVE_EXIT_RESUME_ERROR_DEG)
    # Keep resume hysteresis below trigger to avoid chatter
    self._curve_exit_resume_error_deg = min(
      self._curve_exit_resume_error_deg, self._curve_exit_error_deg)
    ce_delay_ms = self._get_float(
      "dp_htd_curve_exit_resume_delay_ms", DEFAULT_CURVE_EXIT_RESUME_DELAY_MS)
    self._curve_exit_resume_delay_sec = max(0.0, ce_delay_ms) / 1000.0

  def _transition(self, new_state: HTDState, reason: str,
                   pause_reason: PauseReason = PauseReason.NONE) -> None:
    if new_state == self._state and (
        new_state != HTDState.PAUSED or pause_reason == self._pause_reason):
      return
    self._state = new_state
    self._pause_reason = pause_reason if new_state == HTDState.PAUSED else PauseReason.NONE
    self._recovery_condition_met_since = None
    if new_state == HTDState.INACTIVE:
      self._curve_latched = False
    elif new_state == HTDState.PAUSED and pause_reason == PauseReason.CURVE_EXIT:
      # Seed filter with current error so we don't falsely resume on first frames
      err = abs(self._last_model_angle - self._last_steer_deg)
      self._angle_error_filter.x = err
      self._last_filtered_error = err
    _log(
      f"HTD {new_state.name} reason={reason} pause={self._pause_reason.name} "
      f"angle={self._last_angle:.1f} model={self._last_model_angle:.1f} "
      f"err={self._last_filtered_error:.1f} pressed={self._last_pressed} "
      f"curve_latched={self._curve_latched} delay={self._resume_delay_sec:.2f}"
    )

  def update(
    self,
    lat_active: bool,
    steering_angle_deg: float,
    v_ego: float,
    steering_pressed: bool,
    model_desired_angle_deg: float,
    dt: float = 0.01,
    lane_changing: bool = False,
  ) -> tuple[bool, HTDState]:
    self._read_params()
    self._angle_error_filter.dt = dt

    self._last_angle = abs(steering_angle_deg)
    self._last_steer_deg = steering_angle_deg
    self._last_pressed = steering_pressed
    self._last_model_angle = model_desired_angle_deg
    self._lane_changing = lane_changing

    # Recovery 1: master switch off -> immediate resume
    if not self._enabled:
      if self._state != HTDState.INACTIVE:
        self._transition(HTDState.INACTIVE, "disabled")
      return True, self._state

    if v_ego < MIN_SPEED_MS:
      if self._state != HTDState.INACTIVE:
        self._transition(HTDState.INACTIVE, "low_speed")
      return True, self._state

    # Remember we were in a meaningful curve (model wanted large steer)
    if abs(model_desired_angle_deg) >= self._curve_latch_deg:
      self._curve_latched = True

    if self._state == HTDState.INACTIVE:
      if lat_active and self._should_trigger_human():
        self._transition(HTDState.PAUSED, "trigger_human", PauseReason.HUMAN_TURN)
        return False, self._state
      if lat_active and self._should_trigger_curve_exit():
        self._transition(HTDState.PAUSED, "trigger_curve_exit", PauseReason.CURVE_EXIT)
        return False, self._state
      return True, self._state

    # PAUSED: lateral control remains off until a recovery path completes
    if self._pause_reason == PauseReason.CURVE_EXIT:
      recovery_ready = self._curve_exit_recovery_ready(
        steering_angle_deg, model_desired_angle_deg)
    elif not steering_pressed:
      recovery_ready = self._recovery_delay_elapsed("hands_off", self._resume_delay_sec)
    else:
      angle_error = abs(model_desired_angle_deg - steering_angle_deg)
      self._angle_error_filter.update(angle_error)
      self._last_filtered_error = self._angle_error_filter.x
      if self._last_filtered_error < self._resume_angle_diff_deg:
        recovery_ready = self._recovery_delay_elapsed("angle_aligned", self._resume_delay_sec)
      else:
        self._recovery_condition_met_since = None
        recovery_ready = False

    if recovery_ready:
      self._transition(HTDState.INACTIVE, "resume")
      return True, self._state

    return False, self._state

  def _should_trigger_human(self) -> bool:
    return self._last_pressed and self._last_angle >= self._angle_threshold_deg

  def _should_trigger_curve_exit(self) -> bool:
    """
    Release lat after a curve when model is nearly straight but |actual − model|
    is still large — vehicle self-centers, then we re-engage near desired.
    """
    if not self._curve_exit_enabled:
      return False
    if self._lane_changing:
      return False
    if self._last_pressed:
      return False
    if not self._curve_latched:
      return False

    model_abs = abs(self._last_model_angle)
    if model_abs > self._curve_exit_model_deg:
      return False

    angle_error = abs(self._last_model_angle - self._last_steer_deg)
    return angle_error >= self._curve_exit_error_deg

  def _curve_exit_recovery_ready(self, steering_angle_deg: float,
                                 model_desired_angle_deg: float) -> bool:
    # Model wants another turn — re-engage promptly, do not miss the curve
    if abs(model_desired_angle_deg) > max(self._curve_exit_model_deg * 2.0, self._curve_latch_deg):
      self._recovery_condition_met_since = None
      _log(
        f"HTD curve_exit abort model_turn model={model_desired_angle_deg:.1f} "
        f"steer={steering_angle_deg:.1f}"
      )
      return True

    angle_error = abs(model_desired_angle_deg - steering_angle_deg)
    self._angle_error_filter.update(angle_error)
    self._last_filtered_error = self._angle_error_filter.x

    # Immediate takeback when near desired; delay lets desired κ snap to wheel for blend
    if angle_error < self._curve_exit_resume_error_deg:
      # Prefer shared HTD resume delay when curve-exit delay unset/zero legacy
      delay = self._curve_exit_resume_delay_sec
      if delay <= 0.0:
        delay = self._resume_delay_sec
      return self._recovery_delay_elapsed("curve_exit_aligned", delay)

    self._recovery_condition_met_since = None
    return False

  def _recovery_delay_elapsed(self, reason: str, delay_sec: float) -> bool:
    if delay_sec <= 0.0:
      return True

    now = time.monotonic()
    if self._recovery_condition_met_since is None:
      self._recovery_condition_met_since = now
      return False

    if now - self._recovery_condition_met_since >= delay_sec:
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
