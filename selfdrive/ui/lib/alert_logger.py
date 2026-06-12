from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openpilot.common.swaglog import cloudlog


_ALERT_STATUS = {
  0: "normal",
  1: "userPrompt",
  2: "critical",
}

_ALERT_SIZE = {
  0: "none",
  1: "small",
  2: "mid",
  3: "full",
}


@dataclass(frozen=True)
class _OnroadAlertKey:
  text1: str
  text2: str
  size: int
  status: int
  alert_type: str


class UiAlertLogger:
  """Log UI-visible alerts when they appear, change, or clear."""

  def __init__(self) -> None:
    self._onroad_key: _OnroadAlertKey | None = None
    self._offroad_visible: dict[str, str] = {}
    self._circular_key: tuple[str, str] | None = None
    self._update_available = False

  def log_onroad(self, alert: Any | None) -> None:
    if alert is None:
      key = None
    else:
      key = _OnroadAlertKey(
        text1=alert.text1 or "",
        text2=alert.text2 or "",
        size=int(alert.size),
        status=int(alert.status),
        alert_type=str(getattr(alert, "alert_type", "") or ""),
      )

    if key == self._onroad_key:
      return

    if self._onroad_key is not None and key is None:
      cloudlog.info("UI onroad alert cleared")

    self._onroad_key = key
    if key is None:
      return

    status = _ALERT_STATUS.get(key.status, str(key.status))
    size = _ALERT_SIZE.get(key.size, str(key.size))
    msg = f"UI onroad alert [{status}/{size}]"
    if key.alert_type:
      msg += f" ({key.alert_type})"
    msg += f": {key.text1}"
    if key.text2:
      msg += f" | {key.text2}"

    if key.status >= 1:
      cloudlog.warning(msg)
    else:
      cloudlog.info(msg)

  def sync_offroad(self, alerts: Any) -> None:
    current: dict[str, tuple[str, int]] = {}
    for alert in alerts:
      if alert.visible and alert.text:
        current[alert.key] = (alert.text, int(alert.severity))

    for key, (text, severity) in current.items():
      prev = self._offroad_visible.get(key)
      if prev != text:
        self._log_offroad_event("shown", key, text, severity)

    for key, text in self._offroad_visible.items():
      if key not in current:
        self._log_offroad_event("cleared", key, text)

    self._offroad_visible = {key: text for key, (text, _) in current.items()}

  def log_update_available(self, visible: bool) -> None:
    if visible == self._update_available:
      return

    self._update_available = visible
    if visible:
      cloudlog.info("UI offroad alert [update]: Update available")
    else:
      cloudlog.info("UI offroad alert cleared [update]")

  def log_circular(self, alert_id: str | None, text: str = "") -> None:
    if alert_id == "standstill":
      key = (alert_id, "")
    else:
      key = (alert_id, text) if alert_id else None
    if key == self._circular_key:
      return

    if self._circular_key is not None and key is None:
      cloudlog.info(f"UI circular alert cleared [{self._circular_key[0]}]")

    self._circular_key = key
    if key is None:
      return

    cloudlog.info(f"UI circular alert [{alert_id}]: {text.replace(chr(10), ' ')}")

  def _log_offroad_event(self, event: str, key: str, text: str, severity: int = 0) -> None:
    preview = " ".join(text.split())
    if len(preview) > 160:
      preview = preview[:157] + "..."

    msg = f"UI offroad alert [{event}] {key}: {preview}"
    if event == "shown" and severity > 0:
      cloudlog.warning(msg)
    else:
      cloudlog.info(msg)


ui_alert_logger = UiAlertLogger()
