#!/usr/bin/env python3
"""BluePilot device info helpers."""

import logging
import os

logger = logging.getLogger(__name__)

_OPENPILOT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
_PANDA_VERSION_FILE = os.path.join(_OPENPILOT_ROOT, 'panda/board/obj/version')


def get_panda_version() -> str | None:
  """Return live Panda firmware version, or the locally built version as fallback."""
  try:
    from panda import Panda

    with Panda() as p:
      version = p.get_version()
      if isinstance(version, str):
        version = version.strip()
        if version:
          return version
  except Exception as e:
    logger.debug("Could not read live Panda version: %s", e)

  try:
    if os.path.exists(_PANDA_VERSION_FILE):
      with open(_PANDA_VERSION_FILE, encoding='utf-8') as f:
        version = f.read().strip()
      return version if version else None
  except Exception as e:
    logger.debug("Could not read Panda version file: %s", e)

  return None
