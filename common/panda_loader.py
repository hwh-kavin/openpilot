"""BluePilot: load panda or panda_tici per hwj dp260513 C3 logic (TICI_HW / TICI_TRES)."""
import importlib
import os


def load_panda_module():
  if os.environ.get("TICI_HW") and os.environ.get("TICI_TRES") != "1":
    return importlib.import_module("panda_tici")
  return importlib.import_module("panda")
