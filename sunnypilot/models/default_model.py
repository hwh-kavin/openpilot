import argparse
import os
import hashlib
import requests
import re

from openpilot.common.basedir import BASEDIR
from openpilot.sunnypilot import get_file_hash
from openpilot.sunnypilot.models.model_name import DEFAULT_MODEL, DEFAULT_BIG_MODEL
from openpilot.sunnypilot.models.fetcher import ModelFetcher


def get_default_model() -> str:
  # BluePilot: this fork has no chestnut hardware, so the default is always the small model
  return DEFAULT_MODEL


DEFAULT_MODEL_NAME_PATH = os.path.join(BASEDIR, "sunnypilot", "models", "model_name.py")
MODEL_HASH_PATH = os.path.join(BASEDIR, "sunnypilot", "models", "tests", "model_hash")
VISION_ONNX_PATH = os.path.join(BASEDIR, "selfdrive", "modeld", "models", "driving_vision.onnx")
POLICY_ONNX_PATH = os.path.join(BASEDIR, "selfdrive", "modeld", "models", "driving_on_policy.onnx")


def _read_model_name_fields():
  with open(DEFAULT_MODEL_NAME_PATH) as f:
    content = f.read()
  fields = {}
  for line in content.splitlines():
    if "=" in line:
      key, val = line.split("=", 1)
      fields[key.strip()] = val.strip().strip('"')
  return fields


def update_model_hash():
  fields = _read_model_name_fields()
  # BluePilot: the default model here is compiled from the vision + policy ONNX pair
  vision_hash = get_file_hash(VISION_ONNX_PATH)
  policy_hash = get_file_hash(POLICY_ONNX_PATH)
  fingerprint = f"{vision_hash}:{policy_hash}:{fields.get('DEFAULT_MODEL', '')}:{fields.get('DEFAULT_MODEL_REF', '')}"
  combined_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

  with open(MODEL_HASH_PATH, "w") as f:
    f.write(combined_hash)

  print(f"Generated and updated new combined model hash to {MODEL_HASH_PATH}")


def get_ref_for_name(url: str, name: str) -> str:
  response = requests.get(url, timeout=10)
  if response.status_code == 200:
    bundles = response.json()["bundles"]
    matching = [b for b in bundles if re.search(name, f"{b['short_name']} {b['display_name']}", re.IGNORECASE)]
    if matching:
      return max(matching, key=lambda b: int(b["index"]))["ref"]
  return ""


def update_default_model_names(default_model_name: str, default_big_model_name: str):
  print("[CHANGE DEFAULT MODEL NAMES]")
  small_ref = get_ref_for_name(ModelFetcher.MODEL_URL, default_model_name)
  # BluePilot: no chestnut catalog here, resolve the big model from the qcom catalog if present
  big_ref = get_ref_for_name(ModelFetcher.MODEL_URL, default_big_model_name)

  with open(DEFAULT_MODEL_NAME_PATH, "w") as f:
    f.write(f'DEFAULT_MODEL = "{default_model_name}"\n')
    f.write(f'DEFAULT_MODEL_REF = "{small_ref}"\n')
    f.write(f'DEFAULT_BIG_MODEL = "{default_big_model_name}"\n')
    f.write(f'DEFAULT_BIG_MODEL_REF = "{big_ref}"\n')

  print(f'New default small model name: "{default_model_name}" (ref: {small_ref})')
  print(f'New default big model name: "{default_big_model_name}" (ref: {big_ref})')
  print("[DONE]")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Update default model names and hash")
  parser.add_argument("--new_small_model_name", type=str, help="New default small model name")
  parser.add_argument("--new_big_model_name", type=str, help="New default big model name")
  args = parser.parse_args()

  if args.new_small_model_name is None and args.new_big_model_name is None:
    new_name = input(f'Enter new default small model name (current: "{DEFAULT_MODEL}", leave empty to keep): ').strip()
    new_big_model_name = input(f'Enter new default big model name (current: "{DEFAULT_BIG_MODEL}", leave empty to keep): ').strip()
  else:
    new_name, new_big_model_name = args.new_small_model_name, args.new_big_model_name

  update_default_model_names(new_name or DEFAULT_MODEL, new_big_model_name or DEFAULT_BIG_MODEL)
  update_model_hash()
