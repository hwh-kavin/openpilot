"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import hashlib

from openpilot.sunnypilot import get_file_hash
from openpilot.sunnypilot.models.default_model import MODEL_HASH_PATH, VISION_ONNX_PATH, POLICY_ONNX_PATH, \
                                                      _read_model_name_fields


def test_compare_onnx_hashes():
  # BluePilot: the default model is the vision + policy ONNX pair (no supercombo / big pkl here)
  fields = _read_model_name_fields()
  fingerprint = f"{get_file_hash(VISION_ONNX_PATH)}:{get_file_hash(POLICY_ONNX_PATH)}:" + \
                f"{fields.get('DEFAULT_MODEL', '')}:{fields.get('DEFAULT_MODEL_REF', '')}"
  combined_hash = hashlib.sha256(fingerprint.encode()).hexdigest()

  with open(MODEL_HASH_PATH) as f:
    current_hash = f.read().strip()

  assert combined_hash == current_hash, "Run sunnypilot/models/default_model.py to update the default model name and hash"

