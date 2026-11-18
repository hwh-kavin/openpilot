#!/usr/bin/env python3
import sys
import pathlib
import codecs
import pickle
from typing import Any

from tinygrad.nn.onnx import Domain, OnnxPBParser


class MetadataOnnxPBParser(OnnxPBParser):
  def _parse_ModelProto(self) -> dict:
    obj: dict[str, Any] = {"graph": {"input": [], "output": []}, "metadata_props": []}

    # tinygrad now populates self.opset_imports inside its own _parse_ModelProto, which we override.
    # Without repeating that pass _parse_NodeProto raises AttributeError on a missing opset_imports.
    opset_import: list[dict] = []
    for fid, wire_type in self._parse_message(self.reader.len):
      if fid == 8:
        opset_import.append(self._parse_OperatorSetIdProto())
      else:
        self.reader.skip_field(wire_type)
    self.opset_imports = {Domain.from_onnx(x.get("domain")): x.get("version", 1) for x in opset_import}
    self.reader.seek(0)

    for fid, wire_type in self._parse_message(self.reader.len):
      match fid:
        case 7:
          obj["graph"] = self._parse_GraphProto()
        case 14:
          obj["metadata_props"].append(self._parse_StringStringEntryProto())
        case _:
          self.reader.skip_field(wire_type)
    return obj


def get_name_and_shape(value_info: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
  shape = tuple(int(dim) if isinstance(dim, int) else 0 for dim in value_info["parsed_type"].shape)
  name = value_info["name"]
  return name, shape


def get_metadata_value_by_name(model: dict[str, Any], name: str) -> str | Any:
  for prop in model["metadata_props"]:
    if prop["key"] == name:
      return prop["value"]
  return None


def make_metadata_dict(model_path):
  model = MetadataOnnxPBParser(model_path).parse()
  output_slices = get_metadata_value_by_name(model, 'output_slices')
  assert output_slices is not None, 'output_slices not found in metadata'
  return {
    'model_checkpoint': get_metadata_value_by_name(model, 'model_checkpoint'),
    'output_slices': pickle.loads(codecs.decode(output_slices.encode(), "base64")),
    'input_shapes': dict(get_name_and_shape(x) for x in model["graph"]["input"]),
    'output_shapes': dict(get_name_and_shape(x) for x in model["graph"]["output"]),
  }


if __name__ == "__main__":
  model_path = pathlib.Path(sys.argv[1])
  metadata_path = model_path.parent / (model_path.stem + '_metadata.pkl')
  with open(metadata_path, 'wb') as f:
    pickle.dump(make_metadata_dict(model_path), f)
  print(f'saved metadata to {metadata_path}')
