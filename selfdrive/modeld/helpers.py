import json
import os
import struct
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
USBGPU_VID = 0xADD1
USBGPU_PID = 0x0001


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  prefix = 'big_' if usbgpu else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def usbgpu_present() -> bool:
  for d in Path("/sys/bus/usb/devices").glob("*"):
    try:
      if int((d / "idVendor").read_text(), 16) == USBGPU_VID and \
          int((d / "idProduct").read_text(), 16) == USBGPU_PID:
        return True
    except Exception:
      pass
  return False


def chestnut_present() -> bool:
  """BluePilot: no chestnut (external big-model accelerator) on this fork."""
  return False


def artifact_file(path) -> str:
  """Return a readable file for a build artifact, which may have been split into chunks.

  The build splits large artifacts into `<name>.chunkNNofMM` next to a `.chunkmanifest` and removes
  the original file (see common/file_chunker.py), while load_pickle() wants a single readable file.
  A single-chunk artifact is used in place; multi-chunk ones are stitched together next to the
  chunks and reused on later starts.

  The stitched copy must not go to /tmp: on AGNOS that is a small tmpfs and a driving artifact is
  ~80 MB, which is enough to fail the load with MemoryError.
  """
  p = Path(path)
  if p.is_file():
    return str(p)

  from openpilot.common.file_chunker import get_chunk_name, get_manifest_path

  manifest = Path(get_manifest_path(str(p)))
  if not manifest.is_file():
    raise FileNotFoundError(str(p))

  num_chunks = int(manifest.read_text().strip())
  if num_chunks == 1:
    return get_chunk_name(str(p), 0, 1)

  chunks = [get_chunk_name(str(p), i, num_chunks) for i in range(num_chunks)]
  cached = p.with_name(p.name + '.reassembled')
  newest_chunk = max(os.path.getmtime(c) for c in chunks)

  if not cached.is_file() or cached.stat().st_mtime < newest_chunk:
    with open(cached, 'wb') as out:
      for chunk in chunks:
        with open(chunk, 'rb') as f:
          while data := f.read(8 * 1024 * 1024):
            out.write(data)

  return str(cached)


def _is_out_of_band(path: str) -> bool:
  """An OOB artifact is an int64 opcode length followed by the pickle stream.

  compile_onnx.py --out-of-band writes that container, while compile_modeld.py still writes a plain
  pickle, so the loader has to look at the header instead of assuming one format.
  """
  with open(path, 'rb') as f:
    head = f.read(8)
    if len(head) < 8:
      return False
    if struct.unpack('<q', head)[0] <= 0:
      return False
    return f.read(1) == b'\x80'


def load_oob(path, chestnut: bool = False):
  """Load a tinygrad model artifact (out-of-band container or plain pickle)."""
  import sys
  from openpilot.common.hardware import TICI
  from tinygrad import Context
  device = 'QCOM' if TICI else 'METAL' if sys.platform == 'darwin' else 'CPU:LLVM'
  artifact = artifact_file(path)
  out_of_band = _is_out_of_band(artifact)
  with Context(DEV=device):
    from tinygrad_repo.examples.openpilot.helpers import load_pickle
    return load_pickle(artifact, out_of_band=out_of_band)
