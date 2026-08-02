"""
Shared-memory IPC between carlifed (producer) and UI (consumer).

Layout (file-backed mmap under /dev/shm):
  Header (256 bytes) + 2 × RGBA frames (double buffered)

Frames are phone map screenshots — already oriented; UI blits directly
(no rotation / crop).
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

SHM_PATH = "/dev/shm/bluepilot_carlife_frame"
STATUS_PATH = "/dev/shm/bluepilot_carlife_status.json"
MAGIC = b"CLMP"
HEADER_SIZE = 256
MAX_W = 1024
MAX_H = 1024
BYTES_PER_PIXEL = 4
FRAME_STRIDE = MAX_W * MAX_H * BYTES_PER_PIXEL
SLOT_COUNT = 2

# magic(4s) version(I) seq(I) active(I) ready(I)
# width(I) height(I) timestamp_ms(Q) enable(I)
_HEADER_FMT = "<4sIIIIIIQI"
_HEADER_STRUCT = struct.Struct(_HEADER_FMT)
assert _HEADER_STRUCT.size <= HEADER_SIZE

TOTAL_SIZE = HEADER_SIZE + SLOT_COUNT * FRAME_STRIDE


@dataclass
class CarLifeFrameHeader:
  version: int = 1
  seq: int = 0
  active: int = 0
  ready: int = 0
  width: int = 0
  height: int = 0
  timestamp_ms: int = 0
  enable: int = 0


def _pack_header(h: CarLifeFrameHeader) -> bytes:
  raw = _HEADER_STRUCT.pack(
    MAGIC, h.version, h.seq, h.active, h.ready,
    h.width, h.height, int(h.timestamp_ms), h.enable,
  )
  return raw.ljust(HEADER_SIZE, b"\x00")


def _unpack_header(buf: memoryview | bytes) -> CarLifeFrameHeader | None:
  data = bytes(buf[:_HEADER_STRUCT.size])
  try:
    (magic, version, seq, active, ready,
     width, height, timestamp_ms, enable) = _HEADER_STRUCT.unpack(data)
  except struct.error:
    return None
  if magic != MAGIC:
    return None
  return CarLifeFrameHeader(
    version=version, seq=seq, active=active, ready=ready,
    width=width, height=height, timestamp_ms=timestamp_ms, enable=enable,
  )


class CarLifeFrameShm:
  def __init__(self, create: bool = False):
    self._fd: int | None = None
    self._mm: mmap.mmap | None = None
    if create:
      self._open_create()
    else:
      self._open_existing()

  def _open_create(self) -> None:
    fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o666)
    try:
      os.ftruncate(fd, TOTAL_SIZE)
      mm = mmap.mmap(fd, TOTAL_SIZE)
    except Exception:
      os.close(fd)
      raise
    self._fd = fd
    self._mm = mm
    if self._mm[:4] != MAGIC:
      self.write_header(CarLifeFrameHeader())

  def _open_existing(self) -> None:
    if not os.path.exists(SHM_PATH):
      return
    try:
      fd = os.open(SHM_PATH, os.O_RDWR)
      mm = mmap.mmap(fd, TOTAL_SIZE)
    except Exception:
      return
    self._fd = fd
    self._mm = mm

  @property
  def available(self) -> bool:
    return self._mm is not None

  def close(self) -> None:
    if self._mm is not None:
      self._mm.close()
      self._mm = None
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None

  def read_header(self) -> CarLifeFrameHeader | None:
    if self._mm is None:
      return None
    return _unpack_header(self._mm)

  def write_header(self, h: CarLifeFrameHeader) -> None:
    if self._mm is None:
      return
    self._mm.seek(0)
    self._mm.write(_pack_header(h))

  def slot_offset(self, slot: int) -> int:
    return HEADER_SIZE + (slot % SLOT_COUNT) * FRAME_STRIDE

  def write_frame(self, slot: int, rgba: bytes, width: int, height: int) -> None:
    if self._mm is None:
      return
    if width <= 0 or height <= 0 or width > MAX_W or height > MAX_H:
      return
    nbytes = width * height * BYTES_PER_PIXEL
    if len(rgba) < nbytes:
      return
    off = self.slot_offset(slot)
    self._mm[off:off + nbytes] = rgba[:nbytes]

  def read_frame_bytes(self, slot: int, width: int, height: int) -> memoryview | None:
    if self._mm is None:
      return None
    if width <= 0 or height <= 0 or width > MAX_W or height > MAX_H:
      return None
    nbytes = width * height * BYTES_PER_PIXEL
    off = self.slot_offset(slot)
    return memoryview(self._mm)[off:off + nbytes]

  def publish_frame(self, rgba: bytes, width: int, height: int, *,
                    timestamp_ms: int, ready: bool, enable: int) -> None:
    h = self.read_header() or CarLifeFrameHeader()
    next_slot = 1 - int(h.active)
    self.write_frame(next_slot, rgba, width, height)
    h.active = next_slot
    h.seq = int(h.seq) + 1
    h.ready = 1 if ready else 0
    h.width = width
    h.height = height
    h.timestamp_ms = int(timestamp_ms)
    h.enable = enable
    self.write_header(h)

  def update_enable(self, enable: bool) -> None:
    h = self.read_header()
    if h is None:
      if self._mm is None:
        try:
          self._open_create()
        except Exception:
          return
      h = CarLifeFrameHeader()
    h.enable = 1 if enable else 0
    self.write_header(h)
