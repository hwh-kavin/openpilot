"""
Shared-memory IPC between amapd (producer) and UI (consumer).

Layout (file-backed mmap under /dev/shm):
  Header (256 bytes) + 2 × RGBA frames (double buffered)
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

SHM_PATH = "/dev/shm/bluepilot_amap_frame"
MAGIC = b"AMAP"
HEADER_SIZE = 256
MAX_W = 1024
MAX_H = 1024
BYTES_PER_PIXEL = 4
FRAME_STRIDE = MAX_W * MAX_H * BYTES_PER_PIXEL
SLOT_COUNT = 2

# Header packing (little-endian)
# magic(4s) version(I) seq(I) active(I) ready(I) gps_valid(I)
# width(I) height(I) bearing(f) request_w(I) request_h(I) enable(I)
# road_name(128s) — UTF-8 address label under GPS arrow
# magic, version, seq, active, ready, gps_valid, width, height, bearing,
# request_w, request_h, enable, road_name
_HEADER_FMT = "<4sIIIIIIIfIII128s"
_HEADER_STRUCT = struct.Struct(_HEADER_FMT)
assert _HEADER_STRUCT.size <= HEADER_SIZE

TOTAL_SIZE = HEADER_SIZE + SLOT_COUNT * FRAME_STRIDE


@dataclass
class AmapFrameHeader:
  version: int = 1
  seq: int = 0
  active: int = 0
  ready: int = 0
  gps_valid: int = 0
  width: int = 0
  height: int = 0
  bearing: float = 0.0
  request_w: int = 0
  request_h: int = 0
  enable: int = 0
  road_name: str = ""


def _utf8_truncate(s: str, max_bytes: int) -> bytes:
  raw = (s or "").encode("utf-8", errors="ignore")
  if len(raw) <= max_bytes:
    return raw
  raw = raw[:max_bytes]
  while raw and (raw[-1] & 0xC0) == 0x80:
    raw = raw[:-1]
  if raw and (raw[-1] & 0xC0) == 0xC0:
    raw = raw[:-1]
  return raw


def _pack_header(h: AmapFrameHeader) -> bytes:
  name = _utf8_truncate(h.road_name, 127)
  raw = _HEADER_STRUCT.pack(
    MAGIC, h.version, h.seq, h.active, h.ready, h.gps_valid,
    h.width, h.height, float(h.bearing),
    h.request_w, h.request_h, h.enable,
    name,
  )
  return raw.ljust(HEADER_SIZE, b"\x00")


def _unpack_header(buf: memoryview | bytes) -> AmapFrameHeader | None:
  data = bytes(buf[:_HEADER_STRUCT.size])
  try:
    (magic, version, seq, active, ready, gps_valid,
     width, height, bearing, request_w, request_h, enable, name) = _HEADER_STRUCT.unpack(data)
  except struct.error:
    return None
  if magic != MAGIC:
    return None
  return AmapFrameHeader(
    version=version, seq=seq, active=active, ready=ready, gps_valid=gps_valid,
    width=width, height=height, bearing=bearing,
    request_w=request_w, request_h=request_h, enable=enable,
    road_name=name.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"),
  )


class AmapFrameShm:
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
    # Initialize empty header if magic missing
    if self._mm[:4] != MAGIC:
      self.write_header(AmapFrameHeader())

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

  def read_header(self) -> AmapFrameHeader | None:
    if self._mm is None:
      return None
    return _unpack_header(self._mm)

  def write_header(self, h: AmapFrameHeader) -> None:
    if self._mm is None:
      return
    self._mm.seek(0)
    self._mm.write(_pack_header(h))

  def slot_offset(self, slot: int) -> int:
    return HEADER_SIZE + (slot % SLOT_COUNT) * FRAME_STRIDE

  def write_frame(self, slot: int, rgba: bytes, width: int, height: int) -> None:
    """Write RGBA frame into slot (must be width*height*4 bytes, width/height <= MAX)."""
    if self._mm is None:
      return
    if width <= 0 or height <= 0 or width > MAX_W or height > MAX_H:
      return
    nbytes = width * height * BYTES_PER_PIXEL
    if len(rgba) < nbytes:
      return
    off = self.slot_offset(slot)
    # Store tightly packed rows into the slot (capacity MAX_W*MAX_H).
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
                    ready: bool, gps_valid: bool, bearing: float, road_name: str,
                    request_w: int, request_h: int, enable: int) -> None:
    """Write next slot and flip active pointer."""
    h = self.read_header() or AmapFrameHeader()
    next_slot = 1 - int(h.active)
    self.write_frame(next_slot, rgba, width, height)
    h.active = next_slot
    h.seq = int(h.seq) + 1
    h.ready = 1 if ready else 0
    h.gps_valid = 1 if gps_valid else 0
    h.width = width
    h.height = height
    h.bearing = float(bearing)
    h.road_name = road_name or ""
    h.request_w = request_w
    h.request_h = request_h
    h.enable = enable
    self.write_header(h)

  def update_request(self, width: int, height: int, enable: bool) -> None:
    """UI-side: request output size / enable rendering."""
    h = self.read_header()
    if h is None:
      # Producer not up yet — create shm so request survives process start.
      if self._mm is None:
        try:
          self._open_create()
        except Exception:
          return
      h = AmapFrameHeader()
    h.request_w = max(1, min(int(width), MAX_W))
    h.request_h = max(1, min(int(height), MAX_H))
    h.enable = 1 if enable else 0
    self.write_header(h)
