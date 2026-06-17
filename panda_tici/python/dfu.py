import os
import usb1
import struct
import binascii

from .base import BaseSTBootloaderHandle
from .spi import STBootloaderSPIHandle, PandaSpiException
from .usb import STBootloaderUSBHandle
from .constants import FW_PATH, McuType

# BluePilot: comma three (C3) internal panda is reached over SPI in bootloader mode
try:
  from openpilot.system.hardware import TICI
except ImportError:
  TICI = False
# End BluePilot


class PandaDFU:
  def __init__(self, dfu_serial: str | None):
    # BluePilot: prefer SPI on C3; USB DFU can mis-identify H7 red panda as F4
    handle: BaseSTBootloaderHandle | None
    self._context = None
    if TICI:
      self._context, handle = PandaDFU.spi_connect(dfu_serial)
      if handle is None:
        self._context, handle = PandaDFU.usb_connect(dfu_serial)
    else:
      self._context, handle = PandaDFU.usb_connect(dfu_serial)
      if handle is None:
        self._context, handle = PandaDFU.spi_connect(dfu_serial)
    # End BluePilot

    if handle is None:
      raise Exception(f"failed to open DFU device {dfu_serial}")

    self._handle: BaseSTBootloaderHandle = handle
    self._mcu_type: McuType = self._handle.get_mcu_type()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def close(self):
    if self._handle is not None:
      self._handle.close()
      self._handle = None
      if self._context is not None:
        self._context.close()

  @staticmethod
  def usb_connect(dfu_serial: str | None):
    handle = None
    context = usb1.USBContext()
    context.open()
    for device in context.getDeviceList(skip_on_error=True):
      if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
        try:
          this_dfu_serial = device.open().getASCIIStringDescriptor(3)
        except Exception:
          continue

        if this_dfu_serial == dfu_serial or dfu_serial is None:
          handle = STBootloaderUSBHandle(device, device.open())
          break

    return context, handle

  @staticmethod
  def spi_connect(dfu_serial: str | None):
    handle = None
    this_dfu_serial = None

    try:
      handle = STBootloaderSPIHandle()
      this_dfu_serial = PandaDFU.st_serial_to_dfu_serial(handle.get_uid(), handle.get_mcu_type())
    except PandaSpiException:
      handle = None

    if dfu_serial is not None and dfu_serial != this_dfu_serial:
      handle = None

    return None, handle

  @staticmethod
  def usb_list() -> list[str]:
    dfu_serials = []
    try:
      with usb1.USBContext() as context:
        for device in context.getDeviceList(skip_on_error=True):
          if device.getVendorID() == 0x0483 and device.getProductID() == 0xdf11:
            try:
              dfu_serials.append(device.open().getASCIIStringDescriptor(3))
            except Exception:
              pass
    except Exception:
      pass
    return dfu_serials

  @staticmethod
  def spi_list() -> list[str]:
    try:
      _, h = PandaDFU.spi_connect(None)
      if h is not None:
        dfu_serial = PandaDFU.st_serial_to_dfu_serial(h.get_uid(), h.get_mcu_type())
        return [dfu_serial, ]
    except PandaSpiException:
      pass
    return []

  @staticmethod
  def st_serial_to_dfu_serial(st: str, mcu_type: McuType = McuType.F4):
    if st is None or st == "none":
      return None
    try:
      uid_base = struct.unpack("H" * 6, bytes.fromhex(st))
      if mcu_type == McuType.H7:
        return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4], uid_base[3])).upper().decode("utf-8")
      else:
        return binascii.hexlify(struct.pack("!HHH", uid_base[1] + uid_base[5], uid_base[0] + uid_base[4] + 0xA, uid_base[3])).upper().decode("utf-8")
    except struct.error:
      return None

  def get_mcu_type(self) -> McuType:
    return self._mcu_type

  def reset(self):
    self._handle.jump(self._mcu_type.config.bootstub_address)

  def program_bootstub(self, code_bootstub):
    self._handle.clear_status()

    # erase bootstub + app sectors
    for i in (0, 1):
      self._handle.erase_sector(i)

    # write bootstub
    self._handle.program(self._mcu_type.config.bootstub_address, code_bootstub)

  def recover(self):
    fn = os.path.join(FW_PATH, self._mcu_type.config.bootstub_fn)
    # BluePilot: C3 builds may only ship H7 bootstubs; USB DFU can select F4 paths
    if not os.path.isfile(fn) and self._mcu_type == McuType.F4:
      fn = os.path.join(FW_PATH, McuType.H7.config.bootstub_fn)
      if os.path.isfile(fn):
        self._mcu_type = McuType.H7
    # End BluePilot
    with open(fn, "rb") as f:
      code = f.read()
    self.program_bootstub(code)
    self.reset()

  @staticmethod
  def list() -> list[str]:
    # BluePilot: internal C3 panda is on SPI; list it before USB DFU devices
    if TICI:
      ret = PandaDFU.spi_list()
      ret += PandaDFU.usb_list()
    else:
      ret = PandaDFU.usb_list()
      ret += PandaDFU.spi_list()
    # End BluePilot
    return list(set(ret))
