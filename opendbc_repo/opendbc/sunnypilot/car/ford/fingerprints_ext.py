"""
BluePilot Ford firmware version extensions.

Contains FW_VERSIONS for BluePilot-only Ford platforms (Ford Edge MK2, Ford Mondeo MK5).
Merged into the main FW_VERSIONS dict at module load time in
opendbc/car/ford/fingerprints.py via merge_fw_versions().
"""

from opendbc.car.ford.values import CAR
from opendbc.car.structs import CarParams

Ecu = CarParams.Ecu

# BluePilot-only Ford platform firmware versions
FW_VERSIONS_EXT = {
  CAR.FORD_EDGE_MK2: {
    (Ecu.eps, 0x730, None): [
      b'M2GC-14D003-AA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'M2GC-2D053-CB\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
      b'M2GC-2D053-EA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'JX7T-14D049-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdCamera, 0x706, None): [
      b'KT4T-14F397-AF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
  },
  CAR.FORD_MONDEO_MK5: {
    # BluePilot: the original PR (#135) included several non-ASCII byte strings alongside each
    # legit part number below -- e.g. a bare b'U', 0xff-padded blobs -- that look like NAK/error
    # responses captured verbatim rather than real FW versions. Short/degenerate entries like
    # those can spuriously match other Ford vehicles' unrelated ECU responses, making the overall
    # fingerprint ambiguous and forcing manual selection. Removed; keeping only the part numbers.
    (Ecu.fwdCamera, 0x706, None): [
      b'KT4T-14F397-AE\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.abs, 0x760, None): [
      b'KG9C-2D053-DF\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.eps, 0x730, None): [
      b'K2GC-14D003-AJ\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.fwdRadar, 0x764, None): [
      b'JX7T-14D049-AD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.engine, 0x7e0, None): [
      b'HS7A-14C204-CJD\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    (Ecu.debug, 0x7d0, None): [
      b'1U5T-14G374-DA\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
    ],
    # BluePilot: removed an (Ecu.adas, 0x730, None) entry here -- b'\xf1\x10DS-K2GC-3F964-AG\x00...',
    # 26 bytes with a non-ASCII \xf1\x10 prefix, same NAK/error-response signature as the garbage
    # already stripped from this file (see the comment above). Mondeo is low-volume and non-US;
    # manual vehicle selection remains available regardless.
  }
}
