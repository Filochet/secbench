###
# Copyright CEA (Commissariat à l'énergie atomique et aux
# énergies alternatives) (2017-2025)
#
# This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
###

"""
PS2000A driver using direct library calls
"""

from ctypes import (
    c_char_p,
    c_double,
    c_float,
    c_int16,
    c_int32,
    c_uint32,
    c_uint64,
    c_void_p,
)
from enum import IntEnum

from .common import def_symbol, load_libps


class PicoInfo(IntEnum):
    """
    Information to be requested to `ps2000aGetUnitInfo`.
    """

    DRIVER_VERSION = 0x00000000
    USB_VERSION = 0x00000001
    HARDWARE_VERSION = 0x00000002
    VARIANT_INFO = 0x00000003
    BATCH_AND_SERIAL = 0x00000004
    CAL_DATE = 0x00000005
    KERNEL_VERSION = 0x00000006
    DIGITAL_HARDWARE_VERSION = 0x00000007
    ANALOGUE_HARDWARE_VERSION = 0x00000008
    FIRMWARE_VERSION_1 = 0x00000009
    FIRMWARE_VERSION_2 = 0x0000000A


def _load_ps2000a_lib():
    lib = load_libps("ps2000a")
    def_symbol(lib, "ps2000aOpenUnit", c_uint32, [c_void_p, c_char_p])
    def_symbol(lib, "ps2000aOpenUnitAsync", c_uint32, [c_void_p, c_char_p])

    def_symbol(lib, "ps2000aOpenUnitProgress", c_uint32, [c_void_p, c_void_p, c_void_p])

    def_symbol(
        lib,
        "ps2000aGetUnitInfo",
        c_uint32,
        [c_int16, c_void_p, c_int16, c_void_p, c_uint32],
    )

    def_symbol(lib, "ps2000aFlashLed", c_uint32, [c_int16, c_int16])

    def_symbol(
        lib,
        "ps2000aCloseUnit",
        c_uint32,
        [
            c_int16,
        ],
    )

    def_symbol(lib, "ps2000aMemorySegments", c_uint32, [c_int16, c_uint32, c_void_p])

    def_symbol(
        lib,
        "ps2000aSetChannel",
        c_uint32,
        [c_int16, c_int32, c_int16, c_int32, c_int32, c_float],
    )

    def_symbol(
        lib, "ps2000aSetDigitalPort", c_uint32, [c_int16, c_int32, c_int16, c_int16]
    )

    def_symbol(lib, "ps2000aSetNoOfCaptures", c_uint32, [c_int16, c_uint32])
    def_symbol(
        lib,
        "ps2000aGetTimebase",
        c_uint32,
        [c_int16, c_uint32, c_int32, c_void_p, c_int16, c_void_p, c_uint32],
    )
    def_symbol(
        lib,
        "ps2000aGetTimebase2",
        c_uint32,
        [c_int16, c_uint32, c_int32, c_void_p, c_int16, c_void_p, c_uint32],
    )
    def_symbol(
        lib,
        "ps2000aSetSigGenArbitrary",
        c_uint32,
        [
            c_int16,
            c_int32,
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            c_void_p,
            c_int32,
            c_int32,
            c_int32,
            c_int32,
            c_uint32,
            c_uint32,
            c_int32,
            c_int32,
            c_int16,
        ],
    )
    def_symbol(
        lib,
        "ps2000aSetSigGenBuiltIn",
        c_uint32,
        [
            c_int16,
            c_int32,
            c_uint32,
            c_int16,
            c_float,
            c_float,
            c_float,
            c_float,
            c_int32,
            c_int32,
            c_uint32,
            c_uint32,
            c_int32,
            c_int32,
            c_int16,
        ],
    )
    def_symbol(
        lib,
        "ps2000aSetSigGenPropertiesArbitrary",
        c_uint32,
        [
            c_int16,
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            c_int32,
            c_uint32,
            c_uint32,
            c_int32,
            c_int32,
            c_int16,
        ],
    )
    def_symbol(
        lib,
        "ps2000aSetSigGenPropertiesBuiltIn",
        c_uint32,
        [
            c_int16,
            c_double,
            c_double,
            c_double,
            c_double,
            c_int32,
            c_uint32,
            c_uint32,
            c_int32,
            c_int32,
            c_int16,
        ],
    )
    def_symbol(
        lib,
        "ps2000aSigGenFrequencyToPhase",
        c_uint32,
        [c_int16, c_double, c_int32, c_uint32, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aSigGenArbitraryMinMaxValues",
        c_uint32,
        [c_int16, c_void_p, c_void_p, c_void_p, c_void_p],
    )
    def_symbol(lib, "ps2000aSigGenSoftwareControl", c_uint32, [c_int16, c_int16])
    def_symbol(
        lib, "ps2000aSetEts", c_uint32, [c_int16, c_int32, c_int16, c_int16, c_void_p]
    )
    def_symbol(
        lib,
        "ps2000aSetSimpleTrigger",
        c_uint32,
        [c_int16, c_int16, c_int32, c_int16, c_int32, c_uint32, c_int16],
    )
    def_symbol(
        lib,
        "ps2000aSetTriggerDigitalPortProperties",
        c_uint32,
        [c_int16, c_void_p, c_int16],
    )
    def_symbol(
        lib,
        "ps2000aSetDigitalAnalogTriggerOperand",
        c_uint32,
        [c_int16, c_int32],
    )
    def_symbol(
        lib,
        "ps2000aSetTriggerChannelProperties",
        c_uint32,
        [c_int16, c_void_p, c_int16, c_int16, c_int32],
    )
    def_symbol(
        lib,
        "ps2000aSetTriggerChannelConditions",
        c_uint32,
        [c_int16, c_void_p, c_int16],
    )
    def_symbol(
        lib,
        "ps2000aSetTriggerChannelDirections",
        c_uint32,
        [c_int16, c_int32, c_int32, c_int32, c_int32, c_int32, c_int32],
    )
    def_symbol(lib, "ps2000aSetTriggerDelay", c_uint32, [c_int16, c_uint32])
    def_symbol(
        lib,
        "ps2000aSetPulseWidthQualifier",
        c_uint32,
        [c_int16, c_void_p, c_int16, c_int32, c_uint32, c_uint32, c_int32],
    )
    def_symbol(
        lib,
        "ps2000aIsTriggerOrPulseWidthQualifierEnabled",
        c_uint32,
        [c_int16, c_void_p, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aGetTriggerTimeOffset64",
        c_uint32,
        [c_int16, c_void_p, c_void_p, c_uint32],
    )
    def_symbol(
        lib,
        "ps2000aGetValuesTriggerTimeOffsetBulk64",
        c_uint32,
        [c_int16, c_void_p, c_void_p, c_uint32, c_uint32],
    )
    def_symbol(lib, "ps2000aGetNoOfCaptures", c_uint32, [c_int16, c_void_p])
    def_symbol(lib, "ps2000aGetNoOfProcessedCaptures", c_uint32, [c_int16, c_void_p])
    def_symbol(
        lib,
        "ps2000aSetDataBuffer",
        c_uint32,
        [c_int16, c_int32, c_void_p, c_int32, c_uint32, c_int32],
    )
    def_symbol(
        lib,
        "ps2000aSetDataBuffers",
        c_uint32,
        [c_int16, c_int32, c_void_p, c_void_p, c_int32, c_uint32, c_int32],
    )
    def_symbol(lib, "ps2000aSetEtsTimeBuffer", c_uint32, [c_int16, c_void_p, c_int32])
    def_symbol(lib, "ps2000aIsReady", c_uint32, [c_int16, c_void_p])
    def_symbol(
        lib,
        "ps2000aRunBlock",
        c_uint32,
        [
            c_int16,
            c_int32,
            c_int32,
            c_uint32,
            c_int16,
            c_void_p,
            c_uint32,
            c_void_p,
            c_void_p,
        ],
    )
    def_symbol(
        lib,
        "ps2000aRunStreaming",
        c_uint32,
        [
            c_int16,
            c_void_p,
            c_int32,
            c_uint32,
            c_uint32,
            c_int16,
            c_uint32,
            c_int32,
            c_uint32,
        ],
    )
    def_symbol(
        lib, "ps2000aGetStreamingLatestValues", c_uint32, [c_int16, c_void_p, c_void_p]
    )
    def_symbol(
        lib,
        "ps2000aGetMaxDownSampleRatio",
        c_uint32,
        [c_int16, c_uint32, c_void_p, c_int32, c_uint32],
    )
    def_symbol(
        lib,
        "ps2000aGetValues",
        c_uint32,
        [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_void_p],
    )

    def_symbol(
        lib,
        "ps2000aGetValuesBulk",
        c_uint32,
        [c_int16, c_void_p, c_uint32, c_uint32, c_uint32, c_int32, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aGetValuesAsync",
        c_uint32,
        [c_int16, c_uint32, c_uint32, c_uint32, c_int32, c_uint32, c_void_p, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aGetValuesOverlapped",
        c_uint32,
        [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aGetValuesOverlappedBulk",
        c_uint32,
        [c_int16, c_uint32, c_void_p, c_uint32, c_int32, c_uint32, c_int32, c_void_p],
    )
    def_symbol(
        lib,
        "ps2000aStop",
        c_uint32,
        [
            c_int16,
        ],
    )
    def_symbol(lib, "ps2000aHoldOff", c_uint32, [c_int16, c_uint64, c_int32])
    def_symbol(
        lib,
        "ps2000aGetChannelInformation",
        c_uint32,
        [c_int16, c_int32, c_int32, c_int32, c_int32, c_int32],
    )
    def_symbol(lib, "ps2000aEnumerateUnits", c_uint32, [c_void_p, c_void_p, c_void_p])
    def_symbol(
        lib,
        "ps2000aPingUnit",
        c_uint32,
        [
            c_int16,
        ],
    )
    def_symbol(lib, "ps2000aMaximumValue", c_uint32, [c_int16, c_void_p])
    def_symbol(lib, "ps2000aMinimumValue", c_uint32, [c_int16, c_void_p])
    def_symbol(
        lib,
        "ps2000aGetAnalogueOffset",
        c_uint32,
        [c_int16, c_int32, c_int32, c_void_p, c_void_p],
    )
    def_symbol(lib, "ps2000aGetMaxSegments", c_uint32, [c_int16, c_void_p])

    return lib


_LIB_PS2000A = None


def load_ps2000a():
    global _LIB_PS2000A
    if _LIB_PS2000A is None:
        _LIB_PS2000A = _load_ps2000a_lib()
        return _LIB_PS2000A
    else:
        return _LIB_PS2000A