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

import ctypes
import logging
import math
from collections import OrderedDict
from typing import Iterable, Mapping, Tuple

from secbench.api import Discoverable, HardwareInfo
from secbench.api.enums import Slope
from secbench.api.instrument import ScopeAnalogChannel
from secbench.picoscope.lib.ps2000a import PicoInfo, load_ps2000a

from .base import Picobase, PicobaseAnalogChannel, PicoHandles
from .error import PicoscopeApiError, pico_check
from .types import VerticalRange

logger = logging.getLogger(__name__)


class PicoPS2000AScope(Discoverable, Picobase):
    _PICO_ALL_CHANNELS = {
        "A": 0,
        "B": 1,
    }
    _PICO_CHANNELS = {"A": 0, "B": 1}
    _PICO_SLOPES = {
        Slope.rising: 2,
        Slope.falling: 3,
        Slope.either: 4,
    }

    _PICO_VERTICAL_RANGES = [
        # VerticalRange(10e-3, 0, "10 mV"),
        # VerticalRange(20e-3, 1, "20 mV"),
        VerticalRange(50e-3, 2, "50 mV"),
        VerticalRange(100e-3, 3, "100 mV"),
        VerticalRange(200e-3, 4, "200 mV"),
        VerticalRange(500e-3, 5, "500 mV"),
        VerticalRange(1.0, 6, "1 V"),
        VerticalRange(2.0, 7, "2 V"),
        VerticalRange(5.0, 8, "5 V"),
        VerticalRange(10.0, 9, "10 V"),
        VerticalRange(20.0, 10, "20 V"),
        VerticalRange(50.0, 11, "50 V"),
    ]

    @staticmethod
    def ps2000a_methods(lib):
        def setDataBuffer(handle, channel, buffer, buffer_len, downsample_mode):
            return lib.ps2000aSetDataBuffer(
                handle, channel, buffer, buffer_len, 0, downsample_mode
            )

        def setChannel(handle, channel, enabled, coupling, vrange, offset, bandwidth):
            # drop the bandwidth parameter
            return lib.ps2000aSetChannel(
                handle, channel, enabled, coupling, vrange, offset
            )

        handles = PicoHandles(
            psOpenUnit=lib.ps2000aOpenUnit,
            psCloseUnit=lib.ps2000aCloseUnit,
            psIsReady=lib.ps2000aIsReady,
            psGetUnitInfo=lib.ps2000aGetUnitInfo,
            psGetTimebase2=lib.ps2000aGetTimebase2,
            psSetChannel=setChannel,
            psStop=lib.ps2000aStop,
            psSetNoOfCaptures=lib.ps2000aSetNoOfCaptures,
            psRunBlock=lib.ps2000aRunBlock,
            # psSetDataBuffer=lib.ps2000aSetDataBuffer,
            # Patched setDataBuffer to match PS6000 style.
            psSetDataBuffer=setDataBuffer,
            psSetDataBufferBulk=lib.ps2000aSetDataBuffer,
            psGetValues=lib.ps2000aGetValues,
            psGetValuesBulk=lib.ps2000aGetValuesBulk,
            psSetSimpleTrigger=lib.ps2000aSetSimpleTrigger,
            psMemorySegments=lib.ps2000aMemorySegments,
            psPingUnit=lib.ps2000aPingUnit,
        )
        return handles

    def __init__(self, serial_number=None):
        lib = load_ps2000a()
        methods = self.ps2000a_methods(lib)
        super().__init__(
            lib, methods, device_info_cls=PicoInfo, serial_number=serial_number
        )
        self._channels = OrderedDict(
            [
                ("A", PicobaseAnalogChannel(self, "A")),
                ("B", PicobaseAnalogChannel(self, "B")),
            ]
        )

    # ===
    # Discoverable Interface
    # ===
    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        try:
            _ = load_ps2000a()
            return True
        except Exception as e:
            logger.debug(
                f"unable to load libps2000a, PS2000A devices cannot be discovered (exception is {e})"
            )
        return False

    @classmethod
    def discover(cls, hw_info: HardwareInfo) -> Iterable[str]:
        try:
            lib = load_ps2000a()
            count = ctypes.c_int16(0)
            serial_len = ctypes.c_int16(4096)
            serial_buff = (ctypes.c_int8 * serial_len.value)()
            pico_check(
                lib.ps2000aEnumerateUnits(
                    ctypes.byref(count), serial_buff, ctypes.byref(serial_len)
                )
            )

            serial_ids = bytes(serial_buff[: serial_len.value - 1]).decode()
            logger.debug(f"picoscope devices available: {serial_ids}")
            for serial_num in serial_ids.split(","):
                yield serial_num
        except PicoscopeApiError as e:
            logger.info(f"no ps2000a device found: {e}, leaving...")

    @classmethod
    def build(cls, hardware_info: HardwareInfo, serial_number: str):
        return cls(serial_number=None)

    @staticmethod
    def _pico_interval_to_timebase(interval):
        """
        Convert duration (in seconds) to API code.
        """
        # https://www.picotech.com/download/manuals/ps2000apg.en-6.pdf p.16
        max_interval = (((1 << 32) - 1) - 2) // 125000000
        if interval < 8e-9:
            st = math.floor(math.log(interval * 1e9, 2))
            st = max(st, 0)
        else:
            if interval > max_interval:
                interval = max_interval
            st = math.floor((interval * 125000000) + 2)
        return int(st)

    @staticmethod
    def _pico_timebase_to_interval(timebase):
        """
        Convert API code to duration (in seconds).
        """
        # https://www.picotech.com/download/manuals/ps2000apg.en-6.pdf p.16
        if 0 <= timebase <= 2:
            return (1 << timebase) / 1e9
        elif 3 <= timebase <= (1 << 32) - 1:
            return (timebase - 2.0) / 125000000.0
        else:
            raise ValueError("invalid timebase: {}".format(timebase))

    def _pico_adc_range(self) -> Tuple[int, int]:
        adc_min = ctypes.c_int16(0)
        adc_max = ctypes.c_int16(0)
        pico_check(self._lib.ps2000aMinimumValue(self._handle, ctypes.byref(adc_min)))
        pico_check(self._lib.ps2000aMaximumValue(self._handle, ctypes.byref(adc_max)))
        return int(adc_min.value), int(adc_max.value)

    # ===
    # Scope interface
    # ===

    @property
    def description(self) -> str:
        info = self.device_info()
        variant = info.get(PicoInfo.VARIANT_INFO, "?")
        serial = info.get(PicoInfo.BATCH_AND_SERIAL, "?")
        return f"picoscope-PS2000A[variant={variant}, serial={serial}]"

    def channels(self) -> Mapping[str, ScopeAnalogChannel]:
        return self._channels