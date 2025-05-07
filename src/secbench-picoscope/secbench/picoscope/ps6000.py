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
from secbench.picoscope.lib.ps6000 import PicoInfo, load_ps6000

from .base import Picobase, PicobaseAnalogChannel, PicoHandles
from .error import PicoscopeApiError, pico_check
from .types import VerticalRange

logger = logging.getLogger(__name__)


class PicoPS6000Scope(Discoverable, Picobase):
    _PICO_ALL_CHANNELS = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "External": 4,
        "TriggerAux": 5,
    }
    _PICO_CHANNELS = {"A": 0, "B": 1, "C": 2, "D": 3}
    _PICO_VERTICAL_RANGES = [
        # Does not seems supported.
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
    _PICO_SLOPES = {
        Slope.rising: 2,
        Slope.falling: 3,
        Slope.either: 4,
    }
    _MAX_SAMPLES = {
        "6402C": 256 * 1024 * 1024,
        "6402D": 512 * 1024 * 1024,
        "6403C": 512 * 1024 * 1024,
        "6403D": 1024 * 1024 * 1024,
        "6404C": 1024 * 1024 * 1024,
        "6404D": 2 * 1024 * 1024 * 1024,
    }

    @staticmethod
    def ps6000_methods(lib):
        handles = PicoHandles(
            psOpenUnit=lib.ps6000OpenUnit,
            psCloseUnit=lib.ps6000CloseUnit,
            psIsReady=lib.ps6000IsReady,
            psGetUnitInfo=lib.ps6000GetUnitInfo,
            psGetTimebase2=lib.ps6000GetTimebase2,
            psStop=lib.ps6000Stop,
            psSetChannel=lib.ps6000SetChannel,
            psSetNoOfCaptures=lib.ps6000SetNoOfCaptures,
            psRunBlock=lib.ps6000RunBlock,
            psSetDataBuffer=lib.ps6000SetDataBuffer,
            psSetDataBufferBulk=lib.ps6000SetDataBufferBulk,
            psGetValues=lib.ps6000GetValues,
            psGetValuesBulk=lib.ps6000GetValuesBulk,
            psSetSimpleTrigger=lib.ps6000SetSimpleTrigger,
            psMemorySegments=lib.ps6000MemorySegments,
            psPingUnit=lib.ps6000PingUnit,
        )
        return handles

    def __init__(self, serial_number=None):
        lib = load_ps6000()
        methods = self.ps6000_methods(lib)
        super().__init__(
            lib, methods, device_info_cls=PicoInfo, serial_number=serial_number
        )
        self._channels = OrderedDict(
            [
                ("A", PicobaseAnalogChannel(self, "A")),
                ("B", PicobaseAnalogChannel(self, "B")),
                ("C", PicobaseAnalogChannel(self, "C")),
                ("D", PicobaseAnalogChannel(self, "D")),
            ]
        )

    # ===
    # Picobase interface
    # ===
    def _pico_adc_range(self) -> Tuple[int, int]:
        return -32_512, 32_512

    @staticmethod
    def _pico_interval_to_timebase(interval: float) -> int:
        max_sample_time = ((2**32 - 1) - 4) / 156_250_000

        if interval < 6.4e-9:
            timebase = math.floor(math.log(interval * 5e9, 2))
            timebase = max(timebase, 0)
        else:
            # Otherwise in range 2^32-1
            if interval > max_sample_time:
                interval = max_sample_time
            timebase = math.floor((interval * 156_250_000) + 4)
        return int(timebase)

    @staticmethod
    def _pico_timebase_to_interval(timebase: int) -> float:
        assert isinstance(timebase, int), "timebase is an integer"
        if 0 <= timebase <= 4:
            return (1 << timebase) / 5e9
        elif 5 <= timebase <= (1 << 32) - 1:
            return (timebase - 4) / 156_250_000
        else:
            # FIXME(@TH): use a more specific error.
            raise ValueError("invalid timebase: {}".format(timebase))

    # ===
    # Discoverable Interface
    # ===
    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        try:
            _ = load_ps6000()
            return True
        except Exception as e:
            logger.debug(
                f"unable to load libps6000, PS2000A devices cannot be discovered (exception is {e})"
            )
        return False

    @classmethod
    def discover(cls, hw_info: HardwareInfo) -> Iterable[str]:
        try:
            lib = load_ps6000()
            count = ctypes.c_int16(0)
            serial_len = ctypes.c_int16(4096)
            serial_buff = (ctypes.c_int8 * serial_len.value)()
            pico_check(
                lib.ps6000EnumerateUnits(
                    ctypes.byref(count), serial_buff, ctypes.byref(serial_len)
                )
            )

            serial_ids = bytes(serial_buff[: serial_len.value - 1]).decode()
            logger.debug(f"picoscope devices available: {serial_ids}")
            for serial_num in serial_ids.split(","):
                yield serial_num
        except PicoscopeApiError as e:
            logger.info(f"no ps6000 device found: {e}, leaving...")

    @classmethod
    def build(cls, hardware_info: HardwareInfo, serial_number: str):
        return cls(serial_number=None)

    # ===
    # Scope interface
    # ===

    @property
    def description(self) -> str:
        info = self.device_info()
        variant = info.get(PicoInfo.VARIANT_INFO, "?")
        serial = info.get(PicoInfo.BATCH_AND_SERIAL, "?")
        return f"picoscope-PS6000[variant={variant}, serial={serial}]"

    def channels(self) -> Mapping[str, ScopeAnalogChannel]:
        return self._channels