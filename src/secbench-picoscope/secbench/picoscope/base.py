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
from __future__ import annotations

import abc
import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from secbench.api.enums import Arithmetic, Coupling, Decimation, Slope
from secbench.api.exceptions import (
    InstrumentError,
    InstrumentUnsupportedFeature,
    InvalidParameter,
    NoSuchChannelError,
)
from secbench.api.instrument import Scope, ScopeAnalogChannel

from .error import PartialConfigurationError, PicoStatus, pico_check
from .types import ChannelState, TimebaseInfo, VerticalRange

logger = logging.getLogger(__name__)


@dataclass
class PicoHandles:
    psOpenUnit: Any
    psCloseUnit: Any
    psStop: Any
    psIsReady: Any
    psGetUnitInfo: Any
    psGetTimebase2: Any
    psSetChannel: Any
    psMemorySegments: Any
    psSetNoOfCaptures: Any
    psRunBlock: Any
    psSetDataBuffer: Any
    psSetDataBufferBulk: Any
    psGetValues: Any
    psGetValuesBulk: Any
    psSetSimpleTrigger: Any
    psPingUnit: Any = None


class PicobaseAnalogChannel(ScopeAnalogChannel):
    def __init__(self, parent: Picobase, name: str):
        self._parent = parent
        self._name = name
        self._state: ChannelState | None = None

    @property
    def parent(self) -> Scope:
        return self._parent

    @property
    def name(self) -> str:
        return self._name

    def state(self) -> ChannelState | None:
        return self._state

    def enabled(self) -> bool:
        if self._state is None:
            return False
        return self._state.enabled

    def coupling(self) -> Coupling:
        return self._state.coupling

    def range(self) -> float:
        return self._state.range

    def offset(self) -> float:
        return self._state.offset

    def decimation(self) -> Decimation:
        return Decimation.sample

    def disable(self) -> None:
        self._parent.pico_setup_channel(
            name=self._name,
            range=1,
            coupling=Coupling.dc,
            offset=0,
            enabled=False,
            decimation=Decimation.sample,
        )
        self._state = None

    def setup(
        self,
        range: float | None = None,
        coupling: Coupling | None = None,
        offset: float | None = None,
        decimation: Decimation = Decimation.sample,
    ) -> None:
        if self._state is not None:
            if range is None:
                range = self._state.range
            if coupling is None:
                coupling = self._state.coupling
            if offset is None:
                offset = self._state.offset
            # FIXME: we do not implement decimation modes currently.
            # if decimation is None:
            #     decimation = self._state.decimation
        else:
            if any(x is None for x in [range, coupling, offset, decimation]):
                raise ValueError(
                    f"Since channel {self._name} was disabled, all parameters (range, coupling, offset,"
                    f" decimation) must be explicitly given to the first setup call."
                )
        self._state = self._parent.pico_setup_channel(
            name=self._name,
            range=range,
            coupling=coupling,
            offset=offset,
            decimation=decimation,
            enabled=True,
        )

    def set_arithmetic(self, arithmetic: Arithmetic, reset: int = 1):
        raise NotImplementedError()


class Picobase(Scope, abc.ABC):
    """
    A base class that helps in implementing picoscope devices.

    .. note::

        Tips for implementors are prefixed with "for implementors" across the code.

    """

    # For implementors: overwrite this variable in subclasses with channel
    # name (str) -> API code
    _PICO_CHANNELS = {}

    # For implementors: overwrite this variable in subclasses with supported vertical
    # ranges.
    _PICO_VERTICAL_RANGES = []

    # Conversion from secbench's slope to API identifier.
    #
    # For implementors: check that the default implementation matches your device.
    _PICO_SLOPES = {}

    # For implementors: check the encoding are the same for your device.
    _PICO_COUPLINGS = {"DC50": 2, "DC": 1, "AC": 0, Coupling.dc: 1, Coupling.ac: 0}

    # For implementors: check the encoding are the same for your device.
    _PICO_BANDWIDTH = {
        "FULL": 0,
        "20MHZ": 1,
        "25MHZ": 2,
    }

    # For implementors: check the encoding are the same for your device.
    _PICO_RATIO_MODE = {
        "NONE": 0,
        "AGGREGATE": 1,
        "AVERAGE": 2,
        "DECIMATE": 4,
        "DISTRIBUTION": 8,
    }

    def __init__(
        self, lib, methods: PicoHandles, device_info_cls=None, serial_number=None
    ):
        self._handle = None
        handle = ctypes.c_int16(0)
        pico_check(methods.psOpenUnit(ctypes.pointer(handle), serial_number))
        assert handle.value > 0, "picoscope device handle is strictly positive"
        self._handle = handle
        self._lib = lib
        self._methods: PicoHandles = methods
        self._device_info_cls = device_info_cls
        self._device_info = {}

        # Check the device works.
        if self._methods.psPingUnit:
            pico_check(self._methods.psPingUnit(self._handle))

        if self._device_info_cls:
            for k in self._device_info_cls:
                assert isinstance(k, int), "device information are integers API code"
                value = self._pico_query_unit_info(k)
                if value is None:
                    logger.debug(
                        f"unable to query device information {k} (probably unsupported)"
                    )
                    continue
                self._device_info[k] = value

        # FIXME: duplicated code with segmented acquisition.
        max_samples = ctypes.c_uint32(0)
        pico_check(
            self._methods.psMemorySegments(self._handle, 1, ctypes.byref(max_samples))
        )
        self._max_samples = int(max_samples.value)
        self._acq_count = 1

        self._timebase: Optional[TimebaseInfo] = None
        self._samples = None
        adc_min, adc_max = self._pico_adc_range()
        self._adc_min = adc_min
        self._adc_max = adc_max

        self._buffers = {}

    def __del__(self):
        # The class may be partially initialized, we need to check if the load
        # was successful.
        self.close_unit()

    def close_unit(self):
        if self._handle is not None:
            logger.debug("closing unit")
            err = self._methods.psCloseUnit(self._handle)
            if err != 0:
                logger.warning("psCloseUnit failed with")
            self._handle = None

    def _pico_query_unit_info(self, key: int) -> Optional[str]:
        value_len = ctypes.c_int16(0)
        # Get the size of the info
        rc = self._methods.psGetUnitInfo(
            self._handle, None, 0, ctypes.byref(value_len), key
        )
        allowed = [
            PicoStatus.PICO_INFO_UNAVAILABLE.value,
            PicoStatus.PICO_INVALID_INFO.value,
        ]
        if rc in allowed:
            return None
        pico_check(rc)
        data = (ctypes.c_ubyte * value_len.value)()
        pico_check(
            self._methods.psGetUnitInfo(
                self._handle, data, len(data), ctypes.byref(value_len), key
            )
        )
        return bytes(data[:-1]).decode()

    def _pico_data_channels(self):
        return list(self._PICO_CHANNELS.keys())

    @abc.abstractmethod
    def _pico_adc_range(self) -> Tuple[int, int]:
        """
        Return the minimum and maximum supported ADC values.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _pico_interval_to_timebase(x: float) -> int:
        """
        Convert duration to API timebase int.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _pico_timebase_to_interval(x: int) -> float:
        """
        Convert API code to duration (in seconds).
        """
        pass

    def _pico_channel_to_api(self, ch_name: str) -> int:
        """
        Convert a channel name to its API identifier.

        The default implementation does a lookup in the :py:attr:`Picobase._PICO_CHANNELS`.
        """
        if ch_name not in self._PICO_CHANNELS:
            raise NoSuchChannelError(ch_name)
        return self._PICO_CHANNELS[ch_name]

    def _pico_supported_vertical_ranges(self) -> Sequence[VerticalRange]:
        """
        Get supported vertical ranges.

        The default implementation does a lookup in the :py:attr:`Picobase._PICO_VERTICAL_RANGES`.
        """
        return self._PICO_VERTICAL_RANGES

    def _pico_minimal_supported_range(self, requested_range) -> Optional[VerticalRange]:
        ranges = self._pico_supported_vertical_ranges()
        if requested_range > ranges[-1].volts:
            return None
        for v in self._pico_supported_vertical_ranges():
            if v.volts >= requested_range:
                return v
        return None

    def _pico_raw_to_volt(self, channel, raw):
        ch = self.channels()[channel]
        return ch.range() / self._adc_max * raw - ch.offset()

    def _pico_volt_to_raw(self, channel, volt):
        ch = self.channels()[channel]
        return int(self._adc_max * (volt + ch.offset()) / ch.range())

    def device_info(self):
        """
        Return all information available on the device.
        """
        return self._device_info

    def is_ready(self) -> bool:
        """
        Check if an acquisition is ready (i.e., data is available for retrival).
        """
        ready = ctypes.c_int16(0)
        pico_check(self._methods.psIsReady(self._handle, ctypes.byref(ready)))
        return ready.value != 0

    def stop_capture(self):
        """
        Stop any capture pending on the instrument.
        """
        pico_check(self._methods.psStop(self._handle))

    def set_sampling_interval(self, interval, duration, segment_index=0):
        # Find the closest matching parameters.
        timebase_id = self._pico_interval_to_timebase(interval)
        timebase_dt = self._pico_timebase_to_interval(timebase_id)
        requested_samples = int(round(duration / timebase_dt))

        interval_ns = ctypes.c_float(0.0)
        max_samples = ctypes.c_uint32(0)

        pico_check(
            self._methods.psGetTimebase2(
                self._handle,
                timebase_id,
                requested_samples,
                ctypes.byref(interval_ns),
                0,
                ctypes.byref(max_samples),
                segment_index,
            )
        )
        max_samples = max_samples.value
        if max_samples < requested_samples:
            logger.info(
                f"cannot acquire {max_samples} samples (too large, maximum value is {requested_samples})"
            )
            # FIXME(@TH): should we raise an error in that case?
        self._samples = requested_samples
        self._timebase = TimebaseInfo(
            timebase_id,
            float(interval_ns.value) * 1e-9,
            "",
            int(max_samples),
            segment_index,
        )

    def _check_horizontal_configuration(self):
        if self._timebase is None:
            raise PartialConfigurationError(
                "no horizontal configuration defined, please call horizontal() first"
            )

    # ===
    # Scope Interface
    # ===

    # Not implemented, should be handled by each device
    # def channels(self):
    #     ...

    def horizontal_interval(self):
        self._check_horizontal_configuration()
        return self._timebase.time_interval

    def horizontal_duration(self):
        self._check_horizontal_configuration()
        return self._samples * self._timebase.time_interval

    def horizontal_samples(self):
        self._check_horizontal_configuration()
        return self._samples

    def _alloc_memory_buffers(self):
        """
        This function is responsible for allocating and assigning memory buffers
        for segmented acquisition (aka. "rapid mode" in picoscope terminology).
        """
        if self._acq_count == 1:
            # Segmented acquisition is off, nothing to allocate.
            return

        samples = self.horizontal_samples()
        channels = [ch.name for ch in self.channels().values() if ch.enabled()]
        logger.debug(
            f"allocating memory buffers for segmented acquisition (channels, acq_count, samples) = ({len(channels)}, {self._acq_count}, {self._samples})"
        )
        for ch in channels:
            buffer = np.zeros((self._acq_count, samples), dtype=np.int16)
            assert buffer.data.c_contiguous
            ptr = buffer.ctypes.data
            self._buffers[ch] = buffer
            ch_number = self._PICO_CHANNELS[ch]
            for segment_index in range(self._acq_count):
                offset = ptr + 2 * segment_index * samples
                self._methods.psSetDataBufferBulk(
                    self._handle,
                    ch_number,
                    offset,
                    samples,
                    segment_index,
                    self._PICO_RATIO_MODE["NONE"],
                )

    def enable_segmented_acquisition(self, count: int):
        max_samples = ctypes.c_uint32(0)
        rc = self._methods.psMemorySegments(
            self._handle, count, ctypes.byref(max_samples)
        )
        pico_check(rc)
        self._max_samples = max_samples
        self._acq_count = count
        self._alloc_memory_buffers()

    def disable_segmented_acquisition(self):
        self._acq_count = 0

    def trigger_count(self) -> int:
        raise NotImplementedError()

    def disable_trigger_out(self):
        raise NotImplementedError()

    def reset(self):
        for ch in self.channels().values():
            ch.disable()
        self._timebase: Optional[TimebaseInfo] = None

    def sync(self):
        # Nothing to do.
        pass

    def _clear(self, pop_errors: bool):
        # Nothing to do.
        pass

    def bit_resolution(self) -> int:
        return 16

    def set_bit_resolution(self, prec: int):
        if prec != 16:
            logger.error(f"unsupported precision requested: {prec}")
            raise InvalidParameter("ps6000 only supports 16 bits precision")

    def pico_setup_channel(
        self,
        *,
        name: str,
        range: float,
        coupling: Coupling,
        offset: float,
        decimation: Decimation,
        enabled: bool = True,
    ) -> ChannelState:
        assert name in self.channels()
        vrange = self._pico_minimal_supported_range(range)
        if vrange is None:
            raise InvalidParameter(f"requested voltage range is too high: {range}")
        offset_api = ctypes.c_float(offset)
        logger.debug(
            f"channel {name}: requested {range} volts, using {vrange.user_str}"
        )
        rc = self._methods.psSetChannel(
            self._handle,
            self._PICO_CHANNELS[name],
            int(enabled),
            self._PICO_COUPLINGS[coupling],
            vrange.api_code,
            offset_api,
            self._PICO_BANDWIDTH["FULL"],
        )
        pico_check(rc)
        return ChannelState(
            enabled=enabled,
            range=vrange.volts,
            offset=offset,
            max_voltage=vrange.volts,
            coupling=coupling,
        )

    def _run_block(self, pre_samples=0, segment_index=0, oversample=1) -> float:
        self._check_horizontal_configuration()

        if self._acq_count > 1:
            rc = self._methods.psSetNoOfCaptures(
                self._handle, ctypes.c_uint32(self._acq_count)
            )
            pico_check(rc)
        assert pre_samples < self._samples
        post_samples = self._samples - pre_samples
        assert post_samples > 0
        hold_time_ms = ctypes.c_int32(0)
        rc = self._methods.psRunBlock(
            self._handle,
            pre_samples,
            post_samples,
            self._timebase.timebase_id,
            oversample,
            ctypes.byref(hold_time_ms),
            segment_index,
            None,
            None,
        )
        pico_check(rc)
        return hold_time_ms.value / 1e3

    def _arm(self, count: int, iterations: int, poll_interval: float) -> float:
        assert count == 1
        t = time.perf_counter()
        self._run_block()
        return time.perf_counter() - t

    def _wait(self, iterations: int, poll_interval: float) -> float:
        t = time.perf_counter()
        for _ in range(iterations):
            if self.is_ready():
                return time.perf_counter() - t
            time.sleep(poll_interval)
        logger.debug("wait timeout reached, stopping capture.")
        self.stop_capture()
        raise InstrumentError(
            "Timeout reached: instrument is still waiting for acquisitions."
        )

    def _wait_auto(self) -> float:
        raise NotImplementedError()

    def _set_data_format(self, bits: int, little_endian: bool):
        if bits != 16 and not little_endian:
            logger.warning(
                f"unsupported data format requested ({bits}, {little_endian}), ignored"
            )

    def _horizontal(self, interval=None, duration=None, samples=None):
        if duration and samples:
            interval = duration / samples
        elif interval and samples:
            duration = interval * samples
        self.set_sampling_interval(interval, duration)
        self._alloc_memory_buffers()

    def _get_data_segmented(self, channels):
        n_samples = ctypes.c_uint32(self._samples)
        overflow = (ctypes.c_int16 * self._acq_count)()
        pico_check(
            self._methods.psGetValuesBulk(
                self._handle,
                ctypes.byref(n_samples),  # noOfSamples
                0,  # fromSegmentIndex
                self._acq_count - 1,  # toSegmentIndex
                1,  # downSampleRatio
                self._PICO_RATIO_MODE["NONE"],  # downSampleRatioMode
                overflow,  # overflow
            )
        )
        return [self._buffers[ch] for ch in channels]

    def _get_data_single(self, channels, segment_index):
        data = [np.zeros(self._samples, dtype=np.int16) for _ in channels]

        for ch, d in zip(channels, data):
            ptr = d.ctypes.data
            ch_api = self._PICO_CHANNELS[ch]
            pico_check(
                self._methods.psSetDataBuffer(
                    self._handle, ch_api, ptr, len(d), self._PICO_RATIO_MODE["NONE"]
                )
            )
        n_samples = ctypes.c_uint32(self._samples)
        overflow = ctypes.c_int16(0)
        pico_check(
            self._methods.psGetValues(
                self._handle,
                0,
                ctypes.byref(n_samples),
                1,
                self._PICO_RATIO_MODE["NONE"],
                segment_index,
                ctypes.byref(overflow),
            )
        )
        n_samples = int(n_samples.value)
        if n_samples < self._samples:
            logger.info(
                f"got less samples than requested ({n_samples}, {self._samples} requested"
            )
        return data

    def _get_data(self, channels, volts: bool, segment_index=0):
        if self._acq_count > 1:
            data = self._get_data_segmented(channels)
        else:
            data = self._get_data_single(channels, segment_index=segment_index)

        for d in data:
            yield d

    def _setup_simple_trigger(
        self,
        channel,
        threshold=0,
        direction=Slope.rising,
        delay=0,
        timeout=0,
        enabled=True,
    ):
        ch_name = self._PICO_CHANNELS[channel]
        slope = self._PICO_SLOPES[direction]

        threshold_raw = self._pico_volt_to_raw(channel, threshold)
        delay = int(delay / self._timebase.time_interval)
        status = self._methods.psSetSimpleTrigger(
            self._handle,
            int(enabled),
            ch_name,
            threshold_raw,
            slope,
            delay,
            timeout,
        )
        pico_check(status)

    def _enable_trigger_out(self, slope, length, delay):
        raise InstrumentUnsupportedFeature("trigger out")

    def _set_trigger(self, channel, slope, level, delay):
        self._setup_simple_trigger(
            channel, threshold=level, direction=slope, delay=delay, timeout=0
        )