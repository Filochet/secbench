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

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..enums import Arithmetic, Coupling, Decimation
from ..exceptions import InstrumentError, InvalidParameter
from ..helpers import DATACLASS_KW_ONLY_AND_SLOTS
from ..instrument import Scope, ScopeAnalogChannel


def rand_int8_generator(channels, n_traces: int, n_samples: int):
    return [
        np.random.randint(-128, 128, size=(n_traces, n_samples), dtype=np.int8)
        for _ in channels
    ]


@dataclass(**DATACLASS_KW_ONLY_AND_SLOTS)
class ChannelState:
    enabled: bool
    range: float
    offset: float
    coupling: Coupling = Coupling.dc
    decimation: Decimation = Decimation.sample


class SimulatedScopeChannel(ScopeAnalogChannel):
    def __init__(self, parent: SimulatedScope, name: str):
        super().__init__()
        self._parent = parent
        self._name = name
        self._state = ChannelState(enabled=False, range=1, offset=0)

    @property
    def parent(self) -> Scope:
        return self._parent

    @property
    def name(self) -> str:
        return self._name

    def disable(self) -> None:
        self._state.enabled = False

    def enabled(self) -> bool:
        return self._state.enabled

    def coupling(self) -> Coupling:
        return self._state.coupling

    def range(self) -> float:
        return self._state.range

    def offset(self) -> float:
        return self._state.offset

    def decimation(self) -> Decimation:
        return self._state.decimation

    def setup(
        self,
        range: float | None = None,
        coupling: Coupling | None = None,
        offset: float = 0,
        decimation: Decimation = Decimation.sample,
    ) -> None:
        self._state.enabled = True
        if range:
            self._state.range = range
        if coupling:
            self._state.coupling = coupling
        if offset:
            self._state.offset = offset
        if decimation:
            self._state.decimation = decimation

    def set_arithmetic(self, arithmetic: Arithmetic, reset: int = 1):
        pass
        # raise NotImplementedError()


class SimulatedScope(Scope):
    def __init__(self, channel_names: Sequence[str], trace_generator=None):
        super().__init__()

        if trace_generator is None:
            self._trace_generator = rand_int8_generator
        self._channels: dict[str, ScopeAnalogChannel] = {
            name: SimulatedScopeChannel(self, name) for name in channel_names
        }
        self._horizontal_interval = 1e-9
        self._horizontal_samples = 1000
        self._horizontal_duration = self._horizontal_interval * self._horizontal_samples
        self._segmented_acquisition = 0
        self._bit_resolution = 8

        self._trigger_count = 0
        self._arm_count = 0
        self._wait_called = False

    def force_trigger(self):
        """
        Generate a trigger on the simulated model.
        """
        self._trigger_count += 1

    # ===
    # Implementation of the Scope interface.
    # ===

    @property
    def description(self) -> str:
        return "simulated scope"

    def channels(self) -> Mapping[str, ScopeAnalogChannel]:
        return self._channels

    def horizontal_interval(self) -> float:
        return self._horizontal_interval

    def horizontal_duration(self) -> float:
        return self._horizontal_duration

    def horizontal_samples(self) -> int:
        return self._horizontal_samples

    def enable_segmented_acquisition(self, count: int):
        self._segmented_acquisition = count

    def disable_segmented_acquisition(self):
        self._segmented_acquisition = 0

    def bit_resolution(self) -> int:
        return self._bit_resolution

    def set_bit_resolution(self, prec: int):
        if prec != 8:
            raise InvalidParameter("unsupported feature requested")

    def trigger_count(self) -> int:
        return self._trigger_count

    def reset(self):
        pass

    def disable_trigger_out(self):
        pass

    def sync(self):
        pass

    def _arm(self, count: int, _iterations: int, _poll_interval: float) -> float:
        self._trigger_count = 0
        self._arm_count = count
        self._wait_called = False

    def _wait(self, iterations: int, poll_interval: float) -> float:
        if self._segmented_acquisition == 0:
            if self._arm_count > self._trigger_count:
                raise InstrumentError("not enough trigger received")
        else:
            if self._segmented_acquisition > self._trigger_count:
                raise InstrumentError("not enough trigger received")

        self._wait_called = True

    def _wait_auto(self) -> float:
        return self._wait(1, 0)

    def _clear(self, pop_errors: bool):
        pass

    def _set_data_format(self, bits: int, little_endian: bool):
        pass

    def _horizontal(self, interval=None, duration=None, samples=None):
        if interval is None:
            interval = duration / samples

        if duration is None:
            duration = interval * samples

        if samples is None:
            samples = int(duration / interval)

        if not np.isclose(interval * samples, duration):
            raise InstrumentError("inconsistent parameters requested")

        self._horizontal_interval = interval
        self._horizontal_duration = duration
        self._horizontal_samples = samples

    def _set_trigger(self, channel, slope, level, delay):
        pass

    def _enable_trigger_out(self, slope, length, delay):
        pass

    def _get_data(self, channels, volts: bool):
        if volts:
            raise NotImplementedError(
                "this feature is not implemented on the simulated model"
            )
        if not self._wait_called:
            raise InstrumentError(
                "calling Scope.get_data without calling Scope.wait first."
            )

        for ch in channels:
            if not self._channels[ch].enabled:
                raise InstrumentError("channel requested is not enabled")
        n_traces = self._segmented_acquisition if self._segmented_acquisition else 1
        traces = self._trace_generator(channels, n_traces, self._horizontal_samples)

        if self._segmented_acquisition is None:
            # If segmented acquisition is disabled, 1-D arrays must be returned. This is compliant
            # with current scope implementations.
            traces = [t[0] for t in traces]
        return traces