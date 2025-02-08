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
from typing import Mapping, Optional, Sequence, Tuple

from ..enums import BurstMode, Function, OutputLoad, Polarity, Slope, TriggerSource


class AfgChannel(abc.ABC):
    # ===
    # Methods to implement for an AFG channel
    # ===
    @property
    @abc.abstractmethod
    def parent(self):
        """
        Return the parent AFG.
        """
        pass

    @abc.abstractmethod
    def _set_output_state(
        self, enabled, load: Optional[OutputLoad], polarity: Optional[Polarity]
    ):
        pass

    @abc.abstractmethod
    def output_state(self) -> Tuple[bool, OutputLoad, Polarity]:
        """
        Return the output state for the current channel.
        """
        pass

    @abc.abstractmethod
    def _set_trigger_state(
        self, source: Optional[TriggerSource], slope: Optional[Slope]
    ):
        """
        Configure the trigger.
        """
        pass

    @abc.abstractmethod
    def trigger_state(self) -> Tuple[TriggerSource, Slope]:
        """
        Return current trigger configuration.
        """
        pass

    @abc.abstractmethod
    def force_trigger(self):
        """
        Force a manual trigger to a specific channel.
        """
        pass

    @abc.abstractmethod
    def set_burst_mode(self, cycles: int, mode: BurstMode):
        """
        Enable burst mode (generation of a finite number of shapes).

        :Example:

        >>> afg_ch1.set_burst_mode(1, BurstMode.triggered)

        """
        pass

    @abc.abstractmethod
    def burst_count(self) -> int:
        """
        Gets the number of burst for a specific channel.

        :Example:

        >>> ncycles = afg_ch1.burst_count()
        1
        """
        pass

    @abc.abstractmethod
    def set_voltage(self, amplitude: float, offset: float | None = None):
        """
        Set voltage parameters.

        :Example:

        >>> afg_ch1.set_voltage(1.5, 0)
        """
        pass

    @abc.abstractmethod
    def voltage(self) -> Tuple[float, float]:
        """
        Return the voltage parameters of the AFG.

        :Example:

        >>> volt, _ = afg_ch1.voltage()
        >>> volt, offset = afg_ch1.voltage()
        """
        pass

    @abc.abstractmethod
    def set_frequency(self, frequency: float):
        """
        Set the signal frequency (in Hz).

        :Example:

        >>> afg_ch1.set_frequency(1E+5)
        """
        pass

    @abc.abstractmethod
    def frequency(self) -> float:
        """
        Return the frequency for the current function (in Hz).

        :Example:

        >>> frequency = afg_ch1.frequency()
        1E+6
        """
        pass

    @abc.abstractmethod
    def set_function(self, function: Function):
        """
        Set the function of the arbitrary function generator.

        :Example:

        >>> afg_ch1.set_function(Function.sinus)
        >>> afg_ch1.set_function(Function.pulse)
        """
        pass

    @abc.abstractmethod
    def function(self) -> Function:
        """
        Return the active function of the arbitrary function generator.
        """
        pass

    @abc.abstractmethod
    def set_duty_cycle(self, w: float):
        """
        Set the duty cycle  in percentage (i.e., from 0 to 100) for the square wave.
        """
        pass

    @abc.abstractmethod
    def duty_cycle(self) -> float:
        """
        Returns the duty cycle in percentage (i.e., from 0 to 100) of the square wave.
        """
        pass

    @abc.abstractmethod
    def set_ratio(self, w: float):
        """
        Sets the ratio for functions that support it (pulse, ramp, etc.).
        """
        pass

    @abc.abstractmethod
    def ratio(self) -> float:
        """
        Return the ratio of the ramp function in percentage.
        """
        pass

    @abc.abstractmethod
    def pulse_width(self) -> float:
        """
        Returns the pulse width in nanoseconds for the pulse function.
        """
        pass

    @abc.abstractmethod
    def set_pulse_width(self, width_ns: float):
        """
        Set the pulse width in nanoseconds for the pulse function.
        """
        pass

    @abc.abstractmethod
    def pulse_edge_time(self) -> float:
        """
        Return the rising edge time in nanoseconds for pulse function.
        """
        pass

    @abc.abstractmethod
    def set_pulse_edge_time(self, edge_time_ns: float):
        """
        Set the rising edge time in nanoseconds for pulse function.
        """
        pass

    @abc.abstractmethod
    def pulse_delay(self) -> float:
        """
        Retrieve the pulse delay in nanoseconds.
        """
        pass

    @abc.abstractmethod
    def set_pulse_delay(self, delay_ns: float):
        """
        Set the pulse delay in nanoseconds.
        """
        pass

    @abc.abstractmethod
    def set_trigger_delay(self, delay_ns: float):
        """
        Set the delay from the trigger signal.

        This delay should not be used for precise offset. Usually :py:meth:`Afg.set_pulse_delay`
        provides better precision.
        """
        pass

    def set_combined_delay(self, delay_ns: float):
        """
        Configure an arbitrary delay, keeping a good precision.

        This method combines :py:meth:`Afg.set_trigger_delay` and :py:meth:`Afg.set_pulse_delay`.
        This is the recommended method to use if you need to set long delays.
        """
        freq = self.frequency()
        freq_min = 1 / 2e-6
        if freq > freq_min:
            raise ValueError(
                f"frequency is too high to support combined delay, set it lower than {freq_min}"
            )
        delay_us = delay_ns // 1000
        delay_offset = delay_ns % 1000
        self.set_pulse_delay(delay_offset)
        self.set_trigger_delay(delay_us * 1000)

    # ===
    # Methods provided for all AFGs
    # ===

    def _set_freq_voltage(self, frequency: float, voltage: float, offset: float | None):
        self.set_frequency(frequency)
        self.set_voltage(voltage, offset)

    def generate_sinus(
        self, frequency: float, voltage: float, offset: float | None = None
    ):
        self.set_function(Function.sinus)
        self._set_freq_voltage(frequency, voltage, offset)

    def generate_square(
        self,
        frequency: float,
        voltage: float,
        offset: float | None = None,
        duty_cycle: float = 50,
    ):
        """
        Configure a square waveform in a single call.
        """
        self.set_function(Function.square)
        self._set_freq_voltage(frequency, voltage, offset)
        self.set_duty_cycle(duty_cycle)

    def generate_ramp(
        self, frequency: float, voltage: float, offset: float | None = None, ratio=100
    ):
        """
        Configure a ramp waveform in a single call.
        """
        self.set_function(Function.ramp)
        self._set_freq_voltage(frequency, voltage, offset)
        self.set_ratio(ratio)

    def generate_pulse(
        self,
        frequency: float,
        voltage: float,
        offset: float | None = None,
        edge_time_ns: float = 5,
        width_ns: float = 20,
    ):
        """
        Configure a pulse waveform in a single call.
        """
        self.set_function(Function.pulse)
        self._set_freq_voltage(frequency, voltage, offset)
        self.set_pulse_width(width_ns)
        self.set_pulse_edge_time(edge_time_ns)

    def generate_noise(self, voltage: float, offset: float | None = None):
        """
        Configure a noise waveform in a single call.
        """
        self.set_function(Function.noise)
        self.set_voltage(voltage, offset)

    def set_trigger_state(self, source=None, slope=None):
        self._set_trigger_state(source, slope)

    def set_output_state(self, enabled, load=None, polarity=None):
        self._set_output_state(enabled, load, polarity)


class Afg(abc.ABC):
    """
    Abstract interface for Arbitrary function generators.
    """

    # ===
    # Methods to implement.
    # ===

    @property
    @abc.abstractmethod
    def description(self):
        """
        A unique string identifying the instrument.
        """
        pass

    @abc.abstractmethod
    def _clear(self, pop_errors: bool):
        pass

    @abc.abstractmethod
    def set_locked(self, locked: bool):
        """
        Lock or unlock the instrument.

        :Example:

        >>> afg.set_locked(True) # Device is locked
        >>> afg.set_locked(False) # Device is unlocked
        """
        pass

    @abc.abstractmethod
    def channels(self) -> Mapping[str, AfgChannel]:
        """
        Return a list of channels available on this instrument.
        """
        pass

    def default_channel(self):
        """
        Return a conventional default channel for the AFG.

        The default implementation returns the first channel.
        """
        return self.channel_list()[0]

    # ===
    # Methods provided.
    # ===

    def reset(self):
        """
        Reset the instrument.

        The default implementation invokes the :py:meth:`Afg.clear` method.
        """
        self.clear()

    def clear(self, pop_errors=True):
        """
        Clear pending measurements, status flags and optionally pop errors.
        """
        return self._clear(pop_errors)

    def __getitem__(self, key: str):
        """
        An alias for :py:meth:`get_channel`.
        """
        return self.channels()[key]

    def channel_names(self) -> Sequence[str]:
        return list(self.channels().keys())

    def channel_list(self) -> Sequence[AfgChannel]:
        return list(self.channels().values())