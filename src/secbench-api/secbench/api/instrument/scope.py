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
import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..enums import Arithmetic, Coupling, Decimation, Slope
from ..exceptions import NoSuchChannelError
from ..helpers import DATACLASS_KW_ONLY_AND_SLOTS

logger = logging.getLogger(__name__)


class ScopeAnalogChannel(abc.ABC):
    """
    Represents a scope analog channel.

    Use the dict syntax on :class:`Scope`
    instances to retrieve a channel instance.
    """

    # ===
    # Methods that must be implemented.
    # ===
    @property
    @abc.abstractmethod
    def parent(self) -> Scope:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def enabled(self) -> bool:
        pass

    @abc.abstractmethod
    def coupling(self) -> Coupling:
        pass

    @abc.abstractmethod
    def range(self) -> float:
        # TODO: rename vertical_range
        pass

    @abc.abstractmethod
    def offset(self) -> float:
        pass

    @abc.abstractmethod
    def decimation(self) -> Decimation:
        pass

    def disable(self) -> None:
        """
        Disable the channel.
        """
        return

    @abc.abstractmethod
    def setup(
        self,
        range: float | None = None,
        coupling: Coupling | None = None,
        offset: float | None = None,
        decimation: Decimation = Decimation.sample,
    ) -> None:
        """
        Enable the channel and configure it with the given parameters.

        * For analog channels, :attr:`range` and :attr:`coupling` are required.

        :param range: the vertical range, in volt
        :param coupling: the coupling
        :param offset: the offset in volt (default: 0)
        :param decimation: the decimation method (default: sample)
        """
        pass

    @abc.abstractmethod
    def set_arithmetic(self, arithmetic: Arithmetic, reset: int = 1):
        """
        Set the channel :class:`arithmetic <Arithmetic>` function.
        The function is reset after :attr:`reset` triggers.

        :param arithmetic: the arithmetic function to use
        :param reset: the number of triggers after which the function is reset
        """
        pass

    # ===
    # Methods provided.
    # ===

    def set_trigger(self, level: float, slope: Slope = Slope.rising, delay: float = 0):
        return self.parent.set_trigger(self.name, level, slope=slope, delay=delay)

    def __repr__(self):
        return "<{} {}: {}, {:.5f}V {} offset {:+.5f}V {}>".format(
            self.__class__.__name__,
            self.name,
            "enabled" if self.enabled else "disabled",
            self.range(),
            self.coupling().name.upper(),
            self.offset(),
            self.decimation(),
        )


class Scope(abc.ABC):
    """
    Base abstract class for scope hardware.
    """

    # ===
    # Methods that must be implemented.
    # ===
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        The self-description of this instrument. This is typically a string
        representation of the device serial number.

        >>> scope.description
        "1329.7002K14-100654-Hn"

        """
        pass

    @abc.abstractmethod
    def channels(self) -> Mapping[str, ScopeAnalogChannel]:
        """
        Return a mapping of analog channels available.
        """
        pass

    @abc.abstractmethod
    def horizontal_interval(self) -> float:
        """
        Duration in seconds between two points.

        >>> scope.horizontal_interval()
        1e-10

        """
        pass

    @abc.abstractmethod
    def horizontal_duration(self) -> float:
        """
        Duration in seconds of the whole acquisition.

        >>> scope.horizontal_duration()
        1e-07

        """
        pass

    @abc.abstractmethod
    def horizontal_samples(self) -> int:
        """
        Number of samples (points) for the whole acquisition.

        >>> scope.horizontal_samples()
        1000

        """
        pass

    @abc.abstractmethod
    def enable_segmented_acquisition(self, count: int):
        pass

    @abc.abstractmethod
    def disable_segmented_acquisition(self):
        pass

    @abc.abstractmethod
    def bit_resolution(self) -> int:
        """
        Get the number of bits used to represent a single sample.

        Usually this value is 8 or 16 scopes that have high-precision ADCs.
        """
        pass

    @abc.abstractmethod
    def set_bit_resolution(self, prec: int):
        """
        Set the number of bits to represent a single sample.

        The values supported are usually 8, 12 or 16, depending on the scope model.
        """
        pass

    @abc.abstractmethod
    def trigger_count(self) -> int:
        """
        When using an acquisition count greater than 1, return the number of
        triggers that happened since last arming.
        """
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset the instrument to factory defaults. This usually resets the
        timebase and trigger settings to default values and disables all
        channels. This can be a no-op on some hardware.

        >>> scope.reset()
        """
        pass

    @abc.abstractmethod
    def disable_trigger_out(self):
        """
        Disable the trigger out signal.
        """
        pass

    @abc.abstractmethod
    def sync(self):
        """
        Synchronize the class instance with the scope parameters.
        """
        pass

    @abc.abstractmethod
    def _arm(self, count: int, iterations: int, poll_interval: float) -> float:
        pass

    @abc.abstractmethod
    def _wait(self, iterations: int, poll_interval: float) -> float:
        pass

    @abc.abstractmethod
    def _wait_auto(self) -> float:
        pass

    @abc.abstractmethod
    def _clear(self, pop_errors: bool):
        pass

    @abc.abstractmethod
    def _set_data_format(self, bits: int, little_endian: bool):
        pass

    @abc.abstractmethod
    def _horizontal(self, interval=None, duration=None, samples=None):
        pass

    @abc.abstractmethod
    def _set_trigger(self, channel: str, slope: Slope, level: float, delay: float):
        pass

    @abc.abstractmethod
    def _enable_trigger_out(self, slope, length, delay):
        pass

    @abc.abstractmethod
    def _get_data(self, channels: Sequence[str], volts: bool):
        pass

    # ===
    # Methods provided
    # ===
    def __del__(self):
        # Clear scope when object is deleted
        self.clear()

    def channel_names(self) -> Sequence[str]:
        return list(self.channels().keys())

    def channel_list(self) -> Sequence[ScopeAnalogChannel]:
        return list(self.channels().values())

    def set_trigger(
        self, channel: str, level: float, slope: Slope = Slope.rising, delay: float = 0
    ):
        """
        Configure the trigger condition.

        >>> scope.set_trigger('1', Slope.rising, 0.5)

        :param channel: the name of the channel used to trigger
        :param slope: the :class:`Slope` to trigger on
        :param level: the trigger level in volts
        :param delay: the delay in seconds between trigger and start of acquisition.
        """
        self._check_channels(channel)
        self._set_trigger(channel, slope, level, delay)

    def enable_trigger_out(self, slope: Slope, length: float, delay: float = 0) -> None:
        """
        Enable the trigger out signal.

        :param slope: :attr:`Slope.rising` or :attr:`Slope.falling`
        :param length: pulse length in seconds
        :param delay: pulse delay in seconds
        """
        if slope not in (Slope.rising, Slope.falling):
            raise TypeError("invalid Slope")
        self._enable_trigger_out(slope, length, delay)

    def set_horizontal(
        self,
        *,
        interval: float | None = None,
        duration: float | None = None,
        samples: int = None,
    ) -> None:
        """
        Configure the horizontal (time) parameters. Call this method with
        any combination of two of the three available parameters. The third,
        unspecified one, will be computed according to the other two.

        >>> # 10k samples during 2 ms
        >>> scope.horizontal(samples=10e3, duration=2e-3)
        >>> # 1 µs resolution during 0.5 ms
        >>> scope.horizontal(interval=1e-6, duration=.5e-3)
        >>> # 5M samples with 1 µs resolution
        >>> scope.horizontal(samples=5e6, interval=1e-6)

        :param interval: duration in seconds between two points
        :param duration: duration in seconds of the whole acquisition
        :param samples: number of samples (points) for the whole acquisition

        :raises ValueError: if zero, one or three parameters are given instead of two.
        """
        if interval and duration and samples:
            raise ValueError(
                "horizontal() needs only two parameters among `interval`, "
                "`duration` and `samples`"
            )
        if interval and duration:
            self._horizontal(interval=interval, duration=duration)
        elif duration and samples:
            self._horizontal(duration=duration, samples=samples)
        elif interval and samples:
            self._horizontal(interval=interval, samples=samples)
        else:
            raise ValueError(
                "horizontal() needs at least two parameters among `interval`, "
                "`duration` and `samples`"
            )

    def segmented_acquisition(self, count: int | None) -> None:
        """
        Enable (or disable) segmented acquisition (aka. Ultra segmentation)

        :param count: If none, disable segmented acquisition, otherwise
            activate segmented acquisition of ``count`` frames.

        """
        if count is None or count <= 1:
            self.disable_segmented_acquisition()
        else:
            self.enable_segmented_acquisition(count)

    def arm(self, count=1, iterations=1000, poll_interval=1e-3) -> float:
        """
        Arm the instrument for triggering. This command returns immediately and
        does not wait for an actual trigger to happen. To this end, you need to
        call :func:`wait` just after :func:`arm`.

        :param count: number of triggers constituting a single acquisition.
                      Some hardware only supports ``count=1``.
        :param iterations: number of iterations for polling trigger state.
        :param poll_interval:
            delay between state polling iteration if None, will poll without
            interruptions.
        :return: time elapsed.

        >>> scope.arm()  # arm to receive a single trigger
        >>> scope.arm(count=20)  # arm to receive 20 triggers

        """
        return self._arm(count, iterations, poll_interval)

    def wait(self, iterations=1000, poll_interval=1e-3) -> float:
        """
        Wait for the instrument to complete the latest command sent.

        It is particularly useful when called after :func:`arm` to guarantee
        that data was actually acquired.

        >>> scope.wait()  # waits 1 second
        >>> scope.wait(iterations=10, poll_interval=1) # waits 10 second

        :param iterations: Number of polling iteration before assessing timeout.
        :param poll_interval: Duration of a polling loop.
        :return: time elapsed.
        """
        return self._wait(iterations, poll_interval)

    def wait_auto(self) -> float:
        """
        Wait for the instrument to complete the latest command sent.
        It is particularly useful when called after :func:`arm` to guarantee
        that data was actually acquired.

        This function automatically sets up the `iteration` and `polling` parameters
        of the `wait` function.

        :return: time elapsed.
        """
        return self._wait_auto()

    def set_data_format(self, bits=8, little_endian=True):
        """
        Configure the data format returned by :py:meth:`Scope.get_data`
        method.

        :param bits: Number of bits
        :param little_endian: When more than one byte is returned per sample
                              (bits > 8) specify the order of bytes
                              (little-endian = least significant bytes first)

        """
        return self._set_data_format(bits, little_endian)

    def get_data(self, *channels: str | ScopeAnalogChannel, volts: bool = False):
        """
        Retrieve the waveform data for the specified channel(s).

        * If ``volts`` is False, no attempt to convert raw samples to volts is
          made. The returned samples are raw ADC values in the device internal
          format. Usually this means 8 or 16 bit integer values.
        * If ``volts`` is True and the device supports it, returned samples are
          in volts. Usually this means floating point values.

        You can retrieve the actual data type using the numpy :attr:`dtype`
        property, eg. ``print(my_array.dtype)``.

        This method returns a list of waveform data, each being a single
        :class:`numpy.array`.

        >>> c1, = scope.get_data('1')  # note the comma (,)
        >>> c1, c4 = scope.get_data('1', '4')
        >>> c1_volts, = scope.get_data('1', volts=True)

        :param channels: the channel(s) to retrieve data from
        :param volts: return volts instead of raw samples (if supported)
        """
        channels = self._channel_names(*channels)
        return list(self._get_data(channels, volts))

    def config(self, channels=None, only_enabled=False):
        """
        Return a dictionary of scope attributes

        EXAMPLES

        >>> scope.config(channels=['A', 'B'], only_enabled=True)
        { 'scope_name': Foo,
          'scope_horizontal_samples': 10000,
          ...
        }
        """

        d = {
            "scope_name": self.__class__.__name__,
            "scope_description": self.description,
        }
        for attr in [
            "horizontal_samples",
            "horizontal_interval",
            "horizontal_duration",
        ]:
            try:
                v = getattr(self, attr)
                if v is not None:
                    d[f"scope_{attr}"] = v()
            except AttributeError:
                pass
            except NotImplementedError:
                pass
        channels = channels or []
        for c in channels:
            if isinstance(c, str):
                c = self.__getitem__(c)
            c_name = c.name
            if only_enabled and not c.enabled:
                continue
            channel_attrs = {}
            for attr in ["coupling", "offset", "range", "decimation", "enabled"]:
                try:
                    v = getattr(c, attr)
                    if v is not None:
                        channel_attrs[attr] = v()
                except AttributeError:
                    pass
                except NotImplementedError:
                    pass
            d[f"scope_channel_{c_name}"] = channel_attrs
        return d

    def clear(self, pop_errors=False):
        """
        Clear pending measurements, status flags and optionally pop errors.

        Can be used to bring back the scope in a usable mode if something goes
        wrong.
        """
        self._clear(pop_errors)

    def calibrate(
        self, channel: str, clip_detection_callback, method: Calibration | None = None
    ) -> float:
        """
        Perform automatic calibration of the scope's vertical range.

        :param channel: The scope channel to calibrate
        :param clip_detection_callback: A function responsible for detecting
            whether the scope is clipping or not. This callback takes as input
            the scope and a channel and returns a boolean.
        :param method: The calibration algorithm to use. See instances
            of the :py:class:`Calibrate` abstract method.

        :return: The voltage scale found.

        :raises: :py:class:`ValueError` if the maximum number of iteration
            is reached.

        .. versionadded:: 5.1.0

        :Example:

        .. code-block:: python

            from secbench.scope.util import is_clipping

            def is_clipping_callback(scope, channel):
                scope.arm()
                # send DUT input
                scope.wait()
                d, = scope.get_data(channel)
                return is_clipping(d)

            # Latter in your code:
            scope.calibrate('1', is_clipping_callback, method=StepSearchCalibration())

        """
        if method is None:
            method = StepSearchCalibration()
        assert isinstance(method, Calibration)
        return method.run(self, self[channel], clip_detection_callback)

    def __len__(self):
        return len(self.channels())

    def __iter__(self):
        return iter(self.channels())

    def __getitem__(self, channel: str) -> ScopeAnalogChannel:
        return self.channels()[channel]

    def __repr__(self):
        return "<{} `{}`>".format(self.__class__.__name__, self.description)

    # ===
    # Private API. For internal use only.
    # ===

    def _channel_names(self, *channels: str | ScopeAnalogChannel) -> Sequence[str]:
        ch_names = []
        for ch in channels:
            if isinstance(ch, ScopeAnalogChannel):
                ch_names.append(ch.name)
            else:
                ch_names.append(ch)
        self._check_channels(*ch_names)
        return ch_names

    def _check_channels(self, *channels: str):
        available_channels = self.channels()
        for channel in channels:
            if channel not in available_channels:
                raise NoSuchChannelError(channel)


@dataclass(**DATACLASS_KW_ONLY_AND_SLOTS)
class CalibrationData:
    """
    Data returned by calibration callbacks.
    """

    range: float
    offset: float
    is_clipping: bool


class Calibration:
    """
    Abstract interface for scope calibration methods.
    """

    @abc.abstractmethod
    def run(
        self, scope: Scope, channel: ScopeAnalogChannel, clip_detection_callback
    ) -> float:
        """
        Launch the calibration on a given channel.

        This method has to be reimplemented by the different calibration
        algorithms.
        """
        pass


class BinarySearchCalibration(Calibration):
    """
    Perform automatic calibration of the scope vertical range using a
    binary search.

    The initial search window is the voltage range ``[volts_min; volts_max]``.
    The search range is shrunk at each iteration by 2. The algorithm stops if
    the search window size is below ``volts_prec``.

    If the algorithm does not find a non-clipping setting after
    ``max_iterations`` a :py:class:`ValueError` is raised.
    """

    def __init__(
        self, volts_min=0.001, volts_max=1, volts_prec=0.01, max_iterations=15
    ):
        self.volts_min = volts_min
        self.volts_max = volts_max
        self.volts_prec = volts_prec
        self.max_iterations = max_iterations

    def run(
        self, scope: Scope, channel: ScopeAnalogChannel, clip_detection_callback
    ) -> float:
        low = self.volts_min
        high = self.volts_max

        for i in range(self.max_iterations):
            mid = (high + low) / 2
            logger.debug(
                f"calibration step {i} (max={self.max_iterations}), range={mid}"
            )
            channel.setup(range=mid, coupling=channel.coupling())
            if clip_detection_callback(scope, channel).is_clipping:
                low = mid
            else:
                high = mid

            w = high - low
            if w <= self.volts_prec:
                return high

        raise ValueError(f"maximum number of iteration reached {self.max_iterations}")


class LinearSearchCalibration(Calibration):
    """
    Perform automatic calibration of the scope's vertical range using a linear
    search.

    The algorithm starts from ``volts_min`` and increment the voltage by
    ``volt_prec`` until the signal stops clipping. The algorithm stops
    if ``volts_max`` is reached.

    If the algorithm does not find a non-clipping setting after
    ``max_iterations`` a :py:class:`ValueError` is raised.
    """

    def __init__(
        self, volts_min=0.001, volts_max=1, volts_prec=0.05, max_iterations=15
    ):
        self.volts_min = volts_min
        self.volts_max = volts_max
        self.volts_prec = volts_prec
        self.max_iterations = max_iterations

    def run(self, scope: Scope, channel: ScopeAnalogChannel, clip_detection_callback):
        volts = self.volts_min

        for step in range(self.max_iterations):
            logger.debug(
                f"calibration step {step} (max={self.max_iterations}), range={volts}"
            )
            if volts >= self.volts_max:
                break
            # Try a new voltage configuration
            channel.setup(range=volts, coupling=channel.coupling())
            if clip_detection_callback(scope, channel).is_clipping:
                volts += self.volts_prec
            else:
                return volts

        raise ValueError(f"maximum number of iteration ({self.max_iterations}) reached")


class StepSearchCalibration(Calibration):
    """
    Perform automatic calibration of the scope's vertical range using a stepwise
    search.

    The algorithm starts from ``volts_max`` and reduces the voltage range to reach the selected profile.
    It also sets updates the voltage offset.
    """

    def __init__(self, volts_min=0.001, volts_max=20, profile=None):
        if profile is None:
            profile = [0.20, 0.50, 0.90]
        self.volts_min = volts_min
        self.volts_max = volts_max
        self.profile = profile

    def run(self, scope: Scope, channel: ScopeAnalogChannel, callback):
        """
        Run the calibration of scope's voltage range and offset following
        the provided profile.

        :param scope: Scope that can be used to perform acquisitions
        :param channel: Scope channel on which calibration should be done.
        :param callback: A function which returns a tuple ``(adc_dynamics, adc_offset)``.
            ``adc_dynamics`` represents the dynamics of the signal when sampled by ADCs (e.g., ``abs(max - min)``).
            ``adc_offset`` represents the offset of the signal when sampled by ADCs (e.g., ``(min + max) / 2``).
        """

        logger.debug(f"Setting channel to range {self.volts_max}")
        v_range = self.volts_max
        channel.setup(range=v_range, coupling=channel.coupling(), offset=0)
        v_offset = 0
        sample_range = 2 ** scope.bit_resolution()

        for delta in self.profile:
            logger.debug(f"Profile {delta}, calling callback...")
            calibration_data = callback(scope, channel)
            adc_range, adc_offset = calibration_data.range, calibration_data.offset
            logger.debug(
                f"Recieved ADC dynamics {adc_range} and ADC offset {adc_offset}"
            )
            # Compute the relative shift of voltage offset
            v_offset = adc_offset * (v_range / sample_range) + v_offset
            # Compute the proportion of the screen already used
            p_range = adc_range / sample_range
            # Compute the ration between target screen proportion and used screen proportion
            ratio = p_range / delta
            # Compute voltage range accordingly
            v_range = ratio * v_range
            logger.debug(f"Setting channel to range {v_range} and offset {v_offset}")
            channel.setup(range=v_range, offset=v_offset, coupling=channel.coupling())

        return v_range, v_offset