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
import dataclasses
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Protocol, Sequence, Type, TypeVar

from ..enums import Slope, TriggerSource
from ..helpers import DATACLASS_KW_ONLY


@dataclass(**DATACLASS_KW_ONLY)
class DelayParams:
    """
    Common parameters for fault injection.
    """

    delay_ns: float


@dataclass(**DATACLASS_KW_ONLY)
class GlitchParams(DelayParams):
    """
    Parameter of a voltage glitch fault injection.

    * :py:attr:`delay_ns`: delay of the pulse from the trigger signal.
    * :py:attr:`width_ns`: width of the pulse.
    """

    width_ns: float


@dataclass(**DATACLASS_KW_ONLY)
class EMPulseParams(DelayParams):
    """
    Parameters for electro-magnetic fault injection.

    * :py:attr:`delay_ns`: delay of the pulse from the trigger signal.
    * :py:attr:`width_ns`: width of the pulse.
    * :py:attr:`rise_time_ns`: delay to reach the maximum voltage.
    * :py:attr:`amplitude`: amplitude of the EM pulse.
    """

    width_ns: float
    rise_time_ns: float
    amplitude: float


@dataclass(**DATACLASS_KW_ONLY)
class LaserParams(DelayParams):
    """
    Parameters for laser fault injection.

    * :py:attr:`delay_ns`: delay of the pulse from the trigger signal.
    * :py:attr:`width_ns`: width of the pulse.
    * :py:attr:`current`: current of the laser source in Ampere.
    """

    width_ns: float
    current: float


T = TypeVar("T")


class StateProxy:
    def __init__(self, parent: PulserChannel):
        self._parent = parent

    def __getattr__(self, item):
        return self._parent._query_params(item)[0]


class PulserChannel(Generic[T], abc.ABC):
    def __init__(self):
        self.__supported_params = {
            f.name for f in dataclasses.fields(self.param_type())
        }
        self._state = StateProxy(self)

    # ===
    # Methods to be implemented.
    # ===
    @classmethod
    @abc.abstractmethod
    def param_type(cls) -> Type[T]:
        """
        Type of parameters for this channel.

        Must be a dataclass (e.g., :py:class:`GlitchParams`).
        """
        pass

    @property
    @abc.abstractmethod
    def parent(self) -> Pulser:
        """
        Return the parent pulser of this channel.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the channel.
        """
        pass

    @abc.abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the channel.
        """
        pass

    @abc.abstractmethod
    def enabled(self) -> bool:
        """
        Test if a channel is enabled or not.
        """
        pass

    @abc.abstractmethod
    def setup(self, **kwargs):
        """
        Modify one or more parameters of the channel.

        Specific parameters are changed through ``kwargs``, the arguments here must be
        valid fields of :py:meth:`PulserChannel.param_type`.
        """
        pass

    @abc.abstractmethod
    def _query_param(self, name: str) -> Any:
        """
        Query a specific parameter.
        """
        pass

    # ===
    # Methods provided.
    # ===

    @property
    def state(self) -> StateProxy:
        return self._state

    def _query_params(self, *args: str) -> tuple[Any, ...]:
        return tuple(self._query_param(x) for x in args)

    def params(self) -> T:
        """
        Get all parameters of the channel.
        """
        keys = self.param_names()
        values = self._query_params(*keys)
        assert len(keys) == len(values)
        builder = self.param_type()
        return builder(**{k: v for k, v in zip(keys, values)})

    def set_params(self, params: T) -> None:
        """
        Set all parameters of the channel.
        """
        assert isinstance(params, self.param_type())
        keys = self.param_names()
        values = [getattr(params, k) for k in keys]
        self.setup(**{k: v for k, v in zip(keys, values)})

    def param_names(self) -> Sequence[str]:
        return list(self.__supported_params)


class Pulser(abc.ABC):
    """
    Base abstract class for fault injection hardware.
    """

    # ===
    # Methods to be implemented.
    # ===
    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        Self-description of this instrument.
        """
        pass

    @abc.abstractmethod
    def channels(self) -> Mapping[str, PulserChannel[Any]]:
        """
        Get the list of channels available on the instrument.
        """
        pass

    @abc.abstractmethod
    def _clear(self, pop_errors: bool) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def output_enabled(self) -> bool:
        """
        Is the output enabled?
        """
        pass

    @abc.abstractmethod
    def set_output_enabled(self, enabled: bool):
        """
        Enable or disable the pulser output.
        """
        pass

    @abc.abstractmethod
    def setup_trigger(
        self,
        source: TriggerSource = TriggerSource.external,
        slope: Slope = Slope.rising,
    ):
        """
        Configure the trigger of the pulser.
        """
        pass

    # ===
    # Methods provided
    # ===
    def disable(self):
        """
        Disable the pulser.

        This method should ensure that it will be safe to touch the device,
        typically used in destructors. Default implementation calls
        ``set_output_enabled(False)``.
        """
        self.set_output_enabled(False)

    def __del__(self):
        self.disable()

    def channel_names(self) -> Sequence[str]:
        """
        Get the list of channel names.
        """
        return list(self.channels().keys())

    def channel_list(self) -> Sequence[PulserChannel[Any]]:
        """
        Get the list of channels.
        """
        return list(self.channels().values())

    def default_channel(self) -> PulserChannel[Any]:
        """
        Return a conventional default channel for the pulser.

        The default implementation returns the first channel.
        """
        return self.channel_list()[0]

    def clear(self, pop_errors: bool = True) -> Sequence[str]:
        """
        Clear pending errors on the instrument.
        """
        return self._clear(pop_errors)

    def __len__(self):
        return len(self.channels())

    def __iter__(self):
        return iter(self.channels())

    def __getitem__(self, channel: str) -> PulserChannel[Any]:
        return self.channels()[channel]


class EMPulser:
    """
    Inherit this class to mark your class as an EM injector.
    """

    def channels(self) -> Mapping[str, PulserChannel[EMPulseParams]]: ...


class VGlitchPulser(Protocol):
    """
    Inherit this class to mark your class as a voltage glitch injector.
    """

    def channels(self) -> Mapping[str, PulserChannel[GlitchParams]]: ...


class LaserPulser(Protocol):
    """
    Inherit this class to mark your class as a laser injector.
    """

    def channels(self) -> Mapping[str, PulserChannel[LaserParams]]: ...