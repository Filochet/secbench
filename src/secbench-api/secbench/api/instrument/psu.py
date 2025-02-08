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

import abc
from typing import List


class PowerSupplyChannel(abc.ABC):
    """
    Base abstract class for a power supply channel.
    """

    @abc.abstractmethod
    def set_output_enabled(self, enabled: bool):
        """
        Enable the output of this channel.
        """
        pass

    @abc.abstractmethod
    def output_enabled(self) -> bool:
        pass

    @abc.abstractmethod
    def set_voltage(self, voltage: float):
        """
        Set the channel output voltage (in volts).
        """
        pass

    @abc.abstractmethod
    def voltage(self) -> float:
        """
        Return the current voltage value (in volts).
        """
        pass

    @abc.abstractmethod
    def set_current_limit(self, current: float):
        """
        Maximum current limit for the channel in Ampere.
        """
        pass

    @abc.abstractmethod
    def current_limit(self) -> float:
        """
        Return the current value in Ampere.
        """
        pass


class PowerSupply(abc.ABC):
    """
    Base abstract class for a Power Supply hardware.
    """

    # Methods to implement

    @abc.abstractmethod
    def description(self) -> str:
        """
        The self-description of this instrument.

        This is typically a string representation of the device serial number.

        :Example:

        >>> alim.description
        'SPD3XIDD4R5542,1.01.01.02.05,V3.0'
        """
        pass

    @abc.abstractmethod
    def _clear(self, pop_errors: bool):
        pass

    @abc.abstractmethod
    def get_channel(self, channel: str) -> PowerSupplyChannel:
        """
        Return a specific power channel output.
        """
        pass

    @abc.abstractmethod
    def channels(self) -> List[str]:
        pass

    def default_channel(self):
        """
        Return a conventional default channel for the AFG.

        The default implementation returns the first channel.
        """
        return self.get_channel(self.channels()[0])

    def __getitem__(self, key):
        return self.get_channel(key)

    # Methods provided

    def clear(self, pop_errors=True):
        """
        Clear pending measurements, status flags and optionally pop errors.

        Can be used to bring back the scope in a usable mode if something goes
        wrong.
        """
        return self._clear(pop_errors)