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
from typing import Sequence


class HasScreenShot(abc.ABC):
    @abc.abstractmethod
    def get_screenshot(self):
        """
        Get a screenshot of the instrument screen.
        Obviously this is not supported by screen-less devices.
        See PIL.Image_ for available methods.

        >>> scope.get_screenshot().save('screenshot.jpg')

        :rtype: PIL.Image_

        .. _PIL.Image: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
        """
        pass


class HasSetupStorage(abc.ABC):
    @abc.abstractmethod
    def setup_slots(self) -> Sequence[str]:
        pass

    @abc.abstractmethod
    def setup_load(self, name: str) -> None:
        """
        Load instrument settings from the specified file preset on the device.
        Can be used instead of setting up the various channels and parameters
        manually from Python.

        To create such a preset, use the instrument interface or call
        :meth:`setup_save`.

        >>> instr.setup_load('my_preset')

        :param name: file name to load (absolute or relative to default preset
                     location)
        """
        pass

    @abc.abstractmethod
    def setup_save(self, name: str) -> None:
        """
        Save the current instrument settings to the specified file on the
        device.

        To later load this preset, use the instrument interface or call
        :meth:`setup_load`.

        >>> instr.setup_save('my_preset')

        :param name: file name to save (absolute or relative to default preset
                     location)
        """
        pass


class HasWaveformStorage(abc.ABC):
    @abc.abstractmethod
    def write_waveform(self, channel: str, path: str, temporary: bool = False) -> None:
        """
        Write waveform of a channel on the scope local storage.

        :param channel: Channel to be saved
        :param path: Path to the waveform on the device.
        :param temporary:
            if True, will store the waveform in a temporary directory, thus you
            can only pass the file name.
        """
        pass

    @abc.abstractmethod
    def read_waveform(self, path: str, temporary=False, reshape=False):
        """
        Read a binary file at a given location on the scope.

        :param path: Path to the waveform on the device.
        :param temporary:
            If True, will look into the scope internal waveform directory,
            thus you can only pass the file name.
        :param reshape:
            If true, will reshape the data according to the current
            segmentation policy and scope acquisition.
        :returns: the data read.
        """
        pass