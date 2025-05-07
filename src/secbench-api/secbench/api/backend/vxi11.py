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
from typing import Iterator

from typing_extensions import TypeAlias

from ..backend.types import Backend
from ..bench import HardwareInfo, VxiScanner
from ..exceptions import BackendError
from ..instrument import RawInstrument

logger = logging.getLogger(__name__)

VXIBuildArgs: TypeAlias = str


class VXIBackend(Backend):
    def __init__(self, host: str):
        import vxi11  # type: ignore

        self.inst = vxi11.Instrument(host)
        try:
            self.inst.open()
        except OSError as e:
            raise BackendError(f"could not open {host}; check connectivity") from e
        super().__init__()

    def set_timeout(self, secs: float):
        self.inst.timeout = secs

    def close(self):
        self.inst.close()

    def write(self, cmds: str):
        self.inst.write(cmds)

    def query(self, cmds: str) -> str:
        return self.inst.ask(cmds)

    def query_raw(self, cmds: str, size: int) -> bytes:
        self.inst.write(cmds)
        return self.inst.read_raw(size)


class VXIDiscoverableMixin(abc.ABC):
    """
    Make your hardware an instance of ``Discoverable[VXIBuildArgs]``

    You must implement :py:meth:`VXIDiscoverableMixin._vxi_match_idn`.
    """

    @classmethod
    @abc.abstractmethod
    def _vxi_match_idn(cls, idn: str) -> bool:
        """
        A predicate to detect if an entry matches the current hardware.

        :param idn: the device description returned by the ``*IDN?`` SCPI command.
        """
        pass

    @classmethod
    def _vxi_configure(cls, backend: Backend) -> None:
        """
        A hook called after a :py:class:`VxiBackend` is constructed.

        This allows applying hardware-specific configuration.
        """
        return

    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        return hardware_info.has_scanner(VxiScanner)

    @classmethod
    def discover(cls, hw_info: HardwareInfo) -> Iterator[VXIBuildArgs]:
        scanner = hw_info.get_scanner(VxiScanner)
        if scanner is None:
            logger.info("VxiScanner not loaded, cannot discover pyvisa devices.")
            return
        for dev in scanner.devices():
            if cls._vxi_match_idn(dev.idn):
                logger.info(f"found matching device {dev}.")
                yield dev.host

    @classmethod
    def build(
        cls: RawInstrument | VXIDiscoverableMixin,
        hardware_info: HardwareInfo,
        host: VXIBuildArgs,
    ):
        backend = VXIBackend(host)
        cls._vxi_configure(backend)
        return cls.from_backend(backend, hardware_info.user_config())