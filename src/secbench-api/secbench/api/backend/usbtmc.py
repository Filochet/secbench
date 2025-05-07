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
import time
from typing import Iterator, Optional

from typing_extensions import TypeAlias

from ..backend.types import Backend
from ..bench import HardwareInfo, LibUdevScanner, UsbDeviceEntry, UserConfig
from ..exceptions import InstrumentError
from ..instrument import RawInstrument

logger = logging.getLogger(__name__)

USBTMCBuildArgs: TypeAlias = str


class USBTMCBackend(Backend):
    """
    The USBTMC backend allows communicating with instruments through USB.
    """

    def __init__(self, path: str, buffering=None):
        if buffering is None:
            buffering = 10 * 1024 * 1024
        self.force_eom = False
        self.force_flush = True
        self.query_delay = 0
        logger.debug(f"opening USBTMC device {path} (buffer size={buffering})")
        try:
            self._fd_raw = open(path, "w+b", buffering=buffering)  # noqa: SIM115
        except IOError as e:
            logger.error(f"unable to open path {path}")
            raise InstrumentError(
                f"failed to open {path}; check USB cable and permissions"
            ) from e
        super().__init__()

    def set_timeout(self, secs: float):
        logger.info(
            "set_timeout method has no effect on USBTMC devices (calls are blocking)"
        )

    def close(self):
        self._fd_raw.close()

    def query(self, cmds: str):
        if self.force_eom and not cmds.endswith("\n"):
            cmds = cmds + "\n"
        self.write(cmds)
        if self.query_delay:
            time.sleep(self.query_delay)
        return self._fd_raw.readline().rstrip().decode("utf-8")

    def query_raw(self, cmds: str, size: int) -> bytes:
        if self.force_eom and not cmds.endswith("\n"):
            cmds = cmds + "\n"
        self.write(cmds)
        if self.query_delay:
            time.sleep(self.query_delay)
        return self._fd_raw.read(size)

    def write(self, cmds: str):
        if self.force_eom and not cmds.endswith("\n"):
            cmds += "\n"
        try:
            self._fd_raw.write(cmds.encode())
            if self.force_flush:
                self._fd_raw.flush()
        except Exception as e:
            logger.error(f"failed to send command '{cmds}' -> {e}")
            raise InstrumentError("File I/O error") from e


class USBTMCDiscoverableMixin(abc.ABC):
    """
    Make your hardware an instance of :py:class:`Discoverable[USBTMCBuildArgs]`

    - You must define :py:meth:`USBTMCDiscoverableMixin._usbtmc_match`.
    """

    @classmethod
    @abc.abstractmethod
    def _usbtmc_match(cls, entry: UsbDeviceEntry, cfg: UserConfig) -> bool:
        """
        A predicate that must be implemented to detect if an USB entry matches the current hardware.
        """
        pass

    @classmethod
    def _usbtmc_configure(cls, backend: USBTMCBackend) -> None:
        """
        A hook called after a :py:class:`USBTMCBackend` is constructed.

        This allows applying hardware-specific configuration.
        """
        return

    @classmethod
    def _usbtmc_buffering(cls) -> Optional[int]:
        """
        Buffering mode to use when opening USB file descriptors.
        """
        return None

    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        return hardware_info.has_scanner(LibUdevScanner)

    @classmethod
    def build(
        cls: RawInstrument | USBTMCDiscoverableMixin,
        hardware_info: HardwareInfo,
        path: USBTMCBuildArgs,
    ):
        backend = USBTMCBackend(path, buffering=cls._usbtmc_buffering())
        cls._usbtmc_configure(backend)
        return cls.from_backend(backend, hardware_info.user_config())

    @classmethod
    def discover(cls, hardware_info: HardwareInfo) -> Iterator[USBTMCBuildArgs]:
        scanner = hardware_info.get_scanner(LibUdevScanner)
        if scanner is None:
            logger.info("LibUdevScanner not loaded, cannot discover USBTMC devices.")
            return

        cfg = hardware_info.user_config()
        found = list(filter(lambda x: cls._usbtmc_match(x, cfg), scanner.devices()))

        for dev in found:
            for node in dev.nodes:
                yield node