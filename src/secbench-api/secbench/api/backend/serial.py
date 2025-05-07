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
from typing import Iterator, Optional

from typing_extensions import TypeAlias

from ..backend.types import Backend
from ..bench import HardwareInfo, SerialScanner
from ..instrument import RawInstrument

logger = logging.getLogger(__name__)

SerialBuildArgs: TypeAlias = tuple[str, str]


class SerialBackend(Backend):
    """
    An instrument backend over serial port.

    This backend uses pyserial under the hood for serial communications.
    """

    def __init__(self, path: str, serial_number: str | None = None, **kwargs):
        import serial

        self._encoding = "utf-8"
        self._eom = None
        self._use_readline = True
        self._comm = serial.Serial(path, **kwargs)
        self._serial_number = serial_number or ""

    def serial_number(self) -> str:
        return self._serial_number

    def _normalize_command(self, msgs: str) -> bytearray:
        bs = bytearray()
        bs.extend(msgs.encode(self._encoding))
        if self._eom and not bs.endswith(self._eom):
            bs += self._eom
        return bs

    def _normalize_and_write_raw(self, msgs: str):
        cmd = self._normalize_command(msgs)
        logger.debug(f"sending command: {cmd}")
        self._comm.write(cmd)
        self._comm.flush()

    def _drop_line(self):
        line = self._comm.readline()
        logger.debug(f"ignoring line: {line}")

    def set_end_of_message(self, msg: Optional[bytes]):
        self._eom = msg

    def flush(self):
        return self._comm.flush()

    def close(self):
        self._comm.close()

    def write(self, cmds: str):
        self._normalize_and_write_raw(cmds)

    def query(self, cmds: str) -> str:
        self.write(cmds)
        if self._use_readline:
            msg = self._comm.readline()
            return msg.decode(self._encoding)
        else:
            msg = self._comm.readall()
            return msg.decode(self._encoding)

    def query_raw(self, cmds: str, size: int) -> bytes:
        self.write(cmds)
        return self._comm.read(size)

    def set_timeout(self, secs: float):
        self._comm.timeout = secs


class SerialDiscoverableMixin(abc.ABC):
    """
    Make your hardware discoverable through a :py:class:`SerialBackend`.

    You must define :py:meth:`SerialDiscoverableMixin._match_serial`.
    """

    @classmethod
    @abc.abstractmethod
    def _match_serial(cls, idn: str) -> bool:
        """
        Predicate to detect if the serial identifier of a serial port
        matches the current hardware.
        """
        raise NotImplementedError()

    @classmethod
    def _serial_options(cls):
        """
        This method can be overwritten to pass additional arguments
        to the serial port constructor.

        The dictionary returned is passed to pyserial's `Serial`
        class constructor.
        """
        return {}

    @classmethod
    def _setup_backend(cls, backend: Backend) -> None:
        return

    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        return hardware_info.has_scanner(SerialScanner)

    @classmethod
    def discover(cls, hw_info: HardwareInfo) -> Iterator[SerialBuildArgs]:
        scanner = hw_info.get_scanner(SerialScanner)
        if scanner is None:
            logger.info("SerialScanner not loaded, cannot discover serial devices.")
            return

        for dev in scanner.devices():
            if not dev.serial_number:
                continue
            serial = dev.serial_number.lower()
            if cls._match_serial(dev.serial_number.lower()):
                logger.debug(f"found matching serial device {dev}.")
                yield dev.device, serial

    @classmethod
    def build(
        cls: RawInstrument | SerialDiscoverableMixin,
        hardware_info: HardwareInfo,
        args: SerialBuildArgs,
    ):
        path, serial_number = args
        backend = SerialBackend(
            path, serial_number=serial_number, **cls._serial_options()
        )
        cls._setup_backend(backend)
        return cls.from_backend(backend, hardware_info.user_config())