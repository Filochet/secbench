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
from typing import Any, Iterator, Protocol

from typing_extensions import TypeAlias

from ..backend.types import Backend
from ..bench import HardwareInfo, PyVisaScanner

logger = logging.Logger(__name__)

PyVisaResourceManager: TypeAlias = Any
PyVisaBuildArgs: TypeAlias = tuple[PyVisaResourceManager, str]


class PyVisaBackend(Backend):
    """
    Implementation of the :py:class:`Backend` interface through `PyVisa <https://pyvisa.readthedocs.io/en/latest>`__.

    This backend allows interacting with any instrument supported by the pyvisa framework.
    """

    def __init__(self, instr):
        self._instr = instr
        self._eom = False

    def set_timeout(self, secs: float):
        self._instr.timeout = 1000 * secs

    def set_eom(self, b: bool):
        self._eom = b

    def close(self):
        self._instr.close()

    def write(self, cmds: str):
        self._instr.write(cmds)

    def query(self, cmds: str) -> str:
        return self._instr.query(cmds).rstrip()

    def query_raw(self, cmds: str, size: int) -> bytes:
        self.write(cmds)
        return self._instr.read_bytes(size, break_on_termchar=self._eom)


class PyVisaDiscoverableMixin(Protocol):
    """
    Make your hardware discoverable through PyVisa.

    You must define :py:meth:`PyVisaDiscoverableMixin._pyvisa_match_id`.
    """

    @classmethod
    @abc.abstractmethod
    def _pyvisa_match_id(cls, rm, path: str) -> bool:
        """
        A predicate that matches the device.

        :param rm: a pyvisa ressource manager instance
        :param path: pyvisa device descriptor
        :returns: ``True`` is the given instance is
        """
        pass

    @classmethod
    def _pyvisa_configure(cls, backend: PyVisaBackend):
        """
        An optional hook called after the pyvisa backend is created.

        Overriding this method allows customizing the
        backend properties.
        """
        pass

    @classmethod
    def is_supported(cls, hardware_info: HardwareInfo) -> bool:
        return hardware_info.has_scanner(PyVisaScanner)

    @classmethod
    def build(cls, hardware_info: HardwareInfo, args: PyVisaBuildArgs):
        rm, path = args
        backend = PyVisaBackend(rm.open_resource(path))
        cls._pyvisa_configure(backend)
        return cls.from_backend(backend, hardware_info.user_config())

    @classmethod
    def discover(cls, hw_info: HardwareInfo) -> Iterator[PyVisaBuildArgs]:
        scanner = hw_info.get_scanner(PyVisaScanner)
        if scanner is None:
            logger.info("PyVisaScanner not loaded, cannot discover pyvisa devices.")
            return

        for dev in scanner.devices():
            if cls._pyvisa_match_id(scanner.resource_manager(), dev):
                logger.info(f"found matching PyVisa device {dev}.")
                yield scanner.resource_manager(), dev