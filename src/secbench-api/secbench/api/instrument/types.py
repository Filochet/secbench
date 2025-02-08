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

import logging
from typing import TYPE_CHECKING, Iterable, Protocol

from ..backend.types import Backend
from ..exceptions import InstrumentPendingErrors

if TYPE_CHECKING:
    from ..bench import UserConfig

logger = logging.getLogger(__name__)


class RawInstrument(Protocol):
    """
    Interface that **must** be implemented by SCPI-like instruments.

    For remainder, SCPI (Standard Commands for Programmable Instruments) is a
    command format specification. Instruments that complies to SCPI describe
    their commands in their programmer manual in terms of ``query`` and
    ``write`` commands.

    :ivar backend: Current :py:class:`Backend` used by the instrument.

    """

    backend: Backend

    @classmethod
    def from_backend(cls, backend: Backend, cfg: UserConfig) -> RawInstrument:
        """
        Create an instance from a Backend.
        """
        ...

    def has_error(self) -> bool:
        """
        Return ``True`` if the instrument has an error pending.
        """
        ...

    def pop_next_error(self) -> str | None:
        """
        Pop the next pending error from the instrument.
        """
        ...


class InstrumentMixin(RawInstrument):
    """
    Defines convenience methods in instruments.
    """

    def pop_errors(self) -> list[str]:
        errors = []
        while True:
            if not self.has_error():
                break
            last_error = self.pop_next_error()
            errors.append(last_error)
        return errors

    def query(self, msgs: str) -> str:
        return self.backend.query(msgs)

    def query_raw(self, msgs: str, size: int) -> bytes:
        return self.backend.query_raw(msgs, size)

    def write(self, msgs: str):
        self.write_unchecked(msgs)
        pending_errors = self.pop_errors()
        if pending_errors:
            logger.debug(f"command '{msgs}' generated errors: {pending_errors}")
            raise InstrumentPendingErrors(pending_errors)

    def write_unchecked(self, msgs: str):
        self.backend.write(msgs)


class QueryWriteInstrument(Protocol):
    def query(self, msgs: str) -> str: ...

    def write(self, msgs: str) -> str: ...

    def join_commands(self, msgs: Iterable[str]) -> str: ...


class WriteManyMixin:
    """
    Provide a `write_many` method for Instruments that support chained commands.
    """

    def write_many(self: QueryWriteInstrument, *msgs: str):
        self.write(self.join_commands(msgs))

    def query_many(self: QueryWriteInstrument, *msgs: str) -> str:
        return self.query(self.join_commands(msgs))