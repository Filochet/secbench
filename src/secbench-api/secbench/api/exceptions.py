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
"""
Exceptions raised by the secbench module.
"""


class SecbenchError(Exception):
    """
    Base exception for *secbench* related operations.
    """

    pass


class BackendError(SecbenchError):
    """
    An error that occurred in a communication backend.
    """

    def __init__(self, msg):
        super().__init__(msg)


class NoSuchHardwareError(SecbenchError):
    """
    Exception raised when no device of the requested hardware type is
    available.
    """

    def __init__(self, hw_class: type, what=""):
        super().__init__(f"no hardware found for {hw_class.__name__}{what}")


class MissingDependency(SecbenchError):
    """
    A dependency (e.g., a Python package) is missing for using a feature.
    """

    pass


class InstrumentError(SecbenchError):
    """
    Base exception for instrument-related operations.
    """

    def __init__(self, msg):
        super().__init__(msg)


class InstrumentPendingErrors(InstrumentError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(
            f"there are {len(errors)} pending errors on the instruments: {errors}"
        )


class InstrumentUnsupportedFeature(InstrumentError):
    def __init__(self, msg):
        super().__init__(f"operation not supported: {msg}")


class UnsupportedFeature(SecbenchError):
    """
    Exception raised when attempting to use a feature that is not supported.
    """

    def __init__(self, pkg_name: type, usage=""):
        self.pkg_name = pkg_name
        self.usage = ""
        super().__init__(f"missing package {pkg_name}.")


class InvalidParameter(InstrumentError):
    """
    Invalid parameters requested to an instrument.
    """

    pass


class NoSuchChannelError(InstrumentError):
    """
    Exception raised when accessing a channel that is not available on the
    instrument.
    """

    def __init__(self, channel):
        super().__init__(f"no such channel: {channel}")