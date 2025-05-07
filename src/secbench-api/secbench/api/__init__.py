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

from .backend import Backend
from .bench import (
    Bench,
    HardwareInfo,
    LibUdevScanner,
    SerialDeviceEntry,
    SerialScanner,
    UsbCoreScanner,
    UsbDeviceEntry,
    UserConfig,
    VxiDeviceEntry,
    VxiScanner,
    get_bench,
)
from .discovery import Discoverable, DiscoverPolicy, discover, discover_first
from .instrument.features import HasScreenShot, HasSetupStorage, HasWaveformStorage
from .instrument.types import (
    InstrumentMixin,
    QueryWriteInstrument,
    RawInstrument,
    WriteManyMixin,
)
from .types import Location

# NOTE: we import the module ``secbench.instruments`` (if present) to
# make hardware available by default.
try:
    import secbench.instruments  # noqa: F401
except ImportError:
    pass

# NOTE: we import the module ``secbench.picoscope`` (if present) to
# make hardware available by default.
try:
    import secbench.picoscope  # noqa: F401
except ImportError:
    pass


def version() -> str:
    """
    Current version of the :py:mod:`secbench.api` package
    """
    from importlib.metadata import distribution

    return distribution("secbench-api").version


__all__ = [
    "Backend",
    "Bench",
    "DiscoverPolicy",
    "Discoverable",
    "discover",
    "discover_first",
    "Location",
    "get_bench",
    "HardwareInfo",
    "HasSetupStorage",
    "HasWaveformStorage",
    "HasScreenShot",
    "InstrumentMixin",
    "QueryWriteInstrument",
    "SerialDeviceEntry",
    "SerialScanner",
    "UserConfig",
    "UsbDeviceEntry",
    "RawInstrument",
    "LibUdevScanner",
    "UsbCoreScanner",
    "VxiDeviceEntry",
    "VxiScanner",
    "WriteManyMixin",
    "version",
]