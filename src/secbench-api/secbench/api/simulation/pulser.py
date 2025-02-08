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

from typing import Any, Mapping, Sequence

from ..enums import Slope, TriggerSource
from ..instrument import Pulser, PulserChannel
from ..instrument.pulser import EMPulseParams


class SimulatedPulserChannel(PulserChannel[EMPulseParams]):
    def __init__(self, parent: SimulatedPulser, name: str):
        super().__init__()
        self._name = name
        self._parent = parent
        self._enabled = False
        self._params = EMPulseParams(
            delay_ns=0, amplitude=0, width_ns=5, rise_time_ns=5
        )

    @property
    def parent(self):
        return self._parent

    @classmethod
    def param_type(cls):
        return EMPulseParams

    @property
    def name(self) -> str:
        return self._name

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def enabled(self) -> bool:
        return self._enabled

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._params, k, v)

    def _query_param(self, name: str) -> Any:
        return getattr(self._params, name)


class SimulatedPulser(Pulser):
    def __init__(self):
        self._output_enabled = False
        self._channels = {"A": SimulatedPulserChannel(self, "A")}

    @property
    def description(self) -> str:
        return "simulated pulser"

    def channels(self) -> Mapping[str, PulserChannel[Any]]:
        return self._channels

    def _clear(self, pop_errors: bool) -> Sequence[str]:
        return []

    def output_enabled(self) -> bool:
        return self._output_enabled

    def set_output_enabled(self, enabled: bool):
        self._output_enabled = enabled

    def setup_trigger(
        self,
        source: TriggerSource = TriggerSource.external,
        slope: Slope = Slope.rising,
    ):
        pass