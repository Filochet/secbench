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

import pytest

from secbench.api import Bench
from secbench.api.enums import Slope, TriggerSource
from secbench.api.instrument import EMPulser, HasSetupStorage, Pulser

_BENCH = Bench()


@pytest.fixture
def bench():
    return _BENCH


def has_pulser() -> bool:
    return _BENCH.get(EMPulser, required=False) is not None


pytestmark = pytest.mark.skipif(not has_pulser(), reason="no pulser available")


@pytest.fixture
def pulser(bench):
    pulser = bench.get(EMPulser)
    yield pulser
    pulser.disable()
    del pulser


def test_getters_setters(pulser: Pulser):
    ch = pulser.default_channel()
    ch.setup(delay_ns=30)
    assert ch.params().delay_ns == pytest.approx(30)

    ch.setup(width_ns=10)
    assert ch.params().width_ns == pytest.approx(10)

    ch.setup(amplitude=50)
    assert ch.params().amplitude == pytest.approx(50)

    pulser.set_output_enabled(True)
    assert pulser.output_enabled()
    pulser.set_output_enabled(False)
    assert not pulser.output_enabled()

    pulser.setup_trigger(TriggerSource.external, Slope.rising)


def test_save_recall(pulser: Pulser):
    if not isinstance(pulser, HasSetupStorage):
        pytest.skip("HasSetupStorage not supported for this pulser")
    ch = pulser.default_channel()
    ch.setup(delay_ns=100, width_ns=10, amplitude=50)
    slot = pulser.setup_slots()[0]
    pulser.setup_save(slot)

    ch.setup(delay_ns=0, width_ns=20)
    pulser.setup_load(slot)
    assert ch.params().delay_ns == pytest.approx(100)
    assert ch.params().width_ns == pytest.approx(10)
    # WARNING: this statement can fail on the NicMax, probably due to noise on
    # the board.
    assert ch.params().amplitude == pytest.approx(50)