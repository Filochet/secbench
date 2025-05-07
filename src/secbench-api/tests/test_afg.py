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

import itertools
import logging

import pytest

from secbench.api import Bench
from secbench.api.enums import (
    BurstMode,
    Function,
    OutputLoad,
    Polarity,
    Slope,
    TriggerSource,
)
from secbench.api.exceptions import InvalidParameter
from secbench.api.instrument import Afg, AfgChannel

logger = logging.getLogger(__name__)

_BENCH = Bench()


@pytest.fixture
def bench():
    return _BENCH


def has_afg():
    return _BENCH.get(Afg, required=False) is not None


pytestmark = pytest.mark.skipif(not has_afg(), reason="no Function generator available")


@pytest.fixture
def afg(bench):
    afg = bench.get(Afg)
    for _ in range(10):
        afg.clear()
    yield afg


def test_getters_setters(afg):
    afg.reset()

    ch: AfgChannel = afg.default_channel()
    ch.set_function(Function.pulse)
    assert ch.function() == Function.pulse
    ch.set_frequency(1e3)
    assert ch.frequency() == 1e3
    ch.set_voltage(1.2, offset=1)
    amplitude, offset = ch.voltage()
    assert amplitude == 1.2
    assert offset == 1

    ch.set_pulse_width(70)
    assert ch.pulse_width() == 70

    ch.set_function(Function.square)
    assert Function.square == ch.function()
    # TODO: integrate this in Tektronix tests
    # if not isinstance(afg, TektronixAFG31000):
    #     ch.set_duty_cycle(20)
    #     assert 20 == ch.duty_cycle()
    ch.set_function(Function.pulse)
    ch.set_duty_cycle(30)
    assert ch.duty_cycle() == 30

    afg.reset()
    ch.set_function(Function.ramp)
    ch.set_ratio(95)
    assert ch.ratio() == 95
    ch.set_burst_mode(3, BurstMode.triggered)
    assert ch.burst_count() == 3
    ch.set_burst_mode(0, BurstMode.triggered)

    ch.set_output_state(enabled=True)
    assert ch.output_state()[0] is True
    ch.set_output_state(enabled=False)
    assert ch.output_state()[0] is False


def test_function(afg):
    afg.reset()
    ch: AfgChannel = afg.default_channel()

    ch.generate_sinus(2e5, 2.5, offset=1)
    assert ch.function() == Function.sinus
    assert ch.frequency() == 2e5
    amplitude, offset = ch.voltage()
    assert amplitude == 2.5
    assert offset == 1

    ch.generate_ramp(1e5, 1.5, offset=0.7, ratio=70)
    assert ch.function() == Function.ramp
    assert ch.frequency() == 1e5
    amplitude, offset = ch.voltage()
    assert amplitude == 1.5
    assert offset == 0.7
    assert ch.ratio() == 70

    ch.generate_square(1000, 2, offset=-1, duty_cycle=20)
    assert ch.function() == Function.square
    assert ch.frequency() == 1000
    amplitude, offset = ch.voltage()
    assert amplitude == 2
    assert offset == -1
    # TODO: integrate that in tektronix tests
    # if not isinstance(afg, TektronixAFG31000):
    #     assert ch.duty_cycle() == 20

    ch.generate_pulse(1e3, 1.12, offset=0.001, width_ns=60)
    assert ch.function() == Function.pulse
    assert ch.frequency() == 1e3
    amplitude, offset = ch.voltage()
    assert amplitude == 1.12
    assert offset == 0.001

    ch.generate_noise(1.1, offset=0)
    assert ch.function() == Function.noise
    amplitude, offset = ch.voltage()
    assert amplitude == 1.1
    assert offset == 0


def test_trigger(afg):
    afg.reset()
    ch: AfgChannel = afg.default_channel()
    sources = [TriggerSource.internal, TriggerSource.external, TriggerSource.manual]
    slopes = [Slope.rising, Slope.falling]
    for t_src, t_slope in itertools.product(sources, slopes):
        try:
            ch.set_trigger_state(source=t_src, slope=t_slope)
        except InvalidParameter:
            logger.info(
                f"unable to test trigger {t_src} and {t_slope}, unsupported for this instrument."
            )
            continue
        source, slope = ch.trigger_state()
        assert source == t_src
        assert slope == t_slope


def test_output_load(afg):
    afg.reset()
    ch: AfgChannel = afg.default_channel()

    ch.set_output_state(True, OutputLoad.ohms_50, Polarity.normal)
    assert ch.output_state() == (True, OutputLoad.ohms_50, Polarity.normal)

    ch.set_output_state(True, OutputLoad.ohms_10k, Polarity.inverted)
    assert ch.output_state() == (True, OutputLoad.ohms_10k, Polarity.inverted)

    ch.set_output_state(True, OutputLoad.inf, Polarity.inverted)
    assert ch.output_state()[1] == OutputLoad.inf