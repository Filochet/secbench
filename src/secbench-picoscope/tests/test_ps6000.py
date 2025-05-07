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

from secbench.api import get_bench
from secbench.api.enums import Coupling, Decimation, Slope
from secbench.api.exceptions import InvalidParameter
from secbench.picoscope import PicoPS6000Scope
from secbench.picoscope.error import PicoscopeApiError


def has_scope() -> bool:
    hw = get_bench().get(PicoPS6000Scope, required=False)
    return hw is not None


pytestmark = pytest.mark.skipif(not has_scope(), reason="no PS6000 available")


@pytest.fixture(scope="module")
def scope():
    dev = get_bench().get(PicoPS6000Scope)
    yield dev
    dev.close_unit()


def test_description(scope):
    assert "picoscope" in scope.description.lower()


def test_channels(scope):
    assert set(scope.channels()) == {"A", "B", "C", "D"}


def test_reset_defaults(scope):
    scope.reset()
    for channel in scope.channels().values():
        assert not channel.enabled()


@pytest.mark.parametrize("channel", ["A", "B"])
def test_right_sequence_order(scope, channel):
    scope.reset()
    channel = scope[channel]
    channel.setup(range=1, offset=0, coupling=Coupling.dc, decimation=Decimation.sample)
    scope.set_horizontal(samples=1000, duration=1)


def test_enable_disable_channel(scope):
    scope.reset()
    for channel in scope.channels().values():
        assert not channel.enabled()
        channel.setup(range=1, coupling=Coupling.dc, offset=0)
        assert channel.enabled()
        channel.disable()
        assert not channel.enabled()


@pytest.mark.parametrize("range", [5, 1, 1e-1, 1e-2])
@pytest.mark.parametrize("offset", [-0.5, -0.1, 0, 0.1, 0.5])
@pytest.mark.parametrize("coupling", [Coupling.ac, Coupling.dc])
def test_channel_setup(scope, range, offset, coupling):
    scope.reset()
    if offset > range:
        return
    for channel in scope.channels().values():
        channel.setup(range=range, offset=offset, coupling=coupling)
        assert channel.coupling() == coupling
        assert channel.range() >= range
        assert channel.offset() == pytest.approx(offset)
        assert channel.enabled()


def test_invalid_channel_setup(scope):
    scope.reset()
    channel = scope.channel_list()[0]

    with pytest.raises(InvalidParameter) as excinfo:
        channel.setup(range=200, offset=0, coupling=Coupling.dc)
    assert "voltage range is too high" in str(excinfo.value)

    with pytest.raises(PicoscopeApiError):
        channel.setup(range=1, coupling=Coupling.dc, offset=50)


@pytest.mark.parametrize("horizontal_args", [(1e-6, 1000, None), (1e-8, None, 1e-6)])
@pytest.mark.parametrize("channel", ["A", "B"])
def test_set_horizontal(scope, horizontal_args, channel):
    scope.reset()
    channel = scope[channel]
    interval, samples, duration = horizontal_args
    channel.setup(range=1, offset=0, coupling=Coupling.dc, decimation=Decimation.sample)
    scope.set_horizontal(interval=interval, samples=samples, duration=duration)
    if interval:
        assert scope.horizontal_interval() == pytest.approx(interval, abs=1e-2)
    if samples:
        assert scope.horizontal_samples() >= samples
    if duration:
        assert scope.horizontal_duration() == pytest.approx(duration, abs=1e-2)


@pytest.mark.parametrize("channel", ["A", "B"])
@pytest.mark.parametrize("slope", [Slope.rising, Slope.falling, Slope.either])
@pytest.mark.parametrize("level", [-0.5, 0, 0.5, 1])
@pytest.mark.parametrize("delay", [0, 0.2])
def test_trigger(scope, slope, level, delay, channel):
    scope.reset()
    scope[channel].setup(
        range=10, coupling=Coupling.dc, offset=0, decimation=Decimation.sample
    )
    scope.set_horizontal(interval=8e-9, duration=80e-6)
    scope.set_trigger(channel, slope=slope, level=level, delay=delay)