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

import numpy as np
import pytest

# We need this import to mak the instruments visible.
from secbench.api import Bench
from secbench.api.enums import BurstMode, Coupling, Decimation, Slope
from secbench.api.exceptions import InstrumentUnsupportedFeature
from secbench.api.helpers import is_clipping
from secbench.api.instrument import Afg, Scope, ScopeAnalogChannel
from secbench.api.instrument.scope import (
    BinarySearchCalibration,
    CalibrationData,
    LinearSearchCalibration,
    StepSearchCalibration,
)

SCOPE_V_DIVISIONS = 5

_BENCH = Bench()


def has_scope() -> bool:
    return _BENCH.get(Scope, required=False) is not None


def has_afg() -> bool:
    return _BENCH.get(Afg, required=False) is not None


def _gen_dynamics_callback(amplitude: float, offset: float):
    """Build a callback for a specific amplitude"""

    volts = amplitude * (np.random.random(size=1000) - 0.5) + offset

    def f(scope: Scope, channel: ScopeAnalogChannel):
        # For testing purposes, we return random data
        r = channel.range()
        o = channel.offset()

        resolution = scope.bit_resolution()
        vmin, vmax = -(r / 2), (r / 2)
        v_dyn = r
        adc_dynamics = 2**resolution

        samples = adc_dynamics * (
            np.clip(volts, vmin, vmax - 1e-9) - o - vmin
        ) / v_dyn - 2 ** (resolution - 1)
        samples = samples.astype(np.int32)

        assert np.all(samples <= 127) and np.all(samples >= -128)
        return CalibrationData(
            range=np.max(samples) - np.min(samples),
            offset=(np.max(samples) + np.min(samples)) / 2,
            is_clipping=is_clipping(samples.astype(np.int8)),
        )

    return f


pytestmark = pytest.mark.skipif(
    not has_afg() or not has_scope(), reason="No Afg or scope found found"
)


@pytest.fixture
def bench():
    return _BENCH


@pytest.fixture
def afg_channel(bench):
    afg = bench.get(Afg)
    afg.clear(pop_errors=True)
    afg_channel = afg.default_channel()
    # Disable the burst mode.
    afg_channel.set_burst_mode(0, BurstMode.triggered)
    yield afg_channel


@pytest.fixture
def scope(bench):
    scope = bench.get_scope()
    for _ in range(10):
        scope.clear()
    yield scope


def set_default_settings(scope: Scope, afg_channel, duration=10e-4, samples=10000):
    channels = list(scope.channels().values())
    scope.set_horizontal(duration=duration, samples=samples)
    ch_1 = channels[1]
    ch_1.setup(
        range=4,
        coupling=Coupling.dc_low_impedance,
        offset=0,
        decimation=Decimation.sample,
    )
    scope.set_trigger(channel=ch_1.name, delay=0, level=0.45, slope=Slope.rising)
    afg_channel.generate_square(1e4, 2, 0, 50)
    afg_channel.set_output_state(True)
    scope.disable_segmented_acquisition()


@pytest.mark.parametrize("samples", [5000, 10000])
@pytest.mark.parametrize("duration", [1e-3, 1e-6])
@pytest.mark.parametrize("segmented_count", [1, 10, 100])
def test_multichannel(scope, afg_channel, samples, duration, segmented_count):
    channels = list(scope.channels().values())
    channel_names = list(scope.channels().keys())
    scope.set_horizontal(duration=duration, samples=samples)
    scope.segmented_acquisition(segmented_count)
    data_shape = np.zeros((segmented_count, samples)).squeeze().shape

    for ch in channels:
        ch.setup(
            range=4,
            coupling=Coupling.dc_low_impedance,
            offset=0,
            decimation=Decimation.sample,
        )
    scope.set_trigger(channel=channels[1].name, delay=0, level=0.45, slope=Slope.rising)
    afg_channel.generate_square(1e4, 2, 0, 50)

    for rnd in range(4):
        afg_channel.set_output_state(False)
        print("multi channel round ", rnd)
        scope.arm()
        afg_channel.set_output_state(True)
        scope.wait()
        d = scope.get_data(*channel_names)
        assert len(d) == len(channel_names)
        for a in d:
            assert a.shape == data_shape


@pytest.mark.parametrize("segmented_count", [1, 10, 100, 500])
def test_continuous_acquisition(scope, afg_channel, segmented_count):
    set_default_settings(scope, afg_channel)
    channels = list(scope.channels().values())
    scope.segmented_acquisition(segmented_count)
    data_shape = np.zeros((segmented_count, scope.horizontal_samples())).squeeze().shape

    for i in range(4):
        print("single acq, round", i)
        scope.arm(iterations=100000)
        scope.wait()
        (d,) = scope.get_data(channels[1].name)
        assert d.shape == data_shape


@pytest.mark.parametrize("segmented_count", [1, 10, 100, 500])
def test_12bit_acquisition(scope, afg_channel, segmented_count):
    try:
        set_default_settings(scope, afg_channel)
        channels = list(scope.channels().values())
        scope.set_data_format(bits=16)
        scope.segmented_acquisition(segmented_count)
        data_shape = (
            np.zeros((segmented_count, scope.horizontal_samples())).squeeze().shape
        )
    except InstrumentUnsupportedFeature:
        pytest.skip("the scope does not support the requested configuration")
        return

    for _i in range(4):
        scope.arm(iterations=100000)
        scope.wait()
        (d,) = scope.get_data(channels[1].name)
        assert d.shape == data_shape
        assert d.dtype == np.int16

    scope.set_data_format(bits=8)


@pytest.mark.parametrize("segmented_count", [10, 100, 200])
@pytest.mark.parametrize("freq", [5e2, 1e3])
@pytest.mark.parametrize("width", [1000, 500])
def test_burst_acquisition(scope: Scope, afg_channel, segmented_count, freq, width):
    try:
        set_default_settings(scope, afg_channel)
        channels = list(scope.channels().values())
        scope.segmented_acquisition(segmented_count)
        data_shape = (
            np.zeros((segmented_count, scope.horizontal_samples())).squeeze().shape
        )
    except InstrumentUnsupportedFeature:
        pytest.skip("the scope does not support the requested configuration")
        return

    afg_channel.generate_pulse(freq, 2, width_ns=width)
    afg_channel.set_burst_mode(segmented_count, BurstMode.triggered)
    afg_channel.set_output_state(True)
    scope.arm()
    afg_channel.force_trigger()
    scope.wait(iterations=5000)
    (d,) = scope.get_data(channels[1].name)
    assert d.shape == data_shape
    afg_channel.set_output_state(False)


def test_is_clipping_helper():
    ds_0 = np.arange(1, 64, dtype=np.uint8)
    ds_1 = np.array([[1, 10], [-3, 0]], dtype=np.int8)
    ds_2 = np.array([127, -128], dtype=np.int8)
    ds_3 = np.array([[127, 0], [0, -128]], dtype=np.int8)
    ds_4 = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 0], [0, 0, 0]])
    assert not is_clipping(ds_0)
    assert not is_clipping(ds_1)
    assert is_clipping(ds_2)
    assert is_clipping(ds_3)

    assert is_clipping(ds_0, ymin=1)
    assert is_clipping(ds_0, ymax=63)
    assert is_clipping(ds_1, ymax=5)
    assert is_clipping(ds_1, ymin=1)
    assert not is_clipping(ds_4)
    assert not is_clipping(ds_4, ymax=5, ratio=0.55)
    assert is_clipping(ds_4, ymax=5, ratio=0.3)


# NOTE: LeCroy must be set in Timebase -> Horizontal setup -> "Set Maximun Memory"
def test_clip(scope: Scope, afg_channel):
    set_default_settings(scope, afg_channel)
    channels = list(scope.channels().values())

    ii8 = np.iinfo(np.int8)
    ii16 = np.iinfo(np.int16)

    scope.arm()
    scope.wait()
    # Not clipped value
    (data,) = scope.get_data(channels[1].name)
    if data.dtype == np.int8:
        assert data.max() != ii8.max
        assert data.min() != ii8.min
    elif data.dtype == np.int16:
        # NOTE: int16.max (lecroy) is 32512 != ii16.max=32767
        assert data.max() != ii16.max - 255
        assert data.min() != ii16.min

    channels[1].setup(
        range=0.05,
        coupling=Coupling.dc_low_impedance,
        offset=0,
        decimation=Decimation.sample,
    )
    # clipped value
    scope.arm()
    scope.wait()
    data = scope.get_data(channels[1].name)[0]
    if data.dtype == np.int8:
        assert data.max() == ii8.max
        assert data.min() == ii8.min
    elif data.dtype == np.int16:
        # NOTE: int16.max (lecroy) is 32512 != ii16.max=32767
        assert data.max() == ii16.max - 255
        assert data.min() == ii16.min


def test_calibrate_vrange_sim(scope: Scope):
    # Pick the first channel of the scope for testing
    ch = list(scope.channels().keys())[0]

    with pytest.raises(ValueError):
        scope.calibrate(
            ch,
            _gen_dynamics_callback(3, 0),
            method=LinearSearchCalibration(max_iterations=2, volts_min=0.01),
        )
    scope.calibrate(
        ch,
        _gen_dynamics_callback(0.2, 0),
        method=LinearSearchCalibration(max_iterations=200, volts_min=0.01),
    )
    assert scope[ch].range() >= 0.1

    scope.calibrate(
        ch,
        _gen_dynamics_callback(0.2, 0),
        method=BinarySearchCalibration(volts_min=0.01),
    )
    assert scope[ch].range() >= 0.1

    # 0.05 Volts
    scope.calibrate(
        ch,
        _gen_dynamics_callback(0.05, 0),
        method=BinarySearchCalibration(volts_min=0.01),
    )
    assert scope[ch].range() < 1.0
    assert abs(scope[ch].range() - 0.05) < 0.1

    # 0.05 Volts
    scope.calibrate(
        ch,
        _gen_dynamics_callback(0.05, 0.001),
        method=StepSearchCalibration(volts_min=0.01),
    )
    assert scope[ch].range() > 0.05


@pytest.mark.parametrize("dynamics", [1, 2, 7])
@pytest.mark.parametrize("offset", [0.1, 0.5, -0.5])
def test_calibrate_step_sim(scope: Scope, dynamics, offset):
    ch = list(scope.channels().keys())[0]
    # 0.05 Volts
    scope.calibrate(
        ch,
        _gen_dynamics_callback(dynamics, offset),
        method=StepSearchCalibration(volts_min=0.01, volts_max=10, profile=[0.7]),
    )

    # Check that all the dynamics is handled
    assert scope[ch].range() > dynamics
    # Check that the offset is mostly well set
    assert abs(scope[ch].offset() - offset) < 0.10 * abs(offset)


def _is_clipping_callback(scope, channel):
    scope.arm()
    scope.wait()
    (d,) = scope.get_data(channel)
    return CalibrationData(range=0, offset=0, is_clipping=is_clipping(d))


def test_calibrate_vrange_real(scope: Scope, afg_channel):
    # Pick the first channel of the scope for testing
    ch = list(scope.channels().keys())[0]
    set_default_settings(scope, afg_channel)
    # A hacky way to get the AFG
    afg = afg_channel._parent
    if len(afg.channels()) < 2:
        pytest.skip("this test requires an AFG with 2 channels")

    afg_ch_2 = afg.get_channel(afg.channels()[1])
    afg_ch_2.set_output_state(True)

    def test_volts(amplitude):
        afg_ch_2.generate_square(1e4, amplitude, 0, 50)
        scope.calibrate(ch, _is_clipping_callback, BinarySearchCalibration())
        # Note: we know the signal goes from [-amplitude; amplitude].
        # Furthermore, the scope total positive range is obtain by multiplying
        # the unit range with SCOPE_V_DIVISIONS (usually 5).
        assert scope[ch].range() * SCOPE_V_DIVISIONS > 0.5 * amplitude

    test_volts(0.01)
    test_volts(0.05)
    test_volts(0.1)
    test_volts(0.3)


def test_config(scope: Scope):
    cfg = scope.config()
    assert "scope_name" in cfg
    assert "scope_description" in cfg

    ch = list(scope.channels().keys())[0]
    scope[ch].setup(
        range=4,
        coupling=Coupling.dc_low_impedance,
        offset=0,
        decimation=Decimation.sample,
    )
    cfg = scope.config(channels=[ch])
    assert f"scope_channel_{ch}" in cfg