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
import scipy.signal as signal

from secbench.processing.oracles.sliding import (
    sliding_kurt as oracle_sliding_kurt,
)
from secbench.processing.oracles.sliding import (
    sliding_mean as oracle_sliding_mean,
)
from secbench.processing.oracles.sliding import (
    sliding_skew as oracle_sliding_skew,
)
from secbench.processing.oracles.sliding import (
    sliding_std as oracle_sliding_std,
)
from secbench.processing.oracles.sliding import (
    sliding_var as oracle_sliding_var,
)
from secbench.processing.signal import (
    downsample,
    generate_lp_firls,
    sliding_kurt,
    sliding_mean,
    sliding_skew,
    sliding_std,
    sliding_var,
)


def test_lp_generation():
    fs = 1e6
    xs = np.linspace(0, 1, int(fs))

    # Reference, at 50Hz
    ref = np.sin(2 * np.pi * xs * 50 + 0.5)
    noise = 0.2 * np.random.randn(int(fs))

    ref_noisy = ref + noise
    print(np.var(ref_noisy))

    taps = generate_lp_firls(1e3, 5e3, fs)
    ys = signal.lfilter(taps, 1, ref_noisy)
    var_ref = np.var(ref_noisy)
    var_ys = np.var(ys)

    # Not sure this is very robust... We can look at the FFT if high
    # frequencies were removed.
    assert var_ys < var_ref


def test_downsample():
    data_1d = np.random.randint(0, 256, size=10000)
    data_2d = np.random.randint(0, 256, size=(10, 10000))

    # Downsample should have no effect (more samples that needed requested)
    x, y = downsample(data_1d, samples_out=100001)
    assert y.shape == (10000,)
    assert x.shape == (10000,)

    # Basic test, with sample_out
    x, y = downsample(data_1d, samples_out=100)
    assert x.shape == y.shape
    assert x.shape[0] == 100

    # Basic test, with factor
    x, y = downsample(data_1d, factor=2)
    assert y.shape == (5000,)
    assert x.shape == (5000,)

    # Will generate a 0-length trace => not allowed
    with pytest.raises(ValueError):
        downsample(data_1d, factor=100000000)

    # Invalid input parameter
    with pytest.raises(ValueError):
        downsample(data_1d, samples_out=-1)

    # Invalid input parameter
    with pytest.raises(ValueError):
        downsample(data_1d, factor=-1)

    # Basic test (2d version), samples_out
    x, y = downsample(data_2d, samples_out=100)
    assert x.shape == y.shape
    assert x.shape[0] == 10
    assert x.shape[1] == 100


def test_downsample_bug():
    # Should not fail, see issue #232968
    x = np.arange(5000000)
    _, _ = downsample(x, factor=10)


def test_sliding_mean():
    wsize = 10
    data = np.random.randint(0, 256, size=(100, 1000)).astype(np.float32)

    # 1D test
    out = sliding_mean(data[0], window_size=wsize)

    assert np.all(out[: wsize - 1] == 0)
    assert np.all(out >= 0)
    assert np.all(out < 256)

    # compare with the oracle
    oracle_out = oracle_sliding_mean(data[0], wsize)
    assert np.allclose(oracle_out[0], out[0])

    # 2D test
    out = sliding_mean(data, window_size=wsize)

    assert data.shape[0] == 100
    assert np.all(out[:, : wsize - 1] == 0)
    assert np.all(out >= 0)
    assert np.all(out < 256)

    # compare with the oracle
    oracle_out = oracle_sliding_mean(data, wsize)
    assert np.allclose(oracle_out, out)


def test_sliding_var_std():
    wsize = 100
    data = np.random.randint(0, 256, size=(100, 1000)).astype(np.float32)
    # 1D test
    out_v = sliding_var(data[0], window_size=wsize)
    out_std = sliding_std(data[0], window_size=wsize)

    assert np.all(out_v >= 0)
    assert np.all(out_v < 256**2)
    assert np.allclose(out_std**2, out_v)

    # compare with the oracle
    oracle_out_v = oracle_sliding_var(data[0], wsize)
    assert np.allclose(oracle_out_v[0], out_v[0])
    oracle_out_std = oracle_sliding_std(data[0], wsize)
    assert np.allclose(oracle_out_std[0], out_std[0])

    # 2D test
    out_v = sliding_var(data, window_size=wsize)
    out_std = sliding_std(data, window_size=wsize)

    assert np.all(out_v >= 0)
    assert np.all(out_v < 256**2)
    assert np.allclose(out_std**2, out_v)

    # compare with the oracle
    oracle_out_v = oracle_sliding_var(data, wsize)
    assert np.allclose(oracle_out_v, out_v)
    oracle_out_std = oracle_sliding_std(data, wsize)
    assert np.allclose(oracle_out_std, out_std)


def test_sliding_skew():
    wsize = 10
    data = np.random.randint(0, 256, size=(100, 1000)).astype(np.float64)

    # 1D test
    out = sliding_skew(data[0], window_size=wsize)
    assert np.all(out[: wsize - 1] == 0)

    # compare with the oracle
    oracle_out = oracle_sliding_skew(data[0], wsize)
    assert np.allclose(oracle_out[0], out[0])

    # 2D test
    out = sliding_skew(data, window_size=wsize)

    assert data.shape[0] == 100
    assert np.all(out[:, : wsize - 1] == 0)

    # compare with the oracle
    oracle_out = oracle_sliding_skew(data, wsize)
    assert np.allclose(oracle_out, out)


def test_sliding_kurt():
    wsize = 10
    data = np.random.randint(0, 256, size=(100, 1000)).astype(np.float64)

    # 1D test
    out = sliding_kurt(data[0], window_size=wsize)
    assert np.all(out[: wsize - 1] == 0)

    # compare with the oracle
    oracle_out = oracle_sliding_kurt(data[0], wsize)
    assert np.allclose(oracle_out[0], out[0])

    # 2D test
    out = sliding_kurt(data, window_size=wsize)
    assert data.shape[0] == 100
    assert np.all(out[:, : wsize - 1] == 0)

    # compare with the oracle
    oracle_out = oracle_sliding_kurt(data, wsize)
    assert np.allclose(oracle_out, out)