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

import numpy as np
import pytest
from scipy import signal
from scipy.fft import rfftn

from secbench.processing import secbench_native_installed
from secbench.processing.helpers import check_array
from secbench.processing.metrics import CondMeanVar
from secbench.processing.signal import (
    fft_filter,
    moving_sum,
    phase_correlation,
    rfft_mag,
)

secbench_native_testcase = pytest.mark.skipif(
    not secbench_native_installed(), reason="secbench_ffi package is not installed"
)


@secbench_native_testcase
@pytest.mark.parametrize("parallel", (False, True))
def test_rfft_mag(parallel):
    def gen_signal(xs):
        signal = np.random.normal(size=xs.shape[0]) * np.cos(3 * xs)
        signal = signal.astype(np.float32)
        return signal

    for w in [123, 512, 1024, 2048]:
        xs = 30 * np.linspace(0, 5, w)
        src = np.array([gen_signal(xs)], dtype=np.float32)
        ref = np.abs(rfftn(src, axes=1)).astype(np.float32)
        actual_cp = rfft_mag(src, parallel=parallel)
        assert ref.shape == actual_cp.shape
        assert np.allclose(actual_cp, ref, rtol=1e-4)

        dst = np.zeros(actual_cp.shape, dtype=np.float32)
        actual_no_cp = rfft_mag(src, output=dst, parallel=parallel)
        assert ref.shape == actual_no_cp.shape
        assert np.allclose(actual_no_cp, ref, rtol=1e-4)
        assert np.allclose(dst, ref, rtol=1e-4)


def ref_phase_correlation(xs, kernel):
    """
    Pure numpy-based implementation of phase correlation.
    """
    check_array(xs, ndim=2, dtype=np.float32)
    check_array(kernel, ndim=1, dtype=np.float32, shape_axis_0=xs.shape[1])
    kernel_padded = np.zeros_like(xs[0])
    kernel_padded[: kernel_padded.shape[0]] = kernel

    x_fft = np.fft.rfft(xs, axis=1)
    k_fft = np.fft.rfft(kernel_padded)
    m = x_fft * np.conj(k_fft)
    with np.errstate(divide="ignore", invalid="ignore"):
        m /= np.abs(m)
    m = np.nan_to_num(m)
    return np.fft.irfft(m)


def generate_desync_traces(n, width):
    kernel = np.sinc(np.linspace(-20, 20, width)).astype(np.float32)
    xs = np.random.normal(0, scale=0.01, size=(n, width)).astype(np.float32)
    sh = np.random.randint(-30, 30, size=xs.shape[0], dtype=np.int32)
    for x, s in zip(xs, sh, strict=False):
        x += np.roll(kernel, s)
    return xs, kernel, sh


@secbench_native_testcase
@pytest.mark.parametrize("parallel", (False, True))
def test_phase_correlation(parallel):
    w = 256

    for _ in range(10):
        xs, kernel, sh = generate_desync_traces(10, w)
        ref = ref_phase_correlation(xs, kernel).astype(np.float32)

        actual = phase_correlation(xs, kernel, parallel=parallel)
        assert ref.shape == actual.shape
        delta = np.abs(ref - actual)
        assert np.max(delta) < 1e-4

        data_out = np.copy(xs)
        actual_no_cp = phase_correlation(xs, kernel, output=data_out, parallel=parallel)
        assert ref.shape == actual_no_cp.shape
        delta = np.abs(ref - actual_no_cp)
        assert np.max(delta) < 1e-4


@secbench_native_testcase
@pytest.mark.parametrize("input_type", (np.int8, np.int16, np.float32))
@pytest.mark.parametrize("parallel", (False, True))
def test_fft_filter(input_type, parallel):
    data = np.ones((100, 1024)).astype(input_type)

    # Basic check.
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    d_filt = fft_filter(data, kernel, parallel=parallel)
    assert np.allclose(d_filt, 4.0)

    # Randomized test against reference model.
    for _ in range(10):
        data = 10 * np.random.normal(size=(100, 1024)).astype(np.float32)
        # Check the input can be safely converted to int8 in the worst case.
        assert np.all(np.logical_and(data > -128, data <= 127))
        data = data.astype(input_type)
        kernel = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        d_filt = fft_filter(data, kernel, parallel=parallel)
        # Filtering does not behave the same at the boundaries, ignore them.
        d_filt_ref = signal.lfilter(kernel, 1, data, axis=1)
        delta_max = np.abs(d_filt_ref[:, 3:] - d_filt[:, 3:])
        assert np.max(delta_max) < 1e-4

        # Test in-place mode.
        data_out = np.zeros_like(data, dtype=np.float32)
        fft_filter(data, kernel, output=data_out, parallel=parallel)
        assert np.allclose(d_filt, data_out)


@secbench_native_testcase
@pytest.mark.parametrize("input_type", (np.int8, np.int16, np.float32))
@pytest.mark.parametrize("parallel", (False, True))
def test_fft_filter_two_pass(input_type, parallel):
    # Basic check.
    data = np.ones((100, 1024)).astype(input_type)
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    d_filt = fft_filter(data, kernel, parallel=parallel, two_pass=True)
    assert np.allclose(d_filt, 16.0)

    for _ in range(10):
        data = 10 * np.random.normal(size=(100, 1024)).astype(np.float32)
        # Check the input can be safely converted to int8 in the worst case.
        assert np.all(np.logical_and(data > -128, data <= 127))
        data = data.astype(input_type)

        kernel = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        d_filt = fft_filter(data, kernel, parallel=parallel, two_pass=True)
        # Filtering does not behave the same at the boundaries, ignore them.
        d_filt_ref = signal.filtfilt(kernel, 1, data, axis=1)
        delta_max = np.abs(d_filt_ref[:, 3:-3] - d_filt[:, 3:-3])
        assert np.max(delta_max) < 1e-3

    data_out = np.zeros_like(data, dtype=np.float32)
    fft_filter(data, kernel, output=data_out, parallel=parallel, two_pass=True)
    assert np.allclose(d_filt, data_out)


@secbench_native_testcase
@pytest.mark.parametrize("input_type", (np.int8, np.int16, np.float32))
@pytest.mark.parametrize("parallel", (False, True))
def test_moving_sum(input_type, parallel):
    x = np.arange(32, dtype=input_type)
    y = moving_sum(x, window_size=3, parallel=parallel)
    assert y.shape == x.shape
    np.testing.assert_equal(y[:7], np.array([3, 6, 9, 12, 15, 18, 21]))


def cond_mean_var_model(data, labels, num_classes):
    m = np.zeros_like(data, shape=(num_classes, data.shape[1]))
    v = np.zeros_like(data, shape=(num_classes, data.shape[1]))
    for i in range(num_classes):
        in_class_i = labels == i
        if np.count_nonzero(in_class_i) > 0:
            m[i] = np.mean(data[in_class_i], axis=0)
            v[i] = np.var(data[in_class_i], axis=0)
    return m, v


@secbench_native_testcase
@pytest.mark.parametrize("num_classes", [2, 8, 13, 256])
@pytest.mark.parametrize("chunk_size", [0, 4, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cond_mean_var(num_classes, dtype, chunk_size):
    for _ in range(10):
        data = 20 * np.random.random((1000, 100))
        data = data.astype(dtype)
        labels = np.random.randint(0, num_classes, size=data.shape[0], dtype=np.uint16)
        accum = CondMeanVar(1, data.shape[1], num_classes)
        if chunk_size:
            accum = accum.split(chunk_size)
        accum.process_block(data, labels)
        m, v = accum.freeze()
        assert m.shape == (1, num_classes, data.shape[1])
        assert v.shape == m.shape
        m_expected, v_expected = cond_mean_var_model(data, labels, num_classes)
        assert np.allclose(m_expected, m[0], rtol=1e-4)
        assert np.allclose(v_expected, v[0], rtol=1e-4)

        global_m, global_v, global_s = accum.freeze_global_mean_var()
        assert global_s == 1000
        assert np.allclose(global_m, np.mean(data, axis=0), rtol=2e-3)
        assert np.allclose(global_v, np.var(data, axis=0, ddof=1), rtol=2e-3)


@secbench_native_testcase
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.float32, np.float64])
def test_cond_mean_var_save(tmp_path, dtype):
    num_classes = 256
    data = 20 * np.random.random((1000, 10))
    data = data.astype(dtype)
    labels = np.random.randint(0, num_classes, size=data.shape[0], dtype=np.uint16)
    accum_0 = CondMeanVar(1, data.shape[1], num_classes)
    accum_0.process_block(data, labels)
    m_0, v_0 = accum_0.freeze()

    save_path = str(tmp_path / "out.hdf5")
    accum_1 = CondMeanVar(1, data.shape[1], num_classes)
    accum_1.process_block(data[:500], labels[:500])
    accum_1.save(save_path)

    accum_2 = CondMeanVar.from_file(save_path)
    accum_2.process_block(data[500:], labels[500:])
    m_1, v_1 = accum_2.freeze()
    assert np.allclose(m_0, m_1)
    assert np.allclose(v_0, v_1)