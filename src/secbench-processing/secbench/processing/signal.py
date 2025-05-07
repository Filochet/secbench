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
Digital signal processing utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

from ._native import secbench_native_import, transform_2d
from .helpers import InvalidInputError, ScaArray, ScaArray1D, check_array


def generate_lp_firls(f_low: float, f_high: float, fs: float, numtaps: int = 201):
    """
    Build FIR filter coefficients using the least-square approach.

    You need to specify f_low, and f_high, which define respectively the low
    and high frequency where the frequency response will drop.
    """
    desired = (1, 1, 0, 0)
    bands = (0, f_low, f_high, 0.5 * fs)
    return signal.firls(numtaps, bands, desired, fs=fs)


def plot_filter_response(axs, taps, fs, sos=False):
    """
    Plot the frequency and phase response of a filter.

    :Example:

    .. code-block:: python

        fs = 1e6
        taps = generate_lp_firls(1e3, 5e3, fs)

        fig, axs = plt.subplots(1, 2)
        fig.tight_layout()
        plot_filter_response(axs, taps, fs)

    """
    freq, response = signal.sosfreqz(taps) if sos else signal.freqz(taps)
    axs[0].semilogy(0.5 * fs * freq / np.pi, np.abs(response))
    axs[1].plot(0.5 * fs * freq / np.pi, np.angle(response))


def plot_fft(ax, x, fs):
    """
    Plot the discrete Fourier transform of a signal.

    .. code-block:: python

        fs = 1e6
        xs = np.linspace(0, 1, int(fs))
        ref = np.sin(2 * np.pi * xs * 1000 + 0.5)
        plot_fft(fs, ref)

    """
    num_sample = x.shape[0]
    period = 1.0 / fs
    yf = fft(x)
    xf = fftfreq(num_sample, period)[: num_sample // 2]
    ax.plot(xf, 2.0 / num_sample * np.abs(yf[0 : num_sample // 2]))
    ax.grid()


def spectrogram(ax, x, fs, nperseg=1024, noverlap=None, vmin=None):
    """
    Sliding window fourier transform.

    :Example:

    .. code-block:: python

        fs = 1e9
        _fig, _ax = plt.subplots()
        spectrogram(_ax, np.random.random(size=5000), fs, nperseg=512)
        plt.show()

    """
    import matplotlib.pyplot as plt

    noverlap = nperseg - 4 if noverlap is None else noverlap
    f, t, spec = signal.spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, window="hann"
    )
    spec_db = 10 * np.log10(spec)
    vmax = spec_db.max()
    cmap = plt.cm.coolwarm
    return ax.pcolormesh(
        t, f, spec_db, vmin=vmin, vmax=vmax, cmap=cmap, shading="gouraud"
    )


def downsample(
    X: ScaArray, samples_out: Optional[int] = None, factor: Optional[float] = None
) -> tuple:
    """
    Downsample data using the Largest-Triangle-Three-Buckets algorithm.

    It is possible to downsample an array either into `sample_out` samples, or by a factor.

    :param X: A numpy array of shape ``(n_samples, n_features)`` or ``(n_features, )``
    :param samples_out: Maximum length of the output trace.
    :param factor: Decimation factor (e.g., take one sample over factor samples)
    :return (x_s, y_s): A tuple containing the x and y coordinates of the downsampled trace(s)

    """
    import lttbc as lttb

    data_len = X.shape[-1]

    # Parameters check
    if samples_out is not None and factor is not None:
        raise ValueError("Must provide either samples_out or factor")

    if samples_out is None:
        # Compute samples_out from factor
        samples_out = int(data_len / factor)
    if samples_out <= 0:
        raise ValueError(f"invalid samples_out {samples_out}, must be positive.")

    t = np.arange(data_len)
    # Proceed with downsampling
    if X.ndim == 1:
        x_s, y_s = lttb.downsample(t, X, samples_out)
    elif X.ndim == 2:
        x_s = []
        y_s = []
        for trace in X:
            x_1d, y_1d = lttb.downsample(t, trace, samples_out)
            x_s.append(x_1d)
            y_s.append(y_1d)
        x_s, y_s = np.array(x_s), np.array(y_s)
    else:
        raise ValueError("Dimension must be 1 or 2")
    return x_s, y_s


_moving_sum_dispatch = {
    "int8": secbench_native_import("moving_sum_i8"),
    "int16": secbench_native_import("moving_sum_i16"),
    "float32": secbench_native_import("moving_sum_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def moving_sum(
    X: ScaArray,
    *,
    window_size: int,
    scale: float = 1.0,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
):
    """
    Compute a windowed sum of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param scale: rescaling factor to apply on the moving sum.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    fn = _moving_sum_dispatch[X.dtype.name]
    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        scale=scale,
    )


_fft_filter = {
    "int8": secbench_native_import("fft_filter_i8"),
    "int16": secbench_native_import("fft_filter_i16"),
    "float32": secbench_native_import("fft_filter_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def fft_filter(
    X: ScaArray,
    kernel: ScaArray1D,
    *,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
    two_pass=False,
):
    """
    Filter a given signal using FFT method.

    Similar functionality is provided by ``scipy.signal.lfilter`` or
    ``scipy.signal.filtfilt``. However, in comparison, this method has a
    low memory usage, allows working in-place and supports fine control over
    parallelism.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param kernel: a numpy array of shape ``(n_coeffs,)`` and dtype ``np.float32``.
        The kernel must be smaller that the number of features in the input.
    :param two_pass: if ``True``, performs left filtering pass then right
        filtering pass. This provides functionality similar to
        ``scipy.signal.filtfilt``.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    check_array(kernel, dtype=np.float32, ndim=1)
    if kernel.shape[0] > X.shape[1]:
        raise InvalidInputError(
            f"kernel has shape {kernel.shape}, which is larger than input traces (shape: {X.shape}), filter cannot be applied"
        )
    fn = _fft_filter[X.dtype.name]
    return fn(
        output,
        X,
        kernel,
        parallel=parallel,
        chunk_size=chunk_size,
        two_pass=two_pass,
    )


_phase_correlation = {
    "int8": secbench_native_import("phase_correlation_i8"),
    "int16": secbench_native_import("phase_correlation_i16"),
    "float32": secbench_native_import("phase_correlation_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def phase_correlation(
    X: ScaArray,
    kernel: ScaArray1D,
    *,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
):
    """
    Compute phase correlation between an input signal and a kernel.

    :param X: a numpy array of shape `(n_samples, n_features)` or `(n_features,)`.
    :param kernel: a numpy array of shape `(n_coeffs,)` and dtype `np.float32`.
        The kernel must be smaller that the number of features in the input.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    check_array(kernel, dtype=np.float32, ndim=1)
    if kernel.shape[0] > X.shape[1]:
        raise InvalidInputError(
            f"kernel has shape {kernel.shape}, which is larger than input traces (shape: {X.shape}), filter cannot be applied"
        )
    fn = _phase_correlation[X.dtype.name]
    return fn(
        output,
        X,
        kernel,
        parallel=parallel,
        chunk_size=chunk_size,
    )


_rfft_mag = {
    "int8": secbench_native_import("rfft_mag_i8"),
    "int16": secbench_native_import("rfft_mag_i16"),
    "float32": secbench_native_import("rfft_mag_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def rfft_mag(
    X: ScaArray,
    *,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
):
    """
    Magnitude of the real Fourier transform of the signal.

    :param X: a numpy array of shape `(n_samples, n_features)` or `(n_features,)`.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    fn = _rfft_mag[X.dtype.name]
    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
    )


_match_euclidean = {
    "int8": secbench_native_import("match_euclidean_i8"),
    "int16": secbench_native_import("match_euclidean_i16"),
    "float32": secbench_native_import("match_euclidean_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def match_euclidean(
    X: ScaArray,
    kernel: ScaArray1D,
    *,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
):
    """
    Match a given kernel using Euclidean distance.

    Best match is found at the minimum.

    .. note::

        This function returns the square of the Euclidean distance
        with the pattern, since taking the square root is a waste
        of time for SCA alignment. You can call manually apply
        ``np.sqrt`` on the result if needed.

    :param X: a numpy array of shape `(n_samples, n_features)` or `(n_features,)`.
    :param kernel: kernel to be matched in the traces.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    check_array(kernel, dtype=np.float32, ndim=1)
    if kernel.shape[0] > X.shape[1]:
        raise InvalidInputError(
            f"kernel has shape {kernel.shape}, which is larger than input traces (shape: {X.shape}), filter cannot be applied"
        )
    fn = _match_euclidean[X.dtype.name]
    return fn(
        output,
        X,
        kernel,
        parallel=parallel,
        chunk_size=chunk_size,
    )


_match_correlation = {
    "int8": secbench_native_import("match_correlation_i8"),
    "int16": secbench_native_import("match_correlation_i16"),
    "float32": secbench_native_import("match_correlation_f32"),
}


@transform_2d(input_types=(np.int8, np.int16, np.float32), output_types=(np.float32,))
def match_correlation(
    X,
    kernel,
    *,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=np.float32,
):
    """
    Match a given kernel using normalized cross-correlation.

    Best match is found at the maximum.

    :param X: a numpy array of shape `(n_samples, n_features)` or `(n_features,)`.
    :param kernel: kernel to be matched in the traces.

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """
    check_array(kernel, dtype=np.float32, ndim=1)
    if kernel.shape[0] > X.shape[1]:
        raise InvalidInputError(
            f"kernel has shape {kernel.shape}, which is larger than input traces (shape: {X.shape}), filter cannot be applied"
        )
    fn = _match_correlation[X.dtype.name]
    return fn(
        output,
        X,
        kernel,
        parallel=parallel,
        chunk_size=chunk_size,
    )


types_tuple = [
    (np.float32, np.int8),
    (np.float32, np.int16),
    (np.float32, np.float32),
    (np.float64, np.float64),
]

types_rust_conv = {
    np.float32.__name__: "f32",
    np.float64.__name__: "f64",
    np.int8.__name__: "i8",
    np.int16.__name__: "i16",
}


def generate_tuple_fn_dict(types_tuple, fn_name: str):
    return {
        (j.__name__, k.__name__): secbench_native_import(
            f"{fn_name}_{types_rust_conv[j.__name__]}_{types_rust_conv[k.__name__]}"
        )
        for j, k in types_tuple
    }


_sliding_mean = generate_tuple_fn_dict(types_tuple, "sliding_mean")


@transform_2d(
    input_types=(np.int8, np.int16, np.float32, np.float64),
    output_types=(np.float32, np.float64),
)
def sliding_mean(
    X: ScaArray,
    *,
    window_size: int,
    padding_value=None,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=None,
):
    """
    Compute a sliding mean of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param padding_value: value used for padding initial samples (0 if ``None``).

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """

    try:
        fn = _sliding_mean[(dtype.__name__, X.dtype.name)]  # type: ignore
    except KeyError as err:
        raise TypeError(f"invalid input type for X: {dtype}") from err

    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        padding_value=padding_value,
    )


_sliding_var = generate_tuple_fn_dict(types_tuple, "sliding_var")


@transform_2d(
    input_types=(np.int8, np.int16, np.float32, np.float64),
    output_types=(np.float32, np.float64),
)
def sliding_var(
    X: ScaArray,
    *,
    window_size: int,
    padding_value=None,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=None,
):
    """
    Compute a sliding variation of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param padding_value: value used for padding initial samples (0 if ``None``).

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """

    try:
        fn = _sliding_var[(dtype.__name__, X.dtype.name)]  # type: ignore
    except KeyError as err:
        raise TypeError(f"invalid input type for X: {dtype}") from err

    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        padding_value=padding_value,
    )


_sliding_std = generate_tuple_fn_dict(types_tuple, "sliding_std")


@transform_2d(
    input_types=(np.int8, np.int16, np.float32, np.float64),
    output_types=(np.float32, np.float64),
)
def sliding_std(
    X: ScaArray,
    *,
    window_size: int,
    padding_value=None,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=None,
):
    """
    Compute a sliding standard deviation of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param padding_value: value used for padding initial samples (0 if ``None``).

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """

    try:
        fn = _sliding_std[(dtype.__name__, X.dtype.name)]  # type: ignore
    except KeyError as err:
        raise TypeError(f"invalid input type for X: {dtype}") from err

    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        padding_value=padding_value,
    )


_sliding_skew = generate_tuple_fn_dict(types_tuple, "sliding_skew")


@transform_2d(
    input_types=(np.int8, np.int16, np.float32, np.float64),
    output_types=(np.float32, np.float64),
)
def sliding_skew(
    X: ScaArray,
    *,
    window_size: int,
    padding_value=None,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=None,
):
    """
    Compute a sliding skewness of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param padding_value: value used for padding initial samples (0 if ``None``).

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """

    try:
        fn = _sliding_skew[(dtype.__name__, X.dtype.name)]  # type: ignore
    except KeyError as err:
        raise TypeError(f"invalid input type for X: {dtype}") from err

    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        padding_value=padding_value,
    )


_sliding_kurt = generate_tuple_fn_dict(types_tuple, "sliding_kurt")


@transform_2d(
    input_types=(np.int8, np.int16, np.float32, np.float64),
    output_types=(np.float32, np.float64),
)
def sliding_kurt(
    X: ScaArray,
    *,
    window_size: int,
    padding_value=None,
    output=None,
    parallel=False,
    chunk_size: int | None = None,
    dtype=None,
):
    """
    Compute a sliding kurtosis of the input array.

    :param X: a numpy array of shape ``(n_samples, n_features)`` or ``(n_features,)``.
    :param window_size: size of the window (i.e., number of samples added).
    :param padding_value: value used for padding initial samples (0 if ``None``).

    :param output: if given, compute the result in this array. Otherwise,
        an output array will be allocated.
    :param parallel: if ``True``, processes groups of ``chunk_size`` rows
        of ``X`` in parallel. The number of threads is defined by
        environment variable `RAYON_NUM_THREADS`. Otherwise, processing is
        done sequentially.
    :param chunk_size: number of rows of ``X`` processed in parallel.
    :param dtype: output type (only ``np.float32`` is exposed currently).
    """

    try:
        fn = _sliding_kurt[(dtype.__name__, X.dtype.name)]  # type: ignore
    except KeyError as err:
        raise TypeError(f"invalid input type for X: {dtype}") from err

    return fn(
        output,
        X,
        parallel=parallel,
        chunk_size=chunk_size,
        window_size=window_size,
        padding_value=padding_value,
    )