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

import logging
from typing import Optional, Union

from typing_extensions import Annotated, Literal, TypeAlias

try:
    import numba
except ImportError:
    numba = None
import numpy as np
import numpy.typing as npt

_HW8_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

logger = logging.getLogger(__name__)

UnsignedScalar: TypeAlias = Union[np.uint8, np.uint16, np.uint32, np.uint64]
FloatScalar: TypeAlias = Union[np.float32, np.float64]
ScaScalar: TypeAlias = Union[np.int8, np.int16, np.float32]
ScaArray1D: TypeAlias = Annotated[npt.NDArray[ScaScalar], Literal["_"]]
ScaArray2D: TypeAlias = Annotated[npt.NDArray[ScaScalar], Literal["_", "_"]]
ScaArray: TypeAlias = npt.NDArray[ScaScalar]


class InvalidInputError(Exception):
    """
    In :py:mod:`secbench.processing` passing correct arguments
    (shape, dtype, etc) to various function can be tricky.

    The functions in the library will usually raise this error
    if an input is unsupported or invalid.
    """

    def __init__(self, message: str):
        super().__init__(message)


class MissingPackageError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def check_array(
    xs,
    dtype=None,
    ndim=None,
    shape_axis_0=None,
    shape_axis_1=None,
    array_name: str = "",
    check_c_continuous=False,
):
    array_label = "" if not array_name else f" for input '{array_name}'"
    if not isinstance(xs, np.ndarray):
        raise InvalidInputError(
            f"expecting a numpy array{array_label}, input has type {type(xs)}"
        )

    if check_c_continuous and not xs.data.c_contiguous:
        raise InvalidInputError(
            "this function requires a C-continous array as input (slices cannot be passed)."
        )

    if dtype is not None:
        if isinstance(dtype, (tuple, list)):
            if xs.dtype not in dtype:
                raise InvalidInputError(
                    f"invalid dtype{array_label}, supported types are {dtype}, actual type is {xs.dtype}"
                )
        elif xs.dtype != dtype:
            raise InvalidInputError(
                f"invalid dtype{array_label}, expecting {dtype}, actual type is {xs.dtype}"
            )

    if ndim is not None:
        if isinstance(ndim, (tuple, list)):
            if xs.ndim not in ndim:
                raise InvalidInputError(
                    f"invalid number of dimensions{array_label}, should be one of {ndim}, input has {xs.ndim} dimensions"
                )
        elif xs.ndim != ndim:
            raise InvalidInputError(
                f"invalid number of dimensions{array_label}, expecting {ndim} dimension, input has {xs.ndim} dimensions"
            )

    if shape_axis_0 is not None and xs.shape[0] != shape_axis_0:
        raise InvalidInputError(
            f"invalid shape for axis 0{array_label}, expecting {shape_axis_0}, actual value is {xs.shape[0]}"
        )
    if shape_axis_1 and xs.shape[1] != shape_axis_1:
        raise InvalidInputError(
            f"invalid shape for axis 1{array_label}, expecting {shape_axis_1}, actual value is {xs.shape[1]}"
        )


class ChunkIterator:
    def __init__(self, w: int, *args):
        assert w > 0
        self._args = args
        self._w = w
        sizes = [x.shape[0] for x in args]
        assert len(set(sizes)) <= 1
        self._max_size = 0 if len(sizes) == 0 else sizes[0]

    def __len__(self):
        return (self._max_size + self._w - 1) // self._w

    def __iter__(self):
        for s in range(0, self._max_size, self._w):
            e = min(s + self._w, self._max_size)
            chunks = tuple(x[s:e] for x in self._args)
            yield chunks


def chunks(w: int, *args):
    """
    Iterate several arrays with a fixed slice size on axis 0.

    This is super helpful to decompose computations in smaller blocks (e.g.,
    when the dataset cannot fit in memory). It is worth mentioning that this
    iterator is "tqdm" friendly!

    :Examples:

    >>> xs = np.arange(10)
    >>> ys = xs * xs
    >>> for x, y in ChunkIterator(3, xs, ys):
    ...     print(x, y)
    [0 1 2] [0 1 4]
    [3 4 5] [ 9 16 25]
    [6 7 8] [36 49 64]
    [9] [81]
    """
    return ChunkIterator(w, *args)


if numba is not None:

    @numba.njit
    def _add_remove_inner(dst, expanded, xs, to_add, to_remove):
        for i in range(xs.shape[0]):
            head = 0
            for j in range(xs.shape[1]):
                expanded[j + head] = xs[i, j]
                while head < to_add.shape[1] and j == to_add[i, head]:
                    expanded[j + head] = (xs[i, j] + xs[i, j + 1]) / 2
                    head += 1
            head = 0
            for j in range(dst.shape[1]):
                while head < to_remove.shape[1] and j == to_remove[i, head]:
                    head += 1
                dst[i, j] = expanded[j + head]


def add_remove(xs, ratio=0.1):
    """
    Implement the "add-remove" algorithm to simulate jitter in traces.

    :param xs: input array of shape (n_samples, n_features) to be modified.
    :returns: an array with the same shape and type as xs, but with the
        add remove transformation applied.
    """
    if numba is None:
        raise MissingPackageError(
            "the 'add_remove' function requires numba to be installed"
        )
    n_corrupt = int(ratio * xs.shape[1])

    dst = np.zeros_like(xs)

    # Internal buffer for expanding stuff
    expanded = np.zeros(xs.shape[1] + n_corrupt, dtype=xs.dtype)

    to_add = np.random.randint(0, xs.shape[1] - 1, size=(xs.shape[0], n_corrupt))
    to_add.sort(axis=1)
    to_remove = np.random.randint(0, expanded.shape[0], size=(xs.shape[0], n_corrupt))
    to_remove.sort(axis=1)

    _add_remove_inner(dst, expanded, xs, to_add, to_remove)
    return dst


def rank_of(scores, k: int, randomize=True) -> int:
    """
    Compute the rank of a given hypothesis in an array of score.

    Note that this function is randomized so that if multiple hypothesis have the
    same score, the score returned will not be dependent on the sorting algorithm.
    This properly turns to be important for evaluating guessing entropy.

    If you do not want this behavior, pass ``randomize=False`` to this function.

    :param scores: An 1-D array of scores (integers or float)
    :param k: the index of the key on which to compute the rank.
    :param randomize: randomize indices that have the same rank.
    """
    if not randomize:
        s_k = np.argsort(scores)[::-1]
        return int(np.argwhere(s_k == k).flatten()[0])

    key_score = scores[k]
    to_permute = np.argwhere(scores == key_score).flatten()
    perm = to_permute.copy()
    np.random.shuffle(perm)

    full_perm = np.arange(scores.shape[0], dtype=np.int32)
    full_perm[to_permute] = perm

    idx = np.argsort(scores, kind="stable")[::-1]
    idx = full_perm[idx]
    r = np.argwhere(idx == k).flatten()[0]
    return int(r)


def key_scores(
    pred_lg: np.ndarray, target_variable_fn, secret_values, *args, **kwargs
) -> np.ndarray:
    """
    Compute the log Maximum likelihood of each key hypothesis.

    This function is typically used at the end of side-channel attacks to score key hypothesis.

    .. note::

        This function operates on logarithms to avoid numerical stability
        issue (when working with probabilities).

    :param pred_lg: logarithm of predictions obtained on some traces (the expected
        format for this data is the same as Scikit-learn's ``model.predict_proba()``
        methods). This array has a shape (n_traces, n_classes).
    :param target_variable_fn: how to compute target variables under a key hypothesis.
        The first argument is a value picked in ``secret_values`` array, the `args`
        and `kwargs` are then forwarded.
    :param args: arbitrary arguments forwarded to `target_variable_fn`.
    :param kwargs: arbitrary keyword arguments forwarded to `target_variable_fn`.
    :param secret_values: Any iterator that returns secret hypothesis to be tested.

    :return: the score of each key. This array has shape (n_classes,).
        Where ``n_classes`` is the size of the ``secret_values`` iterator.
    """
    scores = []
    for k_hyp in secret_values:
        # Compute the target variables for each trace
        y = target_variable_fn(k_hyp, *args, **kwargs)

        # Sum the prediction likelihood for each trace
        score = sum([r[yi] for r, yi in zip(pred_lg, y, strict=False)])

        scores.append(score)
    return np.array(scores)


def encode_labels(ys: np.ndarray, dtype=np.uint16, indep=True) -> np.ndarray:
    """
    Encode labels in the range [0; n_classes)

    This function is a wrapper around the `LabelEncoder` from sklearn.

    :param dtype: type for output labels.
    :param indep: should the labels share the same support? This option is only valid when the input has two dimensions.
        When set to `True` (the default) all columns of ys are encoded independently.

    .. versionadded: 2.7.0
    """
    from sklearn.preprocessing import LabelEncoder

    if ys.ndim == 1:
        return LabelEncoder().fit_transform(ys).astype(dtype)
    elif ys.ndim == 2:
        if indep:
            cols = []
            for col in range(ys.shape[1]):
                cols.append(LabelEncoder().fit_transform(ys[:, col]))
            return np.array(cols, dtype=dtype).T
        else:
            encoder = LabelEncoder().fit(ys.flatten())
            cols = []
            for col in range(ys.shape[1]):
                cols.append(encoder.transform(ys[:, col]))
            return np.array(cols, dtype=dtype).T
    else:
        raise ValueError(
            f"unsupported input dimension {ys.shape}, ys should have 1 or 2 dimensions"
        )


def qplot(
    x,
    y=None,
    n=20,
    percentile_min=5,
    percentile_max=95,
    color: str = "r",
    plot_mean: bool = False,
    plot_median: bool = False,
    line_color="k",
    ax=None,
    **kwargs,
):
    """
    Generate a quantile plot for a given dataset.

    The ``qplot`` function visualizes the distribution of data by plotting
    percentiles and optionally includes the mean and median lines.

    :param x: If y is not given, should be a 2-dimensional Numpy array of
        shape ``(n_traces, n_features)`` which represents the data to be
        plot. An X-axis will be generated. If ``y`` is provided, then this
        input is a 1-D Numpy array which contains the X range of the plot.
    :param y: a 2-dimensional Numpy array of shape ``(n_traces, n_features)``.
        Represents the data to be plot. X axis is given by argument ``x``.
    :param n: The number of percentile groups to calculate. The higher this
        value the smoother the plot, but the slower it is to render.
    :param percentile_min: The minimum percentile to start the calculation.
    :param percentile_max: The maximum percentile to end the calculation.
    :param color: The color of the filled percentile areas.
    :param plot_mean: Whether to plot the mean line.
    :param plot_median: Whether to plot the median line.
    :param line_color: The color of the mean and median lines.
    :param ax: The axis on which to plot. If not provided, the current active axis is used.
    :param kwargs: Additional keyword arguments passed to matplotlib's ``fill_between``.

    """
    if y is None:
        y = x
    x = np.arange(y.shape[-1])

    perc1 = np.percentile(
        y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0
    )
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / n
    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    for p1, p2 in zip(perc1, perc2, strict=False):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    if plot_mean:
        ax.plot(x, np.mean(y, axis=0), color=line_color)

    if plot_median:
        ax.plot(x, np.median(y, axis=0), color=line_color)

    return ax


def has_tensorflow() -> bool:
    """
    Check if tensorflow is properly working.
    """
    try:
        import tensorflow
        import tensorflow.keras

        # Fake use of tensorflow for flake8
        return tensorflow.version.VERSION is not None
    except ImportError:
        return False


def tensorflow_init_gpu(memory_limit_mb: Optional[int] = None):
    """
    Initialize tensorflow for GPU usage.

    This function will create a logical device with limited memory.
    It turns out to be the only way to correctly limit GPU memory.
    This function is especially useful in SLURM jobs.
    """
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    try:
        if memory_limit_mb is not None:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )
                    ],
                )
        logical_gpus = tf.config.list_logical_devices("GPU")
        logger.debug(f"{len(gpus)}, Physical GPU, {len(logical_gpus)} Logical GPUs")
    except RuntimeError:
        logger.warning("something went wrong during tensorflow initialization")