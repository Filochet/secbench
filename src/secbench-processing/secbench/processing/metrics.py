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

"""
Side-channel metrics for leakages assessment and feature extraction.
"""

import abc
import functools
import itertools
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy
from scipy.stats import norm
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

from ._native import secbench_native_import
from .helpers import check_array
from .helpers import encode_labels as _encode_labels
from .models import lra_unpackbits

logger = logging.getLogger(__name__)


def vpearson(X, y):
    """
    Vectorized version of pearson correlation coefficient.

    Please consider using the :py:func:`vpearson_fast` instead which you be
    order of magnitude faster.

    This function correlates each column of y (usually the hypothesis)
    with each column of X (usually the traces).

    :param X: Input samples. As an array of shape (n_samples, n_features).
    :param y: Target values. As an array of shape (n_samples, n_targets).

    :returns: an array of shape (n_targets, n_features) containing the pearson
        correlation of each feature.

    .. warning::

        Make sure you call this function with float arrays. Otherwise, it
        is quite slow...

    .. versionchanged:: 1.3.0

        Order or parameters was inverted.

    :Example:

    Here is how you can use this function:

    .. code-block:: python

        ys = hamming_weight(aes_sbox(pts ^ np.tile(k, (n_traces, 1))))
        r = vpearson(ys.astype(np.float), data.astype(np.float))
        plt.plot(r.T)
        plt.show()

    """
    n = y.shape[0]
    if n != X.shape[0]:
        raise ValueError("x and y must have the same number of timepoints.")
    mu_y = y.mean(axis=0)
    mu_x = X.mean(axis=0)
    s_x = y.std(axis=0, ddof=n - 1)
    s_y = X.std(axis=0, ddof=n - 1)
    cov = np.dot(y.T, X) - n * np.dot(mu_y[:, np.newaxis], mu_x[np.newaxis, :])
    return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])


def vpearson_fast(X, y):
    """
    Highly optimized, column-wise Pearson correlation.

    This function is much faster than :py:func:`vpearson`.

    :param X: Input samples. As an array of shape (n_samples, n_features).
    :param y: Target values. As an array of shape (n_samples, n_targets).

    :returns: an array of shape (n_targets, n_features) containing the pearson
        correlation of each feature.

    .. versionadded:: 1.3.0

    """
    n, t = X.shape  # n traces of t samples

    # compute O - mean(O)
    DO = X - (np.einsum("nt->t", X, dtype="float64", optimize="optimal") / np.double(n))
    DP = y - (np.einsum("nm->m", y, dtype="float64", optimize="optimal") / np.double(n))

    numerator = np.einsum("nm,nt->mt", DP, DO, optimize="optimal")
    tmp1 = np.einsum("nm,nm->m", DP, DP, optimize="optimal")
    tmp2 = np.einsum("nt,nt->t", DO, DO, optimize="optimal")
    tmp = np.einsum("m,t->mt", tmp1, tmp2, optimize="optimal")
    denominator = np.sqrt(tmp)

    return numerator / denominator


def lra_lsqr(X, y, use_numpy=False):
    """
    Implementation of linear regression analysis using a least square
    regression.

    :param X: An array of shape (n_samples, n_features)
    :param y: An Array of shape (n_hyp, n_samples, n_targets).
    :param use_numpy: use numpy's implementation of least squares. By default,
        uses scipy version (recommended).

    :return: A tuple (score, solutions).
        The array ``score`` is an (n_hyp, n_features) array of scores.
        The array ``solutions`` is a (n_hyp, n_targets, n_features) array,
        which contains the least square solution. This array can be used
        for predicting values.

    :Example:

    For example, this is how you would do an LRA against an AES:

    .. code-block:: python

        # 1. Generate hypothesis
        leak = # compute N leakage hypothesis

        # 2. Decompose each leakage value into bits
        leak = np.unpackbits(leak[:, np.newaxis], axis=1)
        # We need to add an intercept to catch the constant part of the signal.
        # The following code adds a column of 1.
        leak = np.c_[leak, np.ones(leak.shape[0])]

        # 3. Apply LRA
        scores, sols = lra_lsqr(data, [leak])

        # 4. Select the best key
        # WARNING: you may need a better selection function!
        best_key = np.argmax(np.max(np.abs(scores), axis=1))

    """
    scores = []
    sols = []
    ss_tot = np.sum((X - np.mean(X, axis=0)) ** 2, axis=0)
    for target in y:
        # Optional: use scipy implementation
        if use_numpy:
            r = np.linalg.lstsq(target, X, rcond=None)
        else:
            r = scipy.linalg.lstsq(target, X)
        sols.append(r[0])
        ss_res = r[1]
        if ss_res.shape == (0,):
            raise ValueError(
                "unable to find a least-square solution. This is often caused "
                "by the target variable having colinear entries"
            )
        score = 1 - ss_res / ss_tot
        scores.append(score)
    return np.array(scores), np.array(sols)


def cond_mean_metric(f):
    @functools.wraps(f)
    def wrapped(X, y, **kwargs):
        check_X_y(X, y, multi_output=True)
        if y.ndim == 1:
            return f(X, y[:, np.newaxis], **kwargs)[0]
        else:
            assert y.ndim == 2, (
                f"argument 'y' must be 2-dimensional, actual shape={y.shape}"
            )
            return f(X, y, **kwargs)

    return wrapped


@cond_mean_metric
def snr(X: np.ndarray, y: np.ndarray, encode_labels=False, num_classes=None, **kwargs):
    """
    Compute a signal-to-noise ratio.

    .. math:: snr = \\frac{Var(E(X | Y))}{E(Var(X | Y))}

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets).
    :param encode_labels: whether labels should be re-encoded
    :param num_classes: number of classes (otherwise inferred from the maximum value of labels).
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.

    .. versionchanged:: 2.6.0

        SNR implementation is based on the :func:`cond_mean_var` helper.
    """
    if encode_labels:
        y = _encode_labels(y, indep=False)

    if num_classes is None:
        num_classes = np.max(y) + 1
    accum = cond_mean_var(X, y, num_classes, **kwargs)
    return accum.freeze_snr()


@cond_mean_metric
def welch_t_test(X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute a Welch T-Test.

    .. math::

      t = \\frac{E(X | Y = y_1) - E(X | Y = y_2)}
                {Var(X | Y = y_1)/N_1 + Var(X|Y = y_2)/N_2}

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets). Target values must be in the set {0, 1} only
        for this test.
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.

    .. versionchanged:: 2.6.0

        The implementation is based on the :func:`cond_mean_var` helper.
    """
    assert np.max(y) == 1, "labels must be 0, 1 variables"
    assert np.min(y) == 0, "labels must be 0, 1 variables"
    accum = cond_mean_var(X, y, 2, **kwargs)
    return accum.freeze_welch_t_test()


@cond_mean_metric
def dom(X, y, **kwargs):
    """
    Differences of means metric.

    .. math::

      t = |E(X | Y = 0) - E(X | Y = 1)|

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets). Target values must be in the set {0, 1} only
        for this test.
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.

    .. versionchanged:: 2.6.0

        The implementation is based on the :func:`cond_mean_var` helper.
    """
    assert np.max(y) == 1, "labels must be 0, 1 variables"
    assert np.min(y) == 0, "labels must be 0, 1 variables"
    accum = cond_mean_var(X, y, 2, **kwargs)
    return accum.freeze_dom()


@cond_mean_metric
def nicv(
    X: np.ndarray, y: np.ndarray, num_classes=None, encode_labels=False, **kwargs
) -> np.ndarray:
    """
    Compute a Normalized Interclass Variance (Nicv).

    .. math:: nicv = \\frac{Var(E(X | Y))}{Var(X)}

    .. note::

        This metric is usually very similar to the SNR. Only the denominator
        of the formula differs.

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets).
    :param encode_labels: whether labels should be re-encoded
    :param num_classes: number of classes (otherwise inferred from the maximum value of labels).
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.

    .. versionchanged:: 2.6.0

        The implementation is based on the :func:`cond_mean_var` helper.
    """
    if encode_labels:
        y = _encode_labels(y, indep=False)

    if num_classes is None:
        num_classes = np.max(y) + 1
    accum = cond_mean_var(X, y, num_classes, **kwargs)
    return accum.freeze_nicv()


@cond_mean_metric
def sost(
    X: np.ndarray, y: np.ndarray, num_classes=None, encode_labels=False, **kwargs
) -> np.ndarray:
    """
    Compute the Sum of Square T differences.

    .. math::

      \\sum_{i, j, i < j}{
          \\frac{(E(X | y_i) - E(X | y_j))^2}{Var(X | y_i) + Var(X | y_j)}}

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets).
    :param encode_labels: whether labels should be re-encoded
    :param num_classes: number of classes (otherwise inferred from the maximum value of labels).
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.

    .. versionchanged:: 2.6.0

        The implementation is based on the :func:`cond_mean_var` helper.
    """
    if encode_labels:
        y = _encode_labels(y, indep=False)

    if num_classes is None:
        num_classes = np.max(y) + 1
    accum = cond_mean_var(X, y, num_classes, **kwargs)
    return accum.freeze_sost()


def _lra(X: np.ndarray, y: np.ndarray, use_numpy=True):
    """
    Linear Regression Analysis as a leakage metric.
    """
    if y.ndim == 2:
        s, c = lra_lsqr(X, [y], use_numpy=use_numpy)
        return s[0], c[0]
    else:
        assert y.ndim == 3, (
            f"input 'y' must be a 3-dimensional array, current shape={y.shape}"
        )
        return lra_lsqr(X, y, use_numpy=use_numpy)


class LRA:
    """
    An LRA metric with a specific decomposition model.

    This metric returns the R^2 score of the least square solution of:

    .. math:: X = \\beta \\cdot y + \\beta_0

    :Example:

    .. code-block:: python

        from secbench.processing import lra_unpackbits
        from secbench.processing.metrics import LRA

        metric = LRA(model=lra_unpack_bits)
        score = metric(samples, targets)

    Instances of this class are callable and have the same prototype as
    other leakage metrics (i.e., ``f(X, y)``).
    """

    def __init__(
        self,
        model=lra_unpackbits,
        use_numpy=False,
        use_cond_mean=False,
        n_classes=None,
        **kwargs,
    ):
        """
        Create an LRA metric.

        :param model: target variable decomposition model.
        :param use_numpy: select Numpy implementation of least square method.
        :param use_cond_mean: compute average of samples per class before
            doing the LRA. The resulting LRA is much faster.
        :param n_classes: number of classes (only needed when
            use_cond_mean=True).
        :param kwargs: The remaining keyword arguments are forwarded to the
            model as keywords arguments.
        """
        self.model = model
        self._model_kwargs = kwargs
        self.use_numpy = use_numpy
        self.use_cond_mean = use_cond_mean
        if use_cond_mean:
            assert n_classes is not None, "n_classes must be set to "
        self.n_classes = n_classes

    def _run(self, X, y, y_prev=None):
        check_X_y(X, y, multi_output=True)
        if np.any(y_prev):
            check_X_y(X, y_prev, multi_output=True)
            y = self.model(y, y_prev, **self._model_kwargs)
        else:
            y = self.model(y, **self._model_kwargs)
            if y.ndim == 3:
                y = np.swapaxes(y, 0, 1)

        return _lra(X, y, use_numpy=self.use_numpy)

    def _cond_mean_lra(self, X, y):
        encoder = LabelEncoder()
        y_e = encoder.fit_transform(y.flatten()).reshape(y.shape)

        Xp = cond_mean(X, y_e, num_classes=self.n_classes)
        ys = encoder.inverse_transform(np.arange(len(encoder.classes_), dtype=np.uint8))

        if y.ndim == 2:
            scores = []
            coeffs = []
            for row in Xp:
                r = self._run(row, ys)
                s, c = r
                scores.append(s)
                coeffs.append(c)
            return np.array(scores), np.array(coeffs)

        return self._run(Xp, ys)

    def scores_and_coeffs(self, X, y, y_prev=None):
        """
        Return both R^2 scores and coefficients from the LRA.

        :param X: training data. An array of shape (n_samples, n_features).
        :param y: Target values. An array of shape (n_samples,)
            or (n_samples, n_targets).
        :returns: a tuple of arrays ``(scores, coeffs)``. Where ``scores`` is an
            array of (n_features,) or (n_targets, n_features), and where ``coeffs``
            is an array of shape (n_features, n_bits) or (n_targets, n_features, n_bits).
        """
        if self.use_cond_mean:
            return self._cond_mean_lra(X, y)
        return self._run(X, y, y_prev)

    def __call__(self, X, y, y_prev=None):
        return self.scores_and_coeffs(X, y, y_prev)[0]


def pearson(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson's correlation coefficient.

    :param X: training data. An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets).
    :returns: an array of shape (n_features,) or (n_targets, n_features) of
        scores.
    """
    check_X_y(X, y, multi_output=True)
    if y.ndim == 1:
        return vpearson_fast(X, y[:, np.newaxis])[0]
    return vpearson_fast(X, y)


def identity_fn(X, _y):
    return X


_CondMeanVar = secbench_native_import("CondMeanVar")


class CondMeanVarBase(abc.ABC):
    @abc.abstractmethod
    def _impl(self):
        pass

    @abc.abstractmethod
    def freeze(self):
        pass

    @abc.abstractmethod
    def freeze_global_mean_var(self):
        pass

    def process_block(self, X, y):
        """
        Add new data in the accumulator

        :param X: an array of shape ``(n_samples, n_features)`` containing data.
        :param y: an array of shape ``(n_samples, n_targets)`` containing the labels.
        """
        check_array(X, shape_axis_0=y.shape[0], ndim=2)
        check_array(y, shape_axis_0=X.shape[0])
        if y.ndim == 1:
            y = y[:, np.newaxis]
        if not isinstance(y, np.uint16):
            y = y.astype(np.uint16)

        impl = self._impl()
        if X.dtype == np.int8:
            impl.process_block_i8(X, y)
        elif X.dtype == np.int16:
            impl.process_block_i16(X, y)
        elif X.dtype == np.float32:
            impl.process_block_f32(X, y)
        elif X.dtype == np.float64:
            impl.process_block_f64(X, y)
        else:
            raise NotImplementedError(f"unsupported array dtype: {X.dtype}")

    def freeze_snr(self):
        """
        Compute a signal-to-noise ratio for the current accumulator.
        """
        mean, var = self.freeze()
        return np.var(mean, axis=1) / np.mean(var, axis=1)

    def freeze_nicv(self):
        """
        Compute the normalized interclass variance for the current accumulator.
        """
        mean, var = self.freeze()
        _, global_var, _ = self.freeze_global_mean_var()
        return np.var(mean, axis=1) / global_var

    def freeze_welch_t_test(self):
        """
        Compute Welch's T-Test for the current accumulator.

        The accumulator must have two classes.
        """
        m, v = self.freeze()
        s = self._impl().freeze_samples_per_class()
        mx, vx, nx = m[:, 0], v[:, 0], s[:, 0]
        my, vy, ny = m[:, 1], v[:, 1], s[:, 1]
        return (mx - my) / np.sqrt((vx / nx + vy / ny))

    def freeze_dom(self):
        """
        Compute the difference of means.

        The accumulator must have two classes.
        """
        m, _v = self.freeze()
        return m[:, 0] - m[:, 1]

    def freeze_sost(self):
        """
        Compute the sum of square T differences.
        """
        m, v = self.freeze()
        res = np.zeros((m.shape[0], m.shape[2]), dtype=m.dtype)
        for i, j in itertools.combinations(range(m.shape[1]), 2):
            num = m[:, i] - m[:, j]
            res += (num * num) / (v[:, i] + v[:, j])
        return res


class CondMeanVar(CondMeanVarBase):
    """
    Optimized implementation of conditional mean and variance.

    .. versionadded: 2.5.0
    """

    def __init__(self, targets: int, samples: int, num_classes: int):
        """
        Create an empty accumulator.

        :param targets: Number of target variables
        :param samples: Number of samples per traces
        :param num_classes: Number of classes for target variables.
        """
        self._obj = _CondMeanVar(targets, samples, num_classes)

    def _impl(self):
        return self._obj

    @classmethod
    def from_file(cls, path, prefix=""):
        """
        Load a :py:class:`CondMeanVar` instance from a HDF5 snapshot.
        """
        from h5py import File

        file = File(path, mode="r")
        if prefix:
            prefix = prefix + "/"
        m = file[f"{prefix}mean"][:]
        v = file[f"{prefix}var"][:]
        s = file[f"{prefix}samples"][:]
        assert m.shape == v.shape
        assert s.shape[0] == m.shape[0]
        assert s.shape[1] == m.shape[1]
        obj = cls(m.shape[0], m.shape[2], m.shape[1])
        obj._obj.load(m, v, s)
        return obj

    def save(self, path: str, prefix=""):
        """
        Create a snapshot of the current accumulator in a HDF5 file.

        This snapshot can be reloaded with :py:meth:`CondMeanVar.from_file`.
        """
        from h5py import File

        m, v, s = self._obj.save()
        file = File(path, mode="w")
        if prefix:
            prefix = prefix + "/"
        file[f"{prefix}mean"] = m
        file[f"{prefix}var"] = v
        file[f"{prefix}samples"] = s

    def freeze(self):
        """
        Return the current mean and variance per class.

        :return:
            a tuple of arrays ``(mean, variance)``, both arrays have
            shape ``(n_targets, n_classes, n_features)``.
        """
        return self._obj.freeze_mean_var()

    def freeze_global_mean_var(self):
        """
        Return the mean and variance of the data accumulated so far.

        :return: a tuple `(mean, var, samples)`, where mean and variance are
         1-D arrays with the same number of samples than the input data.
        """
        return self._obj.freeze_global_mean_var()

    def split(self, chunk_size: int):
        """
        Turn the object in a parallel instance (:py:class:`CondMeanVarP`).

        The latter can be converted back to a normal accumulator
        with :py:meth:`CondMeanVarP.merge`.

        :param chunk_size:
            The number of samples processed by each thread.
            Smaller chunks lead to higher parallelism, but might decrease
            performances. As a rule of thumb, use something between 2-8 cache
            lines (e.g., chunk_size = 256 for 8 bit data).
        :return:
        """
        assert chunk_size > 0
        raw = self._obj.split(chunk_size)
        return CondMeanVarP(raw)


class CondMeanVarP(CondMeanVarBase):
    def __init__(self, raw):
        self._obj = raw

    def _impl(self):
        return self._obj

    def merge(self) -> CondMeanVar:
        raw = CondMeanVar(1, 1, 1)
        raw._obj = self._obj.merge()
        return raw

    def freeze(self):
        return self.merge().freeze()

    def freeze_global_mean_var(self):
        return self.merge().freeze_global_mean_var()

    def save(self, path, prefix=""):
        self.merge().save(path, prefix=prefix)

    @classmethod
    def from_file(cls, path, chunk_size, prefix=""):
        return CondMeanVar.from_file(path, prefix=prefix).split(chunk_size)


def cond_mean_var(
    X,
    y,
    num_classes: int,
    chunk_size: int = 0,
    initial_state=None,
    preprocess_block=identity_fn,
    parallel_samples=None,
) -> CondMeanVar:
    """
    Compute a conditional mean and variance.

    :param X: An array of shape (n_samples, n_features).
    :param y: Target values. An array of shape (n_samples,)
        or (n_samples, n_targets).
    :param chunk_size: Process data per block. When a HDF5 array is passed as input
        for X and y, this allows to process the data by small parts
        (that can fit in RAM).
    :param initial_state: An initial accumulator (i.e., a :py:class:`CondMeanVar` instance)
        or path from which we will call :py:meth:`CondMeanVar.from_file`. Accumulation
        will be started from this state and not an empty accumulator.
    :param preprocess_block: a function applied on the data block before being
        accumulated. This function has for signature `fn(X, y) -> X_new`. A typical
        example is to do an FFT of the data. Data passed to this callback is guaranteed
        to be in RAM and as numpy array.
    :param parallel_samples: each thread will process a fixed number of samples
        determined by this value. The number of thread is configured by the
        `RAYON_NUM_THREAD` environment variable.

    :return: A :py:class:`CondMeanVar` instance.
    """
    if initial_state:
        if isinstance(initial_state, (str, Path)):
            src = str(initial_state)
            engine = CondMeanVar.from_file(src)
        elif isinstance(initial_state, CondMeanVar):
            engine = initial_state
        else:
            raise TypeError("unsupported initial state for CondMeanVar")
    else:
        if y.ndim == 1:
            y = y[:, np.newaxis]
        targets = y.shape[1]
        engine = CondMeanVar(targets, X.shape[1], num_classes)

    if parallel_samples:
        engine = engine.split(parallel_samples)
        logger.info(f"using {parallel_samples} samples per thread")

    if chunk_size == 0:
        logger.debug("running accumulator on the full dataset (no chunks)")
        assert np.min(y[:]) >= 0, "labels must be positive"
        data = preprocess_block(X, y)
        engine.process_block(data, y)
    else:
        for start in range(0, X.shape[0], chunk_size):
            end = start + chunk_size
            labels = y[start:end]
            assert np.min(labels) >= 0, "labels must be positive"
            d = preprocess_block(X[start:end], labels)
            logger.debug(f"running accumulator on slice [{start}:{start + d.shape[0]}]")
            engine.process_block(d, labels)
    if parallel_samples:
        engine = engine.merge()
    return engine


def cond_mean(
    X,
    y,
    num_classes: int,
    chunk_size: int = 0,
    initial_state=None,
    preprocess_block=identity_fn,
    parallel_samples=None,
):
    cmean = cond_mean_var(
        X,
        y,
        num_classes,
        chunk_size=chunk_size,
        initial_state=initial_state,
        preprocess_block=preprocess_block,
        parallel_samples=parallel_samples,
    )
    m, v = cmean.freeze()
    if y.ndim == 1:
        return m[0]
    return m


def perceived_information(model, X, y_true, entropy: float) -> float:
    """
    Compute the perceived information of a given model.

    :param model: sklearn-like model, which must have a ``predict_proba`` method.
    :param X: A numpy array of shape ``(n_samples, n_features)`` that represents inputs data.
    :param y_true: A numpy array of shape ``(n_samples,)`` that contains correct labels associated with data.
    :param entropy: entropy of the labels.

    """
    pred = model.predict_proba(X)
    return entropy - log_loss(y_true, pred) / np.log(2)


def fisher_transformation(rho: npt.NDArray, n_traces: int):
    """
    Fisher's transformation to derive a test statistic from correlation coefficient.

    :param rho: Correlation coefficient (i.e., output of CPA).
    :param n_traces: the number of traces.

    """
    return 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n_traces)


def pvalues_from_statistic(s_values: npt.NDArray) -> npt.NDArray:
    """
    Compute the p-values from the value of a statistic (e.g., correlation statistic or t-statistic).
    Namely, the p-value is the probability that the test statistic can be observed under the null hypothesis.

    In the case of the t-statistic, the CDF that is normally the one of a student t-distribution is appoximated with the CDF of a Gaussian distribution.
    Such approximation holds when the t-distribution has a high degree of freedom (which is the case with many SCA traces)

    :params s_values: Values of the statistic
    """
    return 2 * (1 - norm.cdf(np.abs(s_values)))


def p_threshold(traces_length: int, alpha: Optional[float] = 1e-5) -> float:
    """
    Computes the theshold on the p-value under which the null hypothesis is rejected for a given type I error probability alpha.
    The theshold considers the lengths of the traces to adapt the mini-p procedure.
    (See "Towards Sound and Optimal Leakage Detection Procedure", Zhang et al.)

    :param traces_length: length of the traces.
    :param alpha: Type I error probability.

    """
    return 1 - (1 - alpha) ** (1 / traces_length)


def t_threshold(traces_length: int, alpha: Optional[float] = 1e-5) -> float:
    """
    Computes the threshold on the t-statistic depending on the trace length.
    (See "Towards Sound and Optimal Leakage Detection Procedure", Zhang et al.)

    :param traces_length: length of the traces.
    :param alpha: Type I error probability.

    """
    p_thresh = p_threshold(traces_length, alpha=alpha)
    # compute CDF^-1
    return norm.ppf(1 - p_thresh / 2, loc=0, scale=1)