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
Reference implementations of sliding processing function.

Those functions are not intended to be used in production, they are not easily parralelizable.

The implementation are extracted from pandas.
"""

import numpy as np
import pandas as pd


def sliding_mean(X: np.ndarray, winsize: int, na_value=0):
    """
    Compute windowed sliding average on the data with stride 1.
    Metric is computed on the last dimension.

    :param X: A (n_samples, n_features) or (n_features) numpy array.
    :param winsize: Size of the window.
    :na_value: Value to use for replacing NaN.

    """
    if X.ndim == 1:
        XT = X
    else:
        XT = X.T

    Y = (
        pd.DataFrame(XT)
        .rolling(winsize)
        .mean()
        .to_numpy(na_value=na_value)
        .reshape(XT.shape)
    )

    if X.ndim == 1:
        return Y
    else:
        return Y.T


def sliding_var(X: np.ndarray, winsize: int, na_value=0):
    """
    Compute windowed sliding variance on the data with stride 1.
    Metric is computed on the last dimension.

    :param X: A (n_samples, n_features) or (n_features) numpy array.
    :param winsize: Size of the window.
    :na_value: Value to use for replacing NaN.

    """
    if len(X.shape) - 1 == 0:
        XT = X
    else:
        XT = X.T

    Y = (
        pd.DataFrame(XT)
        .rolling(winsize)
        .var()
        .to_numpy(na_value=na_value)
        .reshape(XT.shape)
    )

    if len(X.shape) - 1 == 0:
        return Y
    else:
        return Y.T


def sliding_std(X: np.ndarray, winsize: int, na_value=0):
    """
    Compute windowed sliding standard deviation on the data with stride 1.
    Metric is computed on the last dimension.

    :param X: A (n_samples, n_features) or (n_features) numpy array.
    :param winsize: Size of the window.
    :na_value: Value to use for replacing NaN.

    """
    if len(X.shape) - 1 == 0:
        XT = X
    else:
        XT = X.T

    Y = (
        pd.DataFrame(XT)
        .rolling(winsize)
        .std()
        .to_numpy(na_value=na_value)
        .reshape(XT.shape)
    )

    if len(X.shape) - 1 == 0:
        return Y
    else:
        return Y.T


def sliding_skew(X: np.ndarray, winsize: int, na_value=0):
    """
    Compute windowed sliding skewness on the data with stride 1.
    Metric is computed on the last dimension.

    :param X: A (n_samples, n_features) or (n_features) numpy array.
    :param winsize: Size of the window.
    :na_value: Value to use for replacing NaN.

    """
    if len(X.shape) - 1 == 0:
        XT = X
    else:
        XT = X.T

    Y = (
        pd.DataFrame(XT)
        .rolling(winsize)
        .skew()
        .to_numpy(na_value=na_value)
        .reshape(XT.shape)
    )

    if len(X.shape) - 1 == 0:
        return Y
    else:
        return Y.T


def sliding_kurt(X: np.ndarray, winsize: int, na_value=0):
    """
    Compute windowed sliding kurtosis on the data with stride 1.
    Metric is computed on the last dimension.

    :param X: A (n_samples, n_features) or (n_features) numpy array.
    :param winsize: Size of the window.
    :na_value: Value to use for replacing NaN.

    """
    if len(X.shape) - 1 == 0:
        XT = X
    else:
        XT = X.T

    Y = (
        pd.DataFrame(XT)
        .rolling(winsize)
        .kurt()
        .to_numpy(na_value=na_value)
        .reshape(XT.shape)
    )

    if len(X.shape) - 1 == 0:
        return Y
    else:
        return Y.T