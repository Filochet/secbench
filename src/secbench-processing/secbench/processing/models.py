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

import random
import sys

import numpy as np
import numpy.typing as npt

from .helpers import InvalidInputError, UnsignedScalar, check_array

_HW8_TABLE = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)


def hamming_weight(x: npt.NDArray[UnsignedScalar] | int):
    """
    Vectorized Hamming Weight

    Returns the number of bits set to 1 (aka. bit count) in each element of the
    given array.

    .. note::

        This function does a simple dispatching to scalar specific hamming
        weight functions (``hamming_weight_N``). You should consider calling
        them directly if the datatypes are known beforehand.

    :Examples:

    >>> hamming_weight(np.arange(16, dtype=np.uint8))
    array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4], dtype=uint8)

    """
    if isinstance(x, (int, np.uint8, np.uint16, np.uint32, np.uint64)):
        if x < 0 or x >= 2**64:
            raise InvalidInputError(
                f"input must be in the range [0, 2^64), actual value is {x}"
            )
        return hamming_weight_64(int(x))
    elif isinstance(x, np.ndarray):
        check_array(x, dtype=(np.uint8, np.uint16, np.uint32, np.uint64))
        if x.dtype == np.dtype("uint8"):
            return hamming_weight_8(x)
        elif x.dtype == np.dtype("uint16"):
            return hamming_weight_16(x)
        elif x.dtype == np.dtype("uint32"):
            return hamming_weight_32(x)
        elif x.dtype == np.dtype("uint64"):
            return hamming_weight_64(x)
        else:
            raise NotImplementedError("unreachable")
    else:
        raise InvalidInputError(
            f"unsupported input type, should be 'int' or 'np.ndarray', got {type(x)}"
        )


def hamming_distance(x: np.ndarray, y: np.ndarray):
    """Vectorized Hamming Distance

    Returns the number of bits that changed between x and y

    :Examples:

    >>> x = np.array([42, 3, 12], dtype=np.uint8)
    >>> y = np.array([11, 3, 59], dtype=np.uint8)
    >>> hamming_distance(x, y)
    array([2, 0, 5], dtype=uint8)

    """
    return hamming_weight(x ^ y)


def hamming_weight_8(x):
    return _HW8_TABLE[x]


def hamming_weight_16(x):
    return hamming_weight_8(x & 0xFF) + hamming_weight_8(x >> 8)


def hamming_weight_32(x):
    return hamming_weight_16(x & 0xFFFF) + hamming_weight_16(x >> 16)


def hamming_weight_64(x):
    return hamming_weight_32(x & 0xFFFF_FFFF) + hamming_weight_32(x >> 32)


def hamming_weight_leakage(
    y, n_leaking_samples=1, n_random_samples=0, noise=0, shuffle=False
):
    """
    Generate hamming weight leakages from byte values.

    :param y: Input bytes values
    :param noise: Gaussian noise scale added to the samples.
    :param shuffle: Shuffle the order of samples
    :return: Leakage traces.
    """
    n_traces = y.shape[0]
    h = hamming_weight(y)

    if not isinstance(noise, np.ndarray):
        noise = np.repeat(noise, n_traces)

    cols = []
    for i in range(n_leaking_samples):
        col = np.random.normal(0, noise[i], n_traces)
        col += h
        cols.append(col)

    for _ in range(n_random_samples):
        b_cst = np.random.normal(0, 5)
        col = b_cst + np.random.normal(0, noise[0], n_traces)
        cols.append(col)

    if shuffle:
        random.shuffle(cols)
    return np.array(cols).T


def _is_little(dtype) -> bool:
    sys_is_le = sys.byteorder == "little"
    if dtype == np.uint8:
        return True
    return dtype.byteorder == "<" or (dtype.byteorder == "=" and sys_is_le)


def unpackbits(y: np.ndarray, count: int = 0) -> np.ndarray:
    """
    Unpack an array into its bit representation.

    This is function can be viewed as a more general version than Numpy's
    `unpackbits` function. It supports arbitrary integer
    types (`uint16`, `uint32`, etc.).

    .. note::

        Any other type (e.g., `np.float32`) will be decomposed according to their byte
        representation.

    :param y: a numpy array of integer elements. This array can be of any shape.
    :param count: number of bits to keep in the final decomposition. Must be in the
        range `[0, n_bits)` otherwise this parameter has no effect.

    :return: If y has a shape `(n_0, ..., n_k)`, the output will be an array of type
        `np.int8`, with shape `(n_0, ..., n_k, N)` where `N` is the number of bits
        needed to encode the integer elements. The bits are returned in little endian
        order (least significant first).
    """
    output_shape = np.append(y.shape, y.itemsize * 8)
    byteorder = y.dtype.byteorder
    if byteorder == ">":
        y = y.byteswap()
    elif not _is_little(y.dtype):
        raise ValueError(f"unsupported byteorder for input array: {byteorder}")
    decomp_le = np.unpackbits(y.view(dtype=np.uint8), bitorder="little").reshape(
        *output_shape
    )
    if 0 < count < decomp_le.shape[-1]:
        decomp_le = decomp_le[..., :count]
    return decomp_le


def lra_unpackbits(
    y: np.ndarray, center=False, with_intercept=True, count=0
) -> np.ndarray:
    """
    Decompose a target vector into bits, suited for LRA use.

    :param y: a numpy array of integers to be decomposed. This array can be of any shape.
    :param with_intercept: add a column of one to the output to capture the
        intercept when the result is passed to a least square solver.
    :param count: number of bits to keep in the final decomposition. Must be in the
        range `[0, n_bits)` otherwise this parameter has no effect.
    :param center: if given, center the bit decomposition such as the mean is 0
        for random inputs.
    :return:
        If y has a shape ``(n_0, ..., n_k)``, the output will be an array of type
        ``np.int8``, with shape ``(n_0, ..., n_k, N + I)`` where ``N`` is the number
        of bits needed to encode the integer elements, in `I` is 1 when
        ``with_intercept=True``. The bits are returned in little endian order
        (least significant first). The intercept is added in the last column.

    .. seealso::

        The same rules apply on the input as for :py:func:`unpackbits`.
    """
    y_bits = unpackbits(y, count=count)
    if center:
        y_bits = (2.0 * (y_bits - 0.5)).astype(np.int8)
    if with_intercept:
        ones = np.expand_dims(np.ones(y.shape, dtype=np.int8), axis=-1)
        y_bits = np.c_[y_bits, ones]
    return y_bits


def _gf2n_order2_decomp_indices(n):
    indices = []
    for i in range(n):
        indices.append((i, i))
    for i in range(n):
        for j in range(i + 1, n):
            indices.append((i, j))
    return indices


def lra_unpackbits_2nd_order(
    y: np.ndarray,
    with_intercept=True,
    count=0,
    center=False,
) -> np.ndarray:
    """
    Bit decomposition of algebraic degree 2, suited for LRA use.

    Usage is the same, as :py:func:`lra_unpackbits`.

    The order of bits in the decomposition is:

    - First, the bits of ``y`` decomposition
    - Then, the bits ``y_i * y_j`` for ``0 < i < j < N``
    - If applicable the intercept in the last column

    :return:
        If y has a shape ``(n_0, ..., n_k)``, the output will be an array of type
        ``np.int8``, with shape ``(n_0, ..., n_k, W + I)`` where ``W = N * (N + 1) / 2``
        (`N` being the number of bits of y decomposition) needed to encode the integer
        elements, in `I` is 1 when `with_intercept=True`. The bits are returned in
        little endian order (least significant first). The intercept is added in the
        last column.
    """
    y_bits = unpackbits(y, count=count)
    n_bits = y_bits.shape[-1]
    comb = _gf2n_order2_decomp_indices(n_bits)

    output_shape = np.append(y_bits.shape[:-1], len(comb))
    decomp = np.zeros(output_shape, np.int8)
    for idx, (bit_0, bit_1) in enumerate(comb):
        decomp[..., idx] = y_bits[..., bit_0] * y_bits[..., bit_1]

    if center:
        decomp[..., :n_bits] = 2 * decomp[..., :n_bits] - 1
        decomp[..., n_bits:] = 4 * decomp[..., n_bits:] - 1

    if with_intercept:
        ones = np.expand_dims(np.ones(y.shape, dtype=np.int8), axis=-1)
        decomp = np.c_[decomp, ones]

    return decomp


def lra_unpackbits_shd(
    y: np.ndarray, y_prev: np.ndarray, with_intercept=True
) -> np.ndarray:
    """
    Generate signed hamming distance leakage between two variables where  0 -> 1 and 1 -> 0
    are encoded in separate variables.

    :param y: 1D array containing target variable.
    :param y_prev: 1D array containing the distant variable.

    """
    z0 = y
    z1 = y_prev

    if z0.ndim != 1:
        raise ValueError("z0 must be a 1D array")
    if z1.ndim != 1:
        raise ValueError("z1 must be a 1D array")

    # Unpack the bits
    l0 = np.unpackbits(z0.reshape(-1, 1), axis=1)
    l1 = np.unpackbits(z1.reshape(-1, 1), axis=1)

    if l0.shape != l1.shape:
        raise ValueError("The two decompositions must have the same shape")

    return gen_shd(l0, l1, with_intercept=with_intercept)


def gen_shd(bits0: np.ndarray, bits1: np.ndarray, with_intercept=True) -> np.ndarray:
    """
    Generate signed hamming distance model variables where  0 -> 1 and 1 -> 0
    are encoded in separate variables

    :params bits0: (n_samples, n_features) array containing the first variables bit representations
    :params bits1: (n_samples, n_features) array containing the second variables bit representations

    """
    l0 = bits0
    l1 = bits1

    # Generate zero filled leakage array
    leak = np.zeros(shape=(l0.shape[0], l0.shape[1] * 2), dtype=np.int8)
    for bit in range(l0.shape[1]):
        # 0 -> 1
        sel = (l0[:, bit] == 0) & (l1[:, bit] == 1)
        leak[sel, 2 * bit] = 1
        # 1 -> 0
        sel = (l0[:, bit] == 1) & (l1[:, bit] == 0)
        leak[sel, 2 * bit + 1] = 1

    if with_intercept:
        leak = np.c_[leak, np.ones(leak.shape[0], np.int8)]
    return leak