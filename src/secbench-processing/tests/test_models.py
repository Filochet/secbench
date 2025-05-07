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

import random

import numpy as np
import pytest

from secbench.processing import InvalidInputError
from secbench.processing.models import (
    hamming_weight,
    lra_unpackbits,
    lra_unpackbits_2nd_order,
    unpackbits,
)


def test_hamming_weight():
    x = np.array([0xAAAAAAAA, 0x0], dtype=np.uint32)
    y = np.array([0x0, 0x1111_1111_2222_2222], dtype=np.uint64)

    assert list(hamming_weight(x)) == [16, 0]
    assert list(hamming_weight(y)) == [0, 16]

    assert hamming_weight(np.array([1 << 62], dtype=np.uint64)) == 1
    assert hamming_weight(np.array([1 << 12], dtype=np.uint64)) == 1

    assert hamming_weight(np.array([1 << 30], dtype=np.uint32)) == 1
    assert hamming_weight(np.array([1 << 12], dtype=np.uint32)) == 1

    assert hamming_weight(np.array([1 << 15], dtype=np.uint32)) == 1
    assert hamming_weight(np.array([1 << 5], dtype=np.uint32)) == 1


def test_hamming_weight_int():
    # Unsupported input type
    with pytest.raises(InvalidInputError):
        _ = hamming_weight(0.0)

    # Supported, but out of range
    with pytest.raises(InvalidInputError):
        _ = hamming_weight(-1)

    with pytest.raises(InvalidInputError):
        _ = hamming_weight(2**64)

    # Randomized test against oracle.
    for _ in range(200):
        x = random.randint(0, 2**64)
        assert hamming_weight(x) == bin(x).count("1")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("shape", [10, (10, 3), (5, 2, 3)])
@pytest.mark.parametrize("count", [0, 5, 13])
def test_unpackbits(dtype, shape, count):
    y = np.random.randint(0, 2**32, size=shape, dtype=np.uint64)
    if count:
        y = y & ((1 << count) - 1)
    if dtype == np.uint8 and count > 8:
        return

    y = y.astype(dtype)

    decomp = unpackbits(y, count=count)
    n_bits = decomp.shape[-1] if count == 0 else count
    y_rebuilt = np.zeros_like(y, dtype=np.uint64)
    print(decomp.shape)
    for i in range(n_bits):
        y_rebuilt += decomp[..., i].astype(np.uint64) << i
    y_rebuilt = y_rebuilt.astype(y.dtype)
    np.testing.assert_equal(y, y_rebuilt)


def test_lra_unpackbits():
    xs = np.arange(256, dtype=np.uint8)
    ys = lra_unpackbits(xs, center=True)
    assert ys.shape == (256, 9)
    np.testing.assert_equal(ys[:, -1], np.ones(256))
    assert len(np.unique(ys)) == 2, "the output is made of two values"
    print(ys[0])
    np.testing.assert_equal(ys[0, :8], -1 * np.ones(8))
    np.testing.assert_equal(ys[-1], 1 * np.ones(9))

    ys = lra_unpackbits(xs, with_intercept=False)
    assert ys.shape == (256, 8)

    ys = lra_unpackbits(xs, with_intercept=False, count=3)
    assert ys.shape == (256, 3)

    ys = lra_unpackbits(xs, with_intercept=True, count=5)
    assert ys.shape == (256, 6)


def test_lra_unpackbits_example():
    ys = np.array([0, 0b0001_1101, 0x80], dtype=np.uint8)
    expected = np.array(
        [[0] * 8, [1, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.uint8
    )
    expected_i8 = (2 * (expected - 0.5)).astype(np.int8)
    expected_lra = np.c_[expected_i8, np.ones(expected_i8.shape[0], dtype=np.int8)]
    bs = lra_unpackbits(ys, center=True)
    assert bs.dtype == np.int8
    np.testing.assert_equal(bs, expected_lra)

    bs = lra_unpackbits(ys, center=False)
    np.testing.assert_equal(
        bs, np.c_[expected, np.ones(expected.shape[0], dtype=np.int8)]
    )

    bs = lra_unpackbits_2nd_order(ys)
    assert bs.shape == (3, 37)