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

from secbench.processing.helpers import MissingPackageError, add_remove, rank_of


def test_rank_of():
    scores = np.arange(13)
    for i in range(13):
        assert rank_of(scores, i) == 12 - i
        assert rank_of(scores, i, randomize=False) == 12 - i

    scores = np.array([3.0, 8.0, 4.0, 5.0, 4.0, 4.0, 1.0])
    ranks = set()
    for _ in range(100):
        r = rank_of(scores, 2)
        ranks.add(r)
    assert ranks == {2, 3, 4}


def test_add_remove():
    n_samples, n_features = 10, 100
    xs = np.random.random((n_samples, n_features))
    try:
        xs_tr = add_remove(xs, ratio=0.1)
    except MissingPackageError:
        pytest.skip("numba must be installed to run this test")
    assert xs_tr.shape == xs.shape
    assert xs_tr.dtype == xs.dtype
    assert np.sum((xs - xs_tr) ** 2) > 10


def test_encode_labels():
    from secbench.processing.helpers import encode_labels

    labels = np.array([[4, 5, 6, 7, 4], [3, 6, 5, 3, 4]], dtype=np.uint8).T

    # 1D case
    ys_encoded = encode_labels(labels[:, 0])
    assert ys_encoded.ndim == 1
    assert len(np.unique(ys_encoded)) == 4
    assert np.max(ys_encoded) == 3

    # 2D case, independent labels
    ys_encoded = encode_labels(labels)
    assert ys_encoded.shape == labels.shape
    assert ys_encoded.dtype == np.uint16
    assert len(np.unique(ys_encoded[:, 0])) == 4
    assert len(np.unique(ys_encoded[:, 1])) == 4
    assert np.max(ys_encoded) == 3

    # 2D case, not independent labels
    ys_encoded = encode_labels(labels, indep=False)
    assert ys_encoded.shape == labels.shape
    assert len(np.unique(ys_encoded.flatten())) == 5
    assert np.max(ys_encoded) == 4