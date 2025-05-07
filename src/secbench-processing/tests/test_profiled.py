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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from secbench.processing.crypto.aes import aes_sbox
from secbench.processing.helpers import has_tensorflow
from secbench.processing.models import (
    hamming_weight,
    hamming_weight_leakage,
)
from secbench.processing.profiled import (
    ClassPCA,
    DenseLayerSpec,
    GenericNetworkBuilder,
    SklearnModel,
)

tensorflow_tc = pytest.mark.skipif(
    not has_tensorflow(), reason="tensorflow not installed or not working"
)


def run_aes_attack(key: int, model, **kwargs):
    n_traces = 100_000

    p = np.random.randint(0, 256, size=n_traces, dtype=np.uint8)

    y = aes_sbox(p ^ key)
    X = hamming_weight_leakage(y, n_leaking_samples=3, n_random_samples=3, noise=0.01)
    X_train, X_test, p_train, p_test = train_test_split(X, p, test_size=0.01)

    model.fit(X_train, p_train, secret=key, **kwargs)
    scores = model.key_scores(X_test, np.arange(256), p_test)
    return scores


def aes_sbox_model(private, public):
    return hamming_weight(aes_sbox(private ^ public))


def test_class_pca():
    n_samples = 5_000
    y = np.random.randint(0, 256, size=n_samples, dtype=np.uint8)
    X = np.random.random((n_samples, 3000))
    Xt = ClassPCA(n_components=10).fit_transform(X, y)
    assert Xt.shape == (n_samples, 10)


@pytest.mark.parametrize("key", [32, 66])
def test_profiled_qda(key):
    model = SklearnModel(
        QuadraticDiscriminantAnalysis(), target_variable_fn=aes_sbox_model
    )
    scores = run_aes_attack(key, model)
    assert scores.shape == (256,)
    assert np.argmax(scores) == key


@tensorflow_tc
def test_profiled_mlp():
    model = GenericNetworkBuilder(
        conv_layers=[],
        batch_normalization=False,
        dense_layers=[DenseLayerSpec(size=4, activation=None)],
    ).build(6, num_classes=9, target_variable_fn=aes_sbox_model)
    key = 0xAA
    scores = run_aes_attack(key, model)
    assert scores.shape == (256,)
    assert np.argmax(scores) == key