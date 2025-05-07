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

from secbench.processing import metrics, secbench_native_installed
from secbench.processing.crypto.aes import aes_sbox_leakage
from secbench.processing.models import (
    lra_unpackbits,
    lra_unpackbits_2nd_order,
    lra_unpackbits_shd,
)

secbench_ffi_testcase = pytest.mark.skipif(
    not secbench_native_installed(), reason="secbench_ffi package is not installed"
)


def test_vpearson():
    n = 10000
    p = 100
    pts = np.random.randint(0, 256, size=n, dtype=np.uint8)
    leak = aes_sbox_leakage(0x42, pts, noise=0.3)
    power = np.random.normal(size=n * p, scale=3.0).reshape(n, p)
    power[:, 10] = leak

    hyps = [aes_sbox_leakage(k, pts) for k in range(255)]
    hyps = np.array(hyps, dtype=np.uint8).T
    # Check legacy vpearson method
    r = metrics.vpearson(power, hyps)
    k_best = np.argmax(np.max(np.abs(r), axis=1))
    assert k_best == 0x42

    # Check current fast pearson
    r_fast = metrics.vpearson_fast(power, hyps)
    k_best = np.argmax(np.max(np.abs(r_fast), axis=1))
    assert k_best == 0x42


def test_lra():
    n_samples, n_features = 5000, 20

    key = np.random.randint(256)
    k = np.array([key] * n_samples, dtype=np.uint8)
    pt = np.random.randint(0, 256, size=n_samples, dtype=np.uint8)

    # Create simulated power leakage
    poi = np.random.randint(n_features)
    leak = aes_sbox_leakage(k, pt, noise=0.5)
    X = np.random.normal(size=(n_samples, n_features), scale=3.0)
    X[:, poi] = leak
    leak = np.c_[leak, np.ones(leak.shape[0])]

    # Apply LRA
    scores, sols = metrics.lra_lsqr(X, [leak])
    assert np.argmax(scores) == poi


@secbench_ffi_testcase
def test_snr():
    n_samples = 10000
    n_features = 10
    n_targets = 15
    X = np.random.normal(size=(n_samples, n_features))
    y = np.random.randint(0, n_targets, size=(n_samples))

    # bias traces to have significant snr
    for i in range(n_samples):
        X[i, : n_features // 2] = X[i, : n_features // 2] + y[i]

    snr_accumulator = metrics.snr(X, y)
    snr = metrics.snr(X, y)

    print(snr_accumulator - snr)
    assert np.all(np.abs(snr_accumulator - snr) < 1e-3)


@secbench_ffi_testcase
@pytest.mark.parametrize("do_t_test", [metrics.welch_t_test])
def test_wetch_t_test(do_t_test):
    n_samples, n_features = 2000, 5

    target = np.zeros(2 * n_samples, dtype=np.int8)
    target[n_samples:] = 1

    xs = np.random.normal(size=(n_samples, n_features))
    ys = np.random.normal(size=(n_samples, n_features))
    zs = np.random.normal(size=(n_samples, n_features)) + 3

    r = np.abs(do_t_test(np.concatenate((xs, ys), axis=0), target))
    p = np.abs(do_t_test(np.concatenate((xs, zs), axis=0), target))
    assert np.all(r) < 4.5
    assert np.all(r < p)
    assert np.any(p > 4.5)


@secbench_ffi_testcase
def test_dom():
    n_samples, n_features = 2000, 5

    target = np.zeros(2 * n_samples, dtype=np.int8)
    target[n_samples:] = 1
    xs = np.random.normal(size=(n_samples, n_features))
    ys = np.random.normal(size=(n_samples, n_features))
    zs = np.random.normal(size=(n_samples, n_features)) + 3
    assert np.max(metrics.dom(np.concatenate((xs, zs), axis=0), target)) - 3 < 0.5
    assert np.max(metrics.dom(np.concatenate((xs, ys), axis=0), target)) < 0.3


@secbench_ffi_testcase
@pytest.mark.parametrize(
    "score",
    [
        metrics.snr,
        metrics.nicv,
        metrics.sost,
        metrics.pearson,
        metrics.LRA(model=lra_unpackbits, count=3),
        metrics.LRA(model=lra_unpackbits, use_cond_mean=True, n_classes=8, count=3),
        metrics.LRA(
            model=lra_unpackbits_2nd_order, use_cond_mean=True, n_classes=8, count=3
        ),
    ],
)
def test_metrics(score):
    n_targets = 5
    n_samples, n_features = 5000, 10

    train = np.random.normal(scale=0.05, size=(n_samples, n_features))
    classes = np.random.randint(8, size=n_samples, dtype=np.uint8)
    p = np.random.choice(range(n_features))
    train[:, p] += 0.5 * classes

    c_ok = np.random.choice(range(n_targets))
    multi_classes = np.random.randint(8, size=(n_samples, n_targets), dtype=np.uint8)
    multi_classes[:, c_ok] = classes

    r = score(train, classes)
    assert r.ndim == 1
    assert np.argmax(r) == p

    r = score(train, multi_classes)
    assert r.ndim == 2
    assert r.shape == (n_targets, n_features)
    assert np.argmax(r, axis=1)[c_ok] == p


@secbench_ffi_testcase
def test_bad_labels():
    xs = np.random.random((10, 100))
    bad_1 = np.array([-2, -3, -2, 3, 3, 1, 1, 1, 1, -2])
    for metric in [metrics.snr, metrics.nicv, metrics.sost]:
        with pytest.raises(AssertionError):
            _ = metric(xs, bad_1)
        # Should work with labels re-encoding
        _ = metric(xs, bad_1, encode_labels=True)


def test_lra_shd():
    n_samples, n_features = 5000, 10

    train = np.random.normal(scale=0.5, size=(n_samples, n_features))
    classes1 = np.random.randint(256, size=n_samples, dtype=np.uint8)
    classes2 = np.random.randint(256, size=n_samples, dtype=np.uint8)
    p = np.random.choice(range(n_features))
    train[:, p] += 0.5 * classes1 - classes2

    score = metrics.LRA(model=lra_unpackbits_shd, use_cond_mean=False)
    r = score(train, classes1, classes2)
    assert r.ndim == 1
    assert np.argmax(r) == p


def test_lra_colinear():
    train = np.random.normal(size=(5000, 10))
    cls = np.random.randint(0, 7, size=5000, dtype=np.uint8)
    with pytest.raises(ValueError, match="unable to find"):
        metrics.LRA(model=lra_unpackbits)(train, cls)