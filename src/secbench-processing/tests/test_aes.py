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

import numpy as np
import pytest

from secbench.processing.crypto.aes import (
    AES,
    AesOps,
    aes_expand_key,
    aes_inv_expand_key,
    aes_inv_expand_key_step,
    aes_nist_key,
    biased_hd_plaintexts,
    biased_state_plaintexts,
    generate_plaintexts,
    generate_round_states,
)


def test_sub_bytes():
    pts = np.random.randint(0, 256, size=(200, 16), dtype=np.uint8)
    subs = AesOps.sub_bytes(pts)
    un_subs = AesOps.inv_sub_bytes(subs)
    assert np.all(un_subs == pts)


def test_shift_rows():
    pts = np.random.randint(0, 256, size=(200, 16), dtype=np.uint8)
    shifted = AesOps.shift_rows(pts)
    un_shifted = AesOps.inv_shift_rows(shifted)
    assert np.all(un_shifted == pts)


def test_mix_columns():
    pts = np.random.randint(0, 256, size=(200, 16), dtype=np.uint8)
    mixed = AesOps.mix_columns(pts)
    unmixed = AesOps.inv_mix_columns(mixed)
    assert np.all(unmixed == pts)


def test_add_round_key():
    s0 = np.tile(np.uint8(3), 16)
    s1 = np.tile(np.uint8(4), 16)
    s2 = np.tile(np.uint8(5), 16)

    states = np.array([s0, s1, s2])

    rk = np.tile(np.uint8(12), 16)

    updated = AesOps.add_round_key(states, rk)

    assert np.all(updated[0] == 15)
    assert np.all(updated[1] == 8)
    assert np.all(updated[2] == 9)


@pytest.fixture
def pts():
    yield np.random.randint(0, 256, size=(200, 16), dtype=np.uint8)


@pytest.fixture()
def rnd_key():
    yield np.random.randint(0, 256, size=16, dtype=np.uint8)


def test_correctness(rnd_key, pts):
    try:
        from Crypto.Cipher import AES as AESRefModel
    except ImportError:
        pytest.skip("pycryptodome is not installed, cannot verify AES")

    cipher = AES(rnd_key)
    cts = cipher.encrypt(pts)

    # PyCrypto encryption
    ref_cipher = AESRefModel.new(rnd_key.tobytes(), AESRefModel.MODE_ECB)
    for i, pt in enumerate(pts):
        good = np.frombuffer(ref_cipher.encrypt(pt.tobytes()), dtype=np.uint8)
        assert np.all(good == cts[i])


def test_encryption_decryption(rnd_key, pts):
    cipher = AES(rnd_key)
    ciphertexts = cipher.encrypt(pts)
    decrypted = cipher.decrypt(ciphertexts)
    assert np.all(decrypted == pts)


def test_from_round_key(rnd_key, pts):
    round_keys = aes_expand_key(rnd_key)
    cts_ref = AES(rnd_key).encrypt(pts)
    for i in range(11):
        cipher = AES.from_round_key(round_keys[i], i)
        np.testing.assert_equal(cipher.encrypt(pts), cts_ref)


def test_partial_encryption(rnd_key, pts):
    cipher = AES(rnd_key)
    ciphertexts = cipher.encrypt(pts)
    # Testing start and stop rounds
    s_i = cipher.encrypt(pts, stop_round=2)
    finals = cipher.encrypt(s_i, start_round=3)
    assert np.all(finals == ciphertexts)
    # Testing start and stop operations
    s_i = cipher.encrypt(pts, stop_round=4, stop_after=AesOps.sub_bytes)
    finals = cipher.encrypt(s_i, start_round=4, start_after=AesOps.sub_bytes)
    assert np.all(finals == ciphertexts)

    s_i = cipher.encrypt(pts, stop_round=6, stop_after=AesOps.shift_rows)
    finals = cipher.encrypt(s_i, start_round=6, start_after=AesOps.shift_rows)
    assert np.all(finals == ciphertexts)

    s_i = cipher.encrypt(pts, stop_round=2, stop_after=AesOps.mix_columns)
    finals = cipher.encrypt(s_i, start_round=2, start_after=AesOps.mix_columns)
    assert np.all(finals == ciphertexts)


def test_partial_decryption(rnd_key, pts):
    cipher = AES(rnd_key)
    ciphertexts = cipher.encrypt(pts)
    # Testing start and stop rounds
    s_i = cipher.decrypt(ciphertexts, start_round=10, stop_round=5)
    finals = cipher.decrypt(s_i, start_round=4)
    assert np.all(finals == pts)
    # Testing start and stop operations
    s_i = cipher.decrypt(ciphertexts, stop_round=4, stop_after=AesOps.inv_sub_bytes)
    finals = cipher.decrypt(s_i, start_round=4, start_after=AesOps.inv_sub_bytes)
    assert np.all(finals == pts)

    s_i = cipher.decrypt(ciphertexts, stop_round=6, stop_after=AesOps.inv_shift_rows)
    finals = cipher.decrypt(s_i, start_round=6, start_after=AesOps.inv_shift_rows)
    assert np.all(finals == pts)

    s_i = cipher.decrypt(ciphertexts, stop_round=2, stop_after=AesOps.inv_mix_columns)
    finals = cipher.decrypt(s_i, start_round=2, start_after=AesOps.inv_mix_columns)
    assert np.all(finals == pts)


def test_partial_both(rnd_key, pts):
    cipher = AES(rnd_key)
    # Testing start and stop rounds
    s_i = cipher.encrypt(pts, stop_round=3)
    finals = cipher.decrypt(s_i, start_round=3)
    assert np.all(finals == pts)

    s_i = cipher.encrypt(pts, stop_round=3, stop_after=AesOps.sub_bytes)
    finals = cipher.decrypt(s_i, start_round=3, start_after=AesOps.inv_shift_rows)
    assert np.all(finals == pts)


def test_meet_in_middle(rnd_key, pts):
    cipher = AES(rnd_key)
    cts = cipher.encrypt(pts)
    # Round 0
    s_i0 = cipher.encrypt(pts, stop_after=AesOps.add_round_key, stop_round=0)
    s_i1 = cipher.decrypt(cts, stop_after=AesOps.inv_sub_bytes, stop_round=1)
    assert np.all(s_i0 == s_i1)
    # Round 1 to 9
    for i in range(1, 10):
        print(i)
        s_i0 = cipher.encrypt(pts, stop_after=AesOps.sub_bytes, stop_round=i)
        s_i1 = cipher.decrypt(cts, stop_after=AesOps.inv_shift_rows, stop_round=i)
        assert np.all(s_i0 == s_i1)

        s_i0 = cipher.encrypt(pts, stop_after=AesOps.shift_rows, stop_round=i)
        s_i1 = cipher.decrypt(cts, stop_after=AesOps.inv_mix_columns, stop_round=i)
        assert np.all(s_i0 == s_i1)

        s_i0 = cipher.encrypt(pts, stop_after=AesOps.mix_columns, stop_round=i)
        s_i1 = cipher.decrypt(cts, stop_after=AesOps.add_round_key, stop_round=i)
        assert np.all(s_i0 == s_i1)

        s_i0 = cipher.encrypt(pts, stop_after=AesOps.add_round_key, stop_round=i)
        s_i1 = cipher.decrypt(cts, stop_after=AesOps.inv_sub_bytes, stop_round=i + 1)
        assert np.all(s_i0 == s_i1)

    # Round 10
    assert np.all(s_i0 == s_i1)
    s_i0 = cipher.encrypt(pts, stop_after=AesOps.sub_bytes, stop_round=10)
    s_i1 = cipher.decrypt(cts, stop_after=AesOps.inv_shift_rows, stop_round=10)
    assert np.all(s_i0 == s_i1)

    s_i0 = cipher.encrypt(pts, stop_after=AesOps.shift_rows, stop_round=10)
    s_i1 = cipher.decrypt(cts, stop_after=AesOps.add_round_key, stop_round=10)
    assert np.all(s_i0 == s_i1)


def test_reverse_key_schedule(rnd_key):
    cipher = AES(rnd_key)
    k10 = cipher._round_keys[10]
    c_k9 = aes_inv_expand_key_step(k10, 10)
    assert np.all(c_k9 == cipher._round_keys[9])
    rev_keys = aes_inv_expand_key(k10)
    assert np.all(cipher._round_keys == rev_keys)


def test_aes_key_schedule():
    # Fixed test vector for AES key expansion and its inverse.
    k_0_expected = np.array(
        [104, 29, 152, 105, 185, 81, 43, 138, 133, 97, 28, 234, 40, 251, 189, 231],
        dtype=np.uint8,
    )
    k_10_expected = np.array(
        [10, 99, 200, 190, 194, 226, 53, 51, 40, 224, 23, 44, 123, 240, 110, 204],
        dtype=np.uint8,
    )

    k_r = aes_expand_key(k_0_expected)
    np.testing.assert_equal(k_r[10], k_10_expected)

    k_r = aes_inv_expand_key(k_10_expected)
    np.testing.assert_equal(k_r[0], k_0_expected)


@pytest.mark.parametrize(
    "aes_ops",
    [("inv_mix_columns", AesOps.shift_rows), ("inv_shift_rows", AesOps.sub_bytes)],
)
def test_biased_state_plaintexts(aes_ops):
    for random_bytes in [[1], [0, 5], [7, 8], [13, 14, 6]]:
        for aes_round in range(1, 11):
            labels, pts = biased_state_plaintexts(
                100,
                aes_nist_key(),
                target_round=aes_round,
                target_op=aes_ops[0],
                random_bytes=random_bytes,
            )
            aes = AES(aes_nist_key())
            states = aes.encrypt(pts, stop_round=aes_round, stop_after=aes_ops[1])
            for b in range(16):
                if b in random_bytes:
                    assert np.any(states[labels == 1, b] != 255)
                    assert np.any(states[labels == 0, b] != 0)
                else:
                    assert np.all(states[labels == 1, b] == 255)
                    assert np.all(states[labels == 0, b] == 0)


def test_generate_pts(tmp_path):
    # Test single-bytes randomization
    for idx in range(16):
        pts = generate_plaintexts(100, random_bytes=[idx])
        summed = np.sum(pts.astype(np.uint32), axis=0)
        assert summed[idx] != 0
        assert np.all(summed[np.arange(summed.shape[0]) != idx] == 0)

    # Test row randomization
    for i in range(4):
        pts = generate_plaintexts(100, random_rows=[i])
        summed = np.sum(pts.astype(np.uint32), axis=0)
        idx = np.arange(0, 16, 4) + i
        assert np.all(summed[idx] != 0)
        zeros_idx = np.ones(16, dtype=bool)
        zeros_idx[idx] = False
        assert np.all(summed[zeros_idx] == 0)

    # Test column randomization
    for i in range(4):
        pts = generate_plaintexts(100, random_cols=[i])
        summed = np.sum(pts.astype(np.uint32), axis=0)
        idx = np.arange(0, 4) + 4 * i
        assert np.all(summed[idx] != 0)
        zeros_idx = np.ones(16, dtype=bool)
        zeros_idx[idx] = False
        assert np.all(summed[zeros_idx] == 0)


def test_generate_round_states(tmp_path):
    pts = np.random.randint(0, 256, size=(100, 16), dtype=np.uint8)
    key = np.random.randint(0, 256, size=16, dtype=np.uint8)

    for model in ["hw", "hd", "shd"]:
        for target_op in ["add_round_key", "sub_bytes", "shift_rows", "mix_columns"]:
            labels = generate_round_states(key, pts, target_op=target_op, model=model)
            if model == "shd":
                assert labels.shape == (9, 2, 100, 16)
            else:
                assert labels.shape == (9, 100, 16)


def test_hd_bias_state(tmp_path):
    for sign in ["positive", "negative", "unsigned"]:
        for aes_round in range(1, 10):
            for biased_row in [0, 1, 2, 3, 12]:
                labels, pts = biased_hd_plaintexts(
                    100,
                    aes_nist_key(),
                    target_round=aes_round,
                    biased_row=biased_row,
                    sign=sign,
                )
                aes = AES(aes_nist_key())
                states1 = aes.encrypt(
                    pts, stop_round=aes_round - 1, stop_after=AesOps.add_round_key
                )
                states2 = aes.encrypt(
                    pts, stop_round=aes_round, stop_after=AesOps.add_round_key
                )
                if biased_row != 12:
                    for b in range(16):
                        if b % 4 != biased_row:
                            assert np.any(states1[labels == 1, b] != 255)
                            assert np.any(states1[labels == 0, b] != 0)
                            assert np.any(states2[labels == 1, b] != 255)
                            assert np.any(states2[labels == 0, b] != 0)
                        else:
                            if sign == "positive":
                                assert np.all(states1[:, b] == 0)
                            if sign == "negative":
                                assert np.all(states2[:, b] == 0)
                            assert np.all((states1 ^ states2)[labels == 1, b] == 255)
                            assert np.all((states1 ^ states2)[labels == 0, b] == 0)
                else:
                    for b in range(16):
                        if b in [0, 1, 4, 5, 8, 9, 12, 13]:
                            if sign == "positive":
                                assert np.all(states1[:, b] == 0)
                            if sign == "negative":
                                assert np.all(states2[:, b] == 0)
                            assert np.all((states1 ^ states2)[labels == 1, b] == 255)
                            assert np.all((states1 ^ states2)[labels == 0, b] == 0)