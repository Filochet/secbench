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

MAX_ROUNDS = 18
# fmt: off
PI_P_MAT = np.array([0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13, 19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21], dtype=np.uint8)
# fmt: on

RC = np.array(
    [
        0x0000000000000001,
        0x0000000000008082,
        0x800000000000808A,
        0x8000000080008000,
        0x000000000000808B,
        0x0000000080000001,
        0x8000000080008081,
        0x8000000000008009,
        0x000000000000008A,
        0x0000000000000088,
        0x0000000080008009,
        0x000000008000000A,
        0x000000008000808B,
        0x800000000000008B,
        0x8000000000008089,
        0x8000000000008003,
        0x8000000000008002,
        0x8000000000000080,
        0x000000000000800A,
        0x800000008000000A,
        0x8000000080008081,
        0x8000000000008080,
        0x0000000080000001,
        0x8000000080008008,
    ],
    dtype=np.uint64,
)

# fmt: off
RHO_OFFSETS = np.array(
    [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,],
    dtype=np.uint8,
)
# fmt: on


class Keccak_f:
    """
    This class allows to create an instance of the Keccak
    permutation function.
    For more information, you can refer to https://keccak.team/files/Keccak-implementation-3.2.pdf
    .. note::
        Versions Keccak_f[x], for x in `[25, 50, 100]` are not currently supported.
    """

    ALLOWED_VERSIONS = [200, 400, 800, 1600]

    def __init__(self, b):
        if b not in self.ALLOWED_VERSIONS:
            raise ValueError(
                f"Keccak-f[{b}] does not exist, please provide version from {self.ALLOWED_VERSIONS}"
            )

        self._w = b // 25
        self._nrounds = 12 + 2 * int(np.log2(self._w))
        # Compute the table of round constants
        self._dtype = f">u{self._w // 8}"
        self._RC = (RC & ((1 << self._w) - 1)).astype(self._dtype)
        # Compute the RHO_OFFSETS table
        self._r = RHO_OFFSETS % self._w

    @staticmethod
    def rot(states, offset, w):
        """
        Elementwise bit roll of given offset.
        :param states: (N,) Keccak states.
        :param offset: left rolling offset.
        :return: The updated states
        """
        return (states << offset) ^ (states >> w - offset)

    def theta(self, states: np.ndarray):
        """
        Keccak Theta fuction.
        :param states: (N, 25) Keccak state.
        :return: The updated states.
        """
        states = states.reshape(states.shape[0], 5, 5)
        # Compute parity of columns
        C = np.bitwise_xor.reduce(states, axis=-2)
        C1 = np.roll(C, shift=-1, axis=-1)
        C4 = np.roll(C, shift=-4, axis=-1)
        D = self.rot(C1, 1, self._w) ^ C4
        D = np.repeat(D, 5, axis=-2).reshape(states.shape[0], 5, 5)
        states = states ^ D
        return states.reshape(states.shape[0], 25)

    def rho(self, states: np.ndarray):
        """
        Numpy implementation of keccak Rho fuction.
        :param states: (N, 25) Keccak state.
        :return: The updated states.
        """
        return self.rot(states, self._r, self._w)

    @staticmethod
    def pi(states: np.ndarray):
        """
        Keccak Pi fuction.
        Uses a precomputed permutation matrix.
        :param states: (N, 25) Keccak state.
        :return: The updated states.
        """
        return states[:, PI_P_MAT]

    @staticmethod
    def chi(states: np.ndarray):
        """
        Keccak Chi fuction.
        :param states: (N, 25) Keccak state.
        :return: The updated states.
        """
        states = states.reshape(states.shape[0], 5, 5)
        A1 = np.roll(states, -1, axis=-1)
        A2 = np.roll(states, -2, axis=-1)
        states = states ^ (~A1 & A2)
        return states.reshape(states.shape[0], 25)

    def iota(self, states: np.ndarray, round_index: int):
        """
        keccak Iota fuction.
        :param states: (N, 25) Keccak state.
        :param round_index: the round index.
        :return: The updated states.
        """
        states[:, 0] ^= self._RC[round_index]
        return states

    def round(self, states: np.ndarray, round_index: int):
        """
        One keccak round.
        :param states: (N, 25) Keccak state.
        :param round_index: the round index.
        :return: The updated states.
        """
        states = self.theta(states)
        states = self.rho(states)
        states = self.pi(states)
        states = self.chi(states)
        states = self.iota(states, round_index)
        return states

    def permutation(self, states: np.ndarray):
        """
        Perform Keccak-f cryptographic permutation over
        an numpy array of messages.
        :param states: (N, 25) Keccak state.
        :return: The updated states.
        - Example::
            >>> kec = Keccak_f(100)
            >>> states = np.random.randint(0, 256, size=(2, 25), dtype=np.uint8)
            >>> states
            array([[244, 237, 146, 136, 194, 119, 230,  20,  38, 153, 174,  61, 167,
                    242, 195, 179,   8,   8, 136,  17, 205, 246,   3, 170, 138],
                   [ 58,  11, 187, 245, 222, 188,  53, 201, 253, 243, 189, 249,  92,
                    101,  85,  40, 249,  90, 163,  52,   6,  12, 171, 222, 127]],
                  dtype=uint8)
            >>> kec.permutation(states)
            array([[ 41,  65, 240,  43,  90, 191, 154,  77,  96, 226,  90,  29, 231,
                    175, 191, 227, 209,  75, 126, 230, 237, 185, 198,  91, 166],
                   [ 63, 140, 202, 213,  82, 102, 207,  20, 201,  81, 243,  22, 107,
                    233, 116,  81,  64, 106, 110,  44, 177,  10,  56,  49, 220]],
                  dtype=uint8)
        """
        for i in range(self._nrounds):
            states = self.round(states, i)
        return states