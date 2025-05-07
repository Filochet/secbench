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

from secbench.processing import secbench_native_installed
from secbench.processing.crypto import Pcg32

secbench_ffi_testcase = pytest.mark.skipif(
    not secbench_native_installed(), reason="secbench_ffi package is not installed"
)


@secbench_ffi_testcase
@pytest.mark.parametrize("state", [10, 24, 0xDEADAAA])
@pytest.mark.parametrize("inc", [98, 111, 3])
def test_pcg32(state, inc):
    rng = Pcg32(state, inc)
    samples = np.array([rng.generate() for _ in range(10)], dtype=np.uint64)

    rng_2 = Pcg32(state, inc)
    dst = np.zeros(10, dtype=np.uint64)
    rng_2.fill(dst)
    np.testing.assert_equal(samples, dst)