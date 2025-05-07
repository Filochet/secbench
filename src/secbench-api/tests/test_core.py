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

import pytest

from secbench.api import Bench
from secbench.api.hooks import secbench_main


def test_version():
    # Check that version actually works
    from secbench.api import version

    assert version() is not None


def success_hook(r):
    print(f"Success: {r}")


def failure_hook():
    print("Failed")


@secbench_main(on_success=success_hook)
def example_main(bench: Bench):
    """
    Demonstrates how you can wrap a function with secbench_main
    """
    assert isinstance(bench, Bench)
    return 3


@secbench_main(on_failure=failure_hook, exit_on_failure=False)
def example_main_fail(_bench: Bench):
    raise ValueError()


def test_secbench_main(capsys):
    r = example_main()
    captured = capsys.readouterr()
    assert r == 3
    assert "Success: 3" in captured.out
    assert "Failed" not in captured.out

    with pytest.raises(ValueError):
        example_main_fail()
    captured = capsys.readouterr()
    assert "Failed" in captured.out


def test_kwonly_dataclass_compat():
    import sys

    from secbench.api.instrument.pulser import EMPulseParams

    if sys.version_info < (3, 10):
        _ = EMPulseParams(10, 2, 3, 3)
    else:
        # kw_only must be enabled for python>3.10
        with pytest.raises(TypeError):
            _ = EMPulseParams(10, 2, 3, 3)

    _ = EMPulseParams(delay_ns=10, width_ns=3, rise_time_ns=1, amplitude=10)