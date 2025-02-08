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

import contextlib
import time

import pint
import pytest

u = pint.UnitRegistry()


@pytest.fixture
def bench():
    from secbench.api import get_bench

    with get_bench() as bench:
        yield bench


@pytest.fixture
def table(bench):
    with bench.get_table() as table:
        yield table


@pytest.fixture()
def scope_dut():
    from secbench.scope.util import ScopeHifiveDut

    all_duts = [ScopeHifiveDut]
    for dut_cls in all_duts:
        try:
            dut = dut_cls()
            yield dut
            dut.close()
        except Exception:
            pass
    return None


@pytest.fixture
def benchmark():
    @contextlib.contextmanager
    def bencher(msg="", n=None, unit="unit"):
        t = time.monotonic()
        yield
        duration = time.monotonic() - t
        duration = (u.second * duration).to_compact()
        if n:
            print(msg, "duration: ", duration, "that is", duration / n, "per", unit)
        else:
            print(msg, "duration: ", duration)

    return bencher


def pytest_generate_tests(metafunc):
    if "horizontal_args" in metafunc.fixturenames:
        module = metafunc.module.__name__
        if "ps2000a" in module:
            params = [
                # interval, samples
                (1 * u.us, 1e4),
                (10 * u.us, 1e3),
                (10 * u.us, 5e3),
                (10 * u.us, 10e3),
                (100 * u.us, 1e2),
            ]
        elif "rohdeschwartz" in module:
            params = [
                # interval, samples
                (100 * u.us, 10e6),
                (1 * u.us, 10e6),
                (100 * u.ns, 10e6),
                (10 * u.ns, 10e6),
                (10 * u.ns, 10e6),
                (1 * u.ns, 4e6),
                (1 * u.ns, 1e6),
                (1 * u.ns, 1e3),
                (100 * u.ps, 1e3),
            ]
        elif "lecroy" in module:
            params = [
                # interval, samples
                (100 * u.us, 10e6),
                (1 * u.us, 10e6),
                (100 * u.ns, 10e6),
                (10 * u.ns, 10e6),
                (10 * u.ns, 10e6),
                (1 * u.ns, 5e6),
                (1 * u.ns, 1e6),
                (1 * u.ns, 1e3),
                (100 * u.ps, 1e3),
            ]
        else:
            raise NotImplementedError(
                "no sensible horizontal params for {}".format(module)
            )

        def gen_params():
            for interval, samples in params:
                duration = (interval * samples).to(u.s).magnitude
                interval = interval.to(u.s).magnitude
                yield (interval, samples, None)
                yield (interval, None, duration)
                yield (None, samples, duration)

        metafunc.parametrize("horizontal_args", gen_params())