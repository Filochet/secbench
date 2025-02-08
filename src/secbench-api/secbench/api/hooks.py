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
"""
A collection of post-experiments hooks.
"""

import subprocess

from .bench import Bench


def _invoke_hook(hook, *args):
    if not hook:
        return
    if isinstance(hook, list):
        for h in hook:
            h(*args)
        return
    hook(*args)


def secbench_main(on_success=None, on_failure=None, exit_on_failure=True):
    """
    A Decorator for secbench experiments.

    Decorating running your experiment with ``secbench_main``, has a few advantages:

    - It will automatically create a :py:class:`secbench.core.bench.Bench` and
      pass it as first argument to your function.
    - ``secbench_main`` performs some clean-up and allows to supply
      post-experiment actions, like sending you an e-mail or copying results
      (see the example below).

    :param on_success: An optional hook called if the experiment runs
                       successfully
    :param on_failure: An optional hook called if the experiment failed
    :param exit_on_failure: If true, will invoke sys.exit(1) on error

    :Examples:

    Here is a simple example of using the ``secbench_main`` decorator.

    .. code-block:: python

        from secbench.api.hooks import secbench_main, NotifySendHook

        # A custom hook to copy data into a shared directory. You can do
        # anything there.
        def backup_results(results):
            subprocess.check_call(['cp', 'my-store', '/home/...'])

        # Apply the secbench_main decorator to your function -> the argument
        # 'bench' will be passed automatically.
        @secbench_main(on_success=backup_results,
                       on_failure=NotifySendHook("Something went wrong..."))
        def my_experiment(bench):
            # ...
            return results


        if __name__ == '__main__':
            my_experiment()

    """
    import sys
    import traceback

    def inner(f):
        def _wrapped():
            try:
                bench = Bench()
                result = f(bench)
                _invoke_hook(on_success, result)
                return result
            except Exception as e:
                _invoke_hook(on_failure)
                if exit_on_failure:
                    msg = "\n".join(
                        [
                            "the script failed because of an uncatched exception.",
                            f"Exception:\n    {repr(e)} {e}\n",
                            "Backtrace:",
                        ]
                    )
                    print(msg, file=sys.stderr)
                    traceback.print_tb(e.__traceback__, file=sys.stderr)
                    sys.exit(1)
                raise

        return _wrapped

    return inner


class NotifySendHook:
    """
    Open a notification pop-up with a message on your windows manager.

    See ``man notify-send``.
    """

    def __init__(self, message: str, urgency="critical"):
        self.message = message
        self.urgency = urgency

    def __call__(self, *args, **kwargs):
        subprocess.check_call(["notify-send", "-u", self.urgency, self.message])