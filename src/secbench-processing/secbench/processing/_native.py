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
Re-exports from :py:mod:`secbench_native.processing` module
"""

from __future__ import annotations

import functools
import os

import numpy as np

from .helpers import InvalidInputError, MissingPackageError, check_array

try:
    import secbench_native.processing as secbench_native_processing
except ImportError:
    secbench_native_processing = None


def _native_not_available(symbol: str):
    def wrapped(*args, **kwargs):
        """
        Stub for :py:mod:`secbench_native.processing` module.

        .. warning::

            If you see this message, it means that the ``secbench_native.processing`` module is
            not available. You need to install it to access this functionality.
        """
        raise MissingPackageError(
            f"Unable to use '{symbol}', the 'secbench_native.processing' package must be installed."
        )

    return wrapped


def secbench_native_import(symbol: str):
    """
    Safely re-export a function defined in the :py:mod:`secbench_native.processing` module.

    This will generate a user-friendly message if someone
    tries to use (import is allowed) a feature from secbench_native without
    having it installed.
    """
    if secbench_native_processing is None:
        return _native_not_available(symbol)
    return getattr(secbench_native_processing, symbol)


_N_THREADS = os.environ.get("RAYON_NUM_THREADS") or os.cpu_count()


def _chunk_size(n_rows, chunks_per_thread=2.0) -> int | None:
    """
    Compute the default chunk size for
    """
    total = int(n_rows / (chunks_per_thread * _N_THREADS))
    if total == 0:
        return None
    return total


def transform_2d(input_types, output_types, allow_1d_inputs=True, c_continuous=True):
    def wrap(f):
        @functools.wraps(f)
        def inner(
            X,
            *args,
            output=None,
            parallel=False,
            chunk_size=None,
            dtype=None,
            **kwargs,
        ):
            if dtype is None:
                if output is None:
                    if X.dtype == np.float64:
                        dtype = np.float64
                    else:
                        dtype = np.float32
                else:
                    dtype = output.dtype

            if dtype is not None and dtype not in output_types:
                raise InvalidInputError(
                    f"unsupported output type {dtype}, supported values are: {output_types}"
                )
            input_is_1d = X.ndim == 1
            if allow_1d_inputs and input_is_1d:
                X = X[np.newaxis, :]
            if output is not None:
                check_array(
                    output,
                    ndim=2,
                    dtype=dtype,
                    array_name="output",
                    check_c_continuous=c_continuous,
                )
            check_array(
                X,
                ndim=2,
                dtype=input_types,
                array_name="X",
                check_c_continuous=c_continuous,
            )
            # Compute default chunk size if needed.
            if chunk_size is None and parallel:
                chunk_size = _chunk_size(X.shape[0])

            out = f(
                X,
                *args,
                output=output,
                parallel=parallel,
                chunk_size=chunk_size,
                dtype=dtype,
                **kwargs,
            )

            if allow_1d_inputs and input_is_1d:
                return out[0]
            return out

        return inner

    return wrap