// Copyright CEA (Commissariat à l'énergie atomique et aux
// énergies alternatives) (2017-2025)
//
// This software is governed by the CeCILL  license under French law and
// abiding by the rules of distribution of free software.  You can  use,
// modify and/ or redistribute the software under the terms of the CeCILL
// license as circulated by CEA, CNRS and INRIA at the following URL
// "http://www.cecill.info".
//
// As a counterpart to the access to the source code and  rights to copy,
// modify and redistribute granted by the license, users are provided only
// with a limited warranty  and the software's author,  the holder of the
// economic rights,  and the successive licensors  have only  limited
// liability.
//
// In this respect, the user's attention is drawn to the risks associated
// with loading,  using,  modifying and/or developing or reproducing the
// software by the user in light of its specific status of free software,
// that may mean  that it is complicated to manipulate,  and  that  also
// therefore means  that it is reserved for developers  and  experienced
// professionals having in-depth computer knowledge. Users are therefore
// encouraged to load and test the software's suitability as regards their
// requirements in conditions enabling the security of their systems and/or
// data to be ensured and,  more generally, to use and operate it in the
// same conditions as regards security.
//
// The fact that you are presently reading this means that you have had
// knowledge of the CeCILL license and that you accept its terms.

use pyo3::{exceptions::PyException, prelude::*};

pyo3::create_exception!(secbench_ffi, ShapeException, PyException, "Incorrect shape");

impl ShapeException {
    pub fn from_expected_shapes(expected: &[usize], got: &[usize]) -> PyErr {
        Self::new_err(format!("Expected shape {:?}, but got {:?}", expected, got))
    }
}

/// Macro to assert that the given array matches a specific shape
///
/// ## Usage
///
/// To check if an array2 has the correct amount of columns (axis 1) but we don't care amount the
/// amount of rows (axis 0):
/// ```rust
/// let data = Array2::zeros([10, 200]);
/// assert_shape_match!([_, 200] => data);
/// ```
#[macro_export]
macro_rules! assert_shape_match {
    ([$($vals:tt)*] => $target:expr) => {
        let target = $target.shape();
        if assert_shape_match!(@expand_check [target, 0] $($vals)*,) {
            return Err($crate::errors::ShapeException::from_expected_shapes(&assert_shape_match!(@expand_expected [target, 0, []] $($vals)*,), target));
        }
    };

    // Parse expected
    // if _
    (@expand_expected [$target:expr, $i:expr, [$($expected:expr),*]] _, $($rest:tt)*) => {
        assert_shape_match!(@expand_expected [$target, $i+1, [$($expected,)* $target[$i]]] $($rest)*)
    };
    // if value
    (@expand_expected [$target:expr, $i:expr, [$($expected:expr),*]] $val:expr, $($rest:tt)*) => {
        assert_shape_match!(@expand_expected [$target, $i+1, [$($expected,)* $val]] $($rest)*)
    };
    // terminate recursion
    (@expand_expected [$t:expr, $i:expr, [$($expected:expr),*]]) => {
        [$($expected),*]
    };

    // Emit if condition
    // Case for _
    (@expand_check [$target:expr, $i:expr] _, $($rest:tt)*) => {
        assert_shape_match!(@expand_check [$target, $i+1] $($rest)*)
    };
    // Case for expr
    (@expand_check [$target:expr, $i:expr] $val:expr, $($rest:tt)*) => {
        ($target[$i] != $val) || assert_shape_match!(@expand_check [$target, $i+1] $($rest)*)
    };
    // terminal case
    (@expand_check [$t:expr, $i:expr]) => { false };
}