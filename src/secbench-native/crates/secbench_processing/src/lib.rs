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

use pyo3::prelude::*;
use tracing::level_filters::LevelFilter;

use crate::errors::ShapeException;

mod crypto;
mod dsp;
mod errors;

pub fn make_secbench_processing(py: Python) -> PyResult<Bound<PyModule>> {
    let m = PyModule::new_bound(py, "processing")?;

    // Forward logging to caller's stderr.
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::INFO)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Errors
    m.add("ShapeException", py.get_type_bound::<ShapeException>())?;


    // Dsp
    m.add_class::<dsp::CondMeanVar>()?;
    m.add_class::<dsp::CondMeanVarP>()?;
    m.add_function(wrap_pyfunction!(dsp::moving_sum_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::moving_sum_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::moving_sum_f32, &m)?)?;
    
    m.add_function(wrap_pyfunction!(dsp::fft_filter_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::fft_filter_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::fft_filter_f32, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::phase_correlation_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::phase_correlation_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::phase_correlation_f32, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::rfft_mag_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::rfft_mag_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::rfft_mag_f32, &m)?)?;
    
    m.add_function(wrap_pyfunction!(dsp::match_euclidean_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::match_euclidean_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::match_euclidean_f32, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::match_correlation_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::match_correlation_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::match_correlation_f32, &m)?)?;


    // Add Pcg32
    m.add_class::<crypto::Pcg32>()?;

    m.add_function(wrap_pyfunction!(dsp::sliding_mean_f32_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_mean_f32_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_mean_f32_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_mean_f64_f64, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::sliding_var_f32_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_var_f32_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_var_f32_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_var_f64_f64, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::sliding_std_f32_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_std_f32_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_std_f32_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_std_f64_f64, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::sliding_skew_f32_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_skew_f32_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_skew_f32_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_skew_f64_f64, &m)?)?;

    m.add_function(wrap_pyfunction!(dsp::sliding_kurt_f32_i8, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_kurt_f32_i16, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_kurt_f32_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(dsp::sliding_kurt_f64_f64, &m)?)?;

    Ok(m)
}