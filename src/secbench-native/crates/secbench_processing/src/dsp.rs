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

use std::iter::Sum;
use std::ops::AddAssign;

use crate::assert_shape_match;
use num_traits::AsPrimitive;
use numpy::{Element, PyArray1, PyArray2, PyArray3, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use secbench_dsp::fft::{
    FilterSinglePass, FilterState, FilterTwoPass, PhaseCorrelation, RFftMag, TransformState,
};
use secbench_dsp::sliding::{MatchCorrelation, MatchEuclidean, MovingSum, SlidingExecutor, SlidingType};
use secbench_dsp::{DspFloat, IntoFloat, Transform2D};

/// Wrapper for running a Transform2D in many different configurations.
///
/// Configurations supported are: inplace/not inplace, and parallel/not parallel.
pub fn run_transform<'py, T, Dst, Src>(
    transform: &mut T,
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    T: Transform2D<Dst, Src>,
    Src: Element,
    Dst: Element,
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    if !parallel {
        match output {
            None => {
                let o_array = transform.apply_2d(i_array);
                Ok(o_array.to_pyarray_bound(input.py()))
            }
            Some(o_array) => {
                let mut dst = o_array.readwrite();
                let dst = dst.as_array_mut();
                transform.apply_2d_inplace(dst, i_array);
                Ok(o_array)
            }
        }
    } else {
        match output {
            None => {
                let o_array = transform.apply_2d_parallel(i_array, chunk_size);
                Ok(o_array.to_pyarray_bound(input.py()))
            }
            Some(o_array) => {
                let mut dst = o_array.readwrite();
                let dst = dst.as_array_mut();
                transform.apply_2d_inplace_parallel(dst, i_array, chunk_size);
                Ok(o_array)
            }
        }
    }
}

// ====
// Moving sum bindings.
// ====
pub fn generic_moving_sum<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    parallel: bool,
    chunk_size: Option<usize>,
    window_size: usize,
    scale: Dst,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send,
    Dst: Element + DspFloat + 'static + Sync + Send,
{
    let mut ms: MovingSum<Dst, Src> = MovingSum::new(window_size, scale);
    run_transform(&mut ms, output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size, window_size, scale))]
pub fn moving_sum_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    parallel: bool,
    chunk_size: Option<usize>,
    window_size: usize,
    scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_moving_sum(output, input, parallel, chunk_size, window_size, scale)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size, window_size, scale))]
pub fn moving_sum_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    parallel: bool,
    chunk_size: Option<usize>,
    window_size: usize,
    scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_moving_sum(output, input, parallel, chunk_size, window_size, scale)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size, window_size, scale))]
pub fn moving_sum_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
    window_size: usize,
    scale: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_moving_sum(output, input, parallel, chunk_size, window_size, scale)
}

// ====
// Filter bindings.
// ====
pub fn generic_filter<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    kernel: Bound<'py, PyArray1<Dst>>,
    parallel: bool,
    chunk_size: Option<usize>,
    two_pass: bool,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send,
    Dst: Element + DspFloat + 'static + Sync + Send,
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    let mut s: FilterState<Dst, Src> = FilterState::new(i_array.ncols());
    s.load_kernel(kernel.readonly().as_slice().unwrap());
    if two_pass {
        let mut tr: FilterTwoPass<Dst, Src> = FilterTwoPass(s);
        run_transform(&mut tr, output, input, parallel, chunk_size)
    } else {
        let mut tr: FilterSinglePass<Dst, Src> = FilterSinglePass(s);
        run_transform(&mut tr, output, input, parallel, chunk_size)
    }
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size, two_pass))]
pub fn fft_filter_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
    two_pass: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_filter(output, input, kernel, parallel, chunk_size, two_pass)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size, two_pass))]
pub fn fft_filter_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
    two_pass: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_filter(output, input, kernel, parallel, chunk_size, two_pass)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size, two_pass))]
pub fn fft_filter_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
    two_pass: bool,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_filter(output, input, kernel, parallel, chunk_size, two_pass)
}

// ====
// Phase correlation.
// ====
pub fn generic_phase_correlation<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    kernel: Bound<'py, PyArray1<Dst>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send,
    Dst: Element + DspFloat + 'static + Sync + Send,
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    let mut s: FilterState<Dst, Src> = FilterState::new(i_array.ncols());
    s.load_kernel(kernel.readonly().as_slice().unwrap());
    let mut tr: PhaseCorrelation<Dst, Src> = PhaseCorrelation(s);
    run_transform(&mut tr, output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn phase_correlation_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_phase_correlation(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn phase_correlation_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_phase_correlation(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn phase_correlation_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_phase_correlation(output, input, kernel, parallel, chunk_size)
}

// ===
// FFT Magnitude
// ===

pub fn generic_rfft_mag<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send,
    Dst: Element + DspFloat + 'static + Sync + Send,
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    let mut tr: RFftMag<Dst, Src> = RFftMag(TransformState::new(i_array.ncols()));
    run_transform(&mut tr, output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size))]
pub fn rfft_mag_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_rfft_mag(output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size))]
pub fn rfft_mag_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_rfft_mag(output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, *, parallel, chunk_size))]
pub fn rfft_mag_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_rfft_mag(output, input, parallel, chunk_size)
}

// ====
// Sliding statistics
// ====

pub fn generic_sliding_x<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    parallel: bool,
    chunk_size: Option<usize>,
    window_size: usize,
    padding_value: Option<Dst>,
    st: SlidingType,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send + AddAssign,
    Dst: Element + DspFloat + 'static + Sync + Send,
{
    let mut sx: SlidingExecutor<Dst, Src> = SlidingExecutor::new(st, window_size, padding_value);
    run_transform(&mut sx, output, input, parallel, chunk_size)
}

macro_rules! def_sliding {
    ($fn_name:ident, $sliding_type:expr, $Src:ty => $($Dst:ty),*) => { $(
        #[pyfunction]
        #[pyo3(signature = (output, input, *, parallel, chunk_size, window_size, padding_value))]
        pub fn $fn_name<'py>(
            output: Option<Bound<'py, PyArray2<$Dst>>>,
            input: Bound<'py, PyArray2<$Src>>,
            parallel: bool,
            chunk_size: Option<usize>,
            window_size: usize,
            padding_value: Option<$Dst>,
        ) -> PyResult<Bound<'py, PyArray2<$Dst>>> {
            generic_sliding_x(
                output,
                input,
                parallel,
                chunk_size,
                window_size,
                padding_value,
                $sliding_type,
            )
        }
    )* };
}

def_sliding!(sliding_mean_f32_i8, SlidingType::Mean, i8 => f32);
def_sliding!(sliding_mean_f32_i16, SlidingType::Mean, i16 => f32);
def_sliding!(sliding_mean_f32_f32, SlidingType::Mean, f32 => f32);
def_sliding!(sliding_mean_f64_f64, SlidingType::Mean, f64 => f64);

def_sliding!(sliding_var_f32_i8, SlidingType::Var, i8 => f32);
def_sliding!(sliding_var_f32_i16, SlidingType::Var, i16 => f32);
def_sliding!(sliding_var_f32_f32, SlidingType::Var, f32 => f32);
def_sliding!(sliding_var_f64_f64, SlidingType::Var, f64 => f64);

def_sliding!(sliding_std_f32_i8, SlidingType::Std, i8 => f32);
def_sliding!(sliding_std_f32_i16, SlidingType::Std, i16 => f32);
def_sliding!(sliding_std_f32_f32, SlidingType::Std, f32 => f32);
def_sliding!(sliding_std_f64_f64, SlidingType::Std, f64 => f64);

def_sliding!(sliding_skew_f32_i8, SlidingType::Skew, i8 => f32);
def_sliding!(sliding_skew_f32_i16, SlidingType::Skew, i16 => f32);
def_sliding!(sliding_skew_f32_f32, SlidingType::Skew, f32 => f32);
def_sliding!(sliding_skew_f64_f64, SlidingType::Skew, f64 => f64);

def_sliding!(sliding_kurt_f32_i8, SlidingType::Kurt, i8 => f32);
def_sliding!(sliding_kurt_f32_i16, SlidingType::Kurt, i16 => f32);
def_sliding!(sliding_kurt_f32_f32, SlidingType::Kurt, f32 => f32);
def_sliding!(sliding_kurt_f64_f64, SlidingType::Kurt, f64 => f64);


// ====
// Euclidean pattern matching
// ====
pub fn generic_match_euclidean<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    kernel: Bound<'py, PyArray1<Dst>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + Copy + Sync + Send,
    Dst: Element + DspFloat + 'static + AsPrimitive<Dst> + Sum + From<u8> + Sync + Send,
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    let mut tr: MatchEuclidean<Dst, Src> = MatchEuclidean::new(kernel.readonly().as_slice().unwrap(), i_array.ncols());
    run_transform(&mut tr, output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_euclidean_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_euclidean(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_euclidean_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_euclidean(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_euclidean_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_euclidean(output, input, kernel, parallel, chunk_size)
}

// ====
// Correlation pattern matching
// ====
pub fn generic_match_correlation<'py, Dst, Src>(
    output: Option<Bound<'py, PyArray2<Dst>>>,
    input: Bound<'py, PyArray2<Src>>,
    kernel: Bound<'py, PyArray1<Dst>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<Dst>>>
where
    Src: Element + AsPrimitive<Dst> + AddAssign + Copy + Sync + Send,
    Dst: Element + DspFloat + Sum + 'static + AsPrimitive<Dst> + From<u8> + Sync + Send,
    usize: AsPrimitive<Dst>
{
    let i_array = input.readonly();
    let i_array = i_array.as_array();
    let mut tr: MatchCorrelation<Dst, Src> = MatchCorrelation::new(kernel.readonly().as_slice().unwrap(), i_array.ncols());
    run_transform(&mut tr, output, input, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_correlation_i8<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i8>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_correlation(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_correlation_i16<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<i16>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_correlation(output, input, kernel, parallel, chunk_size)
}

#[pyfunction]
#[pyo3(signature = (output, input, kernel, *, parallel, chunk_size))]
pub fn match_correlation_f32<'py>(
    output: Option<Bound<'py, PyArray2<f32>>>,
    input: Bound<'py, PyArray2<f32>>,
    kernel: Bound<'py, PyArray1<f32>>,
    parallel: bool,
    chunk_size: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    generic_match_correlation(output, input, kernel, parallel, chunk_size)
}

type F = f64;

#[pyclass]
pub struct CondMeanVar {
    inner: secbench_dsp::CondMeanVar<F>,
}

impl CondMeanVar {
    fn process_block_inner<I>(
        &mut self,
        data: Bound<PyArray2<I>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()>
    where
        I: IntoFloat<F> + Element + Copy + 'static,
    {
        let data = data.readonly();
        let data = data.as_array();
        let labels = labels.readonly();
        let labels = labels.as_array();
        assert_shape_match!([labels.shape()[0], _] => data);
        self.inner.process_block(data, labels);
        Ok(())
    }
}

#[pymethods]
impl CondMeanVar {
    #[new]
    pub fn new(targets: usize, samples: usize, classes: usize) -> Self {
        let inner = secbench_dsp::multi_condmean::CondMeanVar::<F>::new(targets, samples, classes);
        Self { inner }
    }

    pub fn load(
        &mut self,
        mean: Bound<PyArray3<F>>,
        var: Bound<PyArray3<F>>,
        samples: Bound<PyArray2<u32>>,
    ) {
        let m = mean.readonly();
        let v = var.readonly();
        let s = samples.readonly();
        self.inner
            .load_state(m.as_array(), v.as_array(), s.as_array());
    }

    pub fn save<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray3<F>>,
        Bound<'py, PyArray3<F>>,
        Bound<'py, PyArray2<u32>>,
    )> {
        let (mean, var, samples) = self.inner.dump_state();
        Ok((
            mean.to_pyarray_bound(py),
            var.to_pyarray_bound(py),
            samples.to_pyarray_bound(py),
        ))
    }

    pub fn process_block_i8(
        &mut self,
        data: Bound<PyArray2<i8>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_i16(
        &mut self,
        data: Bound<PyArray2<i16>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_f32(
        &mut self,
        data: Bound<PyArray2<f32>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_f64(
        &mut self,
        data: Bound<PyArray2<f64>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn freeze_mean_var<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray3<F>>, Bound<'py, PyArray3<F>>)> {
        let (mean, var) = self.inner.freeze();
        Ok((mean.to_pyarray_bound(py), var.to_pyarray_bound(py)))
    }

    pub fn freeze_samples_per_class<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        Ok(self.inner.samples_per_class().to_pyarray_bound(py))
    }

    pub fn freeze_global_mean_var<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<F>>, Bound<'py, PyArray1<F>>, u32)> {
        let (mean, var, samples) = self.inner.freeze_global_mean_var();
        Ok((mean.to_pyarray_bound(py), var.to_pyarray_bound(py), samples))
    }

    pub fn split<'py>(
        &self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<Bound<'py, CondMeanVarP>> {
        let obj = CondMeanVarP {
            inner: secbench_dsp::CondMeanVarP::<F>::split(&self.inner, chunk_size),
        };
        Bound::new(py, obj)
    }
}

#[pyclass]
pub struct CondMeanVarP {
    inner: secbench_dsp::CondMeanVarP<F>,
}

impl CondMeanVarP {
    fn process_block_inner<I>(
        &mut self,
        data: Bound<PyArray2<I>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()>
    where
        I: IntoFloat<F> + Element + Copy + 'static + Sync + Send,
    {
        let data = data.readonly();
        let data = data.as_array();
        let labels = labels.readonly();
        let labels = labels.as_array();
        assert_shape_match!([labels.shape()[0], _] => data);
        self.inner.process_block(data, labels);
        Ok(())
    }
}

#[pymethods]
impl CondMeanVarP {
    #[new]
    pub fn new(chunk_size: usize, targets: usize, samples: usize, classes: usize) -> Self {
        Self {
            inner: secbench_dsp::CondMeanVarP::<F>::new(chunk_size, targets, samples, classes),
        }
    }

    pub fn merge<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, CondMeanVar>> {
        Bound::new(
            py,
            CondMeanVar {
                inner: self.inner.merge(),
            },
        )
    }

    pub fn process_block_i8(
        &mut self,
        data: Bound<PyArray2<i8>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_i16(
        &mut self,
        data: Bound<PyArray2<i16>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_f32(
        &mut self,
        data: Bound<PyArray2<f32>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }

    pub fn process_block_f64(
        &mut self,
        data: Bound<PyArray2<f64>>,
        labels: Bound<PyArray2<u16>>,
    ) -> PyResult<()> {
        self.process_block_inner(data, labels)
    }
}