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

use crate::{DspFloat, Transform1D};
use num_traits::AsPrimitive;
use realfft::{num_complex::Complex, ComplexToReal, RealFftPlanner, RealToComplex};
use std::marker::PhantomData;
use std::sync::Arc;

#[derive(Clone)]
pub struct FFTSharedData<T> {
    pub forward: Arc<dyn RealToComplex<T>>,
    pub inverse: Arc<dyn ComplexToReal<T>>,
    pub fft_len: usize,
}

impl<T> FFTSharedData<T>
where
    T: DspFloat,
{
    pub fn new(len: usize) -> Self {
        let mut planner = RealFftPlanner::new();
        let forward = planner.plan_fft_forward(len);
        Self {
            forward,
            inverse: planner.plan_fft_inverse(len),
            fft_len: len,
        }
    }

    pub fn make_output_vec(&self) -> Vec<Complex<T>> {
        self.forward.make_output_vec()
    }

    pub fn make_scratch_vec(&self) -> Vec<Complex<T>> {
        self.forward.make_scratch_vec()
    }
}

#[derive(Clone)]
pub struct FilterState<Dst, Src> {
    ctx: FFTSharedData<Dst>,
    input_data: Vec<Dst>,
    filter_kernel: Vec<Complex<Dst>>,
    fft_scratch: Vec<Complex<Dst>>,
    fft_tr_output: Vec<Complex<Dst>>,
    _src: PhantomData<Src>,
}

impl<Dst, Src> FilterState<Dst, Src>
where
    Dst: DspFloat,
{
    pub fn new(fft_len: usize) -> Self {
        let ctx = FFTSharedData::new(fft_len);
        let input_data = vec![Dst::zero(); fft_len];
        let fft_scratch = ctx.make_scratch_vec();
        let fft_tr_output = ctx.make_output_vec();
        let filter_kernel = ctx.make_output_vec();
        FilterState {
            ctx,
            input_data,
            filter_kernel,
            fft_scratch,
            fft_tr_output,
            _src: Default::default(),
        }
    }

    pub fn load_kernel(&mut self, coeffs: &[Dst]) {
        let fft_len = self.fft_len();
        assert!(coeffs.len() <= fft_len);

        // Zero padding of the input
        let mut kernel_in = Vec::with_capacity(fft_len);
        kernel_in.extend_from_slice(coeffs);
        kernel_in.resize(fft_len, Dst::zero());

        self.ctx
            .forward
            .process_with_scratch(
                &mut kernel_in,
                &mut self.filter_kernel,
                &mut self.fft_scratch,
            )
            .unwrap();
    }

    pub fn fft_len(&self) -> usize {
        self.ctx.fft_len
    }
}

impl<Dst, Src> FilterState<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn filter_input_data(&mut self, output: &mut [Dst]) {
        self.ctx
            .forward
            .process_with_scratch(
                &mut self.input_data,
                &mut self.fft_tr_output,
                &mut self.fft_scratch,
            )
            .unwrap();

        for (x, k) in self.fft_tr_output.iter_mut().zip(self.filter_kernel.iter()) {
            *x *= k;
        }

        // Move data back to time domain
        self.ctx
            .inverse
            .process_with_scratch(&mut self.fft_tr_output, output, &mut self.fft_scratch)
            .unwrap();

        // Normalize
        let norm_factor = Dst::from_usize(self.fft_len()).unwrap();
        for x in output {
            *x /= norm_factor;
        }
    }

    pub fn filter_single_pass(&mut self, output: &mut [Dst], input: &[Src]) {
        let fft_len = self.fft_len();
        debug_assert!(self.filter_kernel.len() > 0, "kernel must be initialized");
        debug_assert!(input.len() <= fft_len);
        debug_assert!(output.len() >= input.len());
        debug_assert!(output.len() >= fft_len);

        // Convert input data.
        self.input_data
            .iter_mut()
            .zip(input.iter())
            .for_each(|(x, y)| {
                *x = y.as_();
            });
        // Transform data in the frequency domain
        self.filter_input_data(output)
    }

    pub fn filter_two_pass(&mut self, output: &mut [Dst], input: &[Src]) {
        self.filter_single_pass(output, input);
        // Reverse output into self.input_data.
        self.input_data
            .iter_mut()
            .zip(output.iter().rev())
            .for_each(|(x, &y)| {
                *x = y;
            });
        self.filter_input_data(output);
        output.reverse();
    }

    /// Compute phase correlation of a filter with a given kernel.
    ///
    /// A good explanation is available on [Wikipedia](https://en.wikipedia.org/wiki/Phase_correlation).
    pub fn phase_correlation(&mut self, output: &mut [Dst], input: &[Src]) {
        let fft_len = self.fft_len();
        debug_assert!(self.filter_kernel.len() > 0, "kernel must be initialized");
        debug_assert!(output.len() >= input.len());
        debug_assert!(output.len() >= fft_len);

        // Convert input data.
        self.input_data
            .iter_mut()
            .zip(input.iter())
            .for_each(|(x, y)| {
                *x = y.as_();
            });

        self.ctx
            .forward
            .process_with_scratch(
                &mut self.input_data,
                &mut self.fft_tr_output,
                &mut self.fft_scratch,
            )
            .unwrap();
        for (x, k) in self.fft_tr_output.iter_mut().zip(self.filter_kernel.iter()) {
            *x *= k.conj();
            let norm = x.norm_sqr();
            if norm > Dst::zero() {
                *x /= norm.sqrt();
            }
        }
        // Move data back to time domain
        self.ctx
            .inverse
            .process_with_scratch(&mut self.fft_tr_output, output, &mut self.fft_scratch)
            .unwrap();

        // Normalize
        let norm_factor = Dst::from_usize(fft_len).unwrap();
        for x in output {
            *x /= norm_factor;
        }
    }
}

#[derive(Clone)]
pub struct FilterSinglePass<Dst, Src>(pub FilterState<Dst, Src>);

impl<Dst, Src> Transform1D<Dst, Src> for FilterSinglePass<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        self.0.filter_single_pass(output, input);
    }
}

#[derive(Clone)]
pub struct FilterTwoPass<Dst, Src>(pub FilterState<Dst, Src>);

impl<Dst, Src> Transform1D<Dst, Src> for FilterTwoPass<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        self.0.filter_two_pass(output, input);
    }
}

#[derive(Clone)]
pub struct PhaseCorrelation<Dst, Src>(pub FilterState<Dst, Src>);

impl<Dst, Src> Transform1D<Dst, Src> for PhaseCorrelation<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        self.0.phase_correlation(output, input);
    }
}

#[derive(Clone)]
pub struct TransformState<Dst, Src> {
    ctx: FFTSharedData<Dst>,
    input_data: Vec<Dst>,
    fft_scratch: Vec<Complex<Dst>>,
    fft_tr_output: Vec<Complex<Dst>>,
    _src: PhantomData<Src>,
}

impl<Dst, Src> TransformState<Dst, Src>
where
    Dst: DspFloat,
{
    pub fn new(fft_len: usize) -> Self {
        let ctx = FFTSharedData::new(fft_len);
        let input_data = vec![Dst::zero(); fft_len];
        let fft_scratch = ctx.make_scratch_vec();
        let fft_tr_output = ctx.make_output_vec();
        TransformState {
            ctx,
            input_data,
            fft_scratch,
            fft_tr_output,
            _src: Default::default(),
        }
    }

    pub fn fft_len(&self) -> usize {
        self.ctx.fft_len
    }

    pub fn rfft_len(&self) -> usize {
        self.fft_len() / 2 + 1
    }
}

impl<Dst, Src> TransformState<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    pub fn rfft_mag(&mut self, output: &mut [Dst], input: &[Src]) {
        let rfft_len = self.rfft_len();
        debug_assert!(output.len() >= rfft_len);
        // Convert input data.
        self.input_data
            .iter_mut()
            .zip(input.iter())
            .for_each(|(x, y)| {
                *x = y.as_();
            });

        self.ctx
            .forward
            .process_with_scratch(
                &mut self.input_data,
                &mut self.fft_tr_output,
                &mut self.fft_scratch,
            )
            .unwrap();

        output[..rfft_len]
            .iter_mut()
            .zip(self.fft_tr_output.iter())
            .for_each(|(dst, src)| {
                *dst = src.norm_sqr().sqrt();
            });
    }
}

#[derive(Clone)]
pub struct RFftMag<Dst, Src>(pub TransformState<Dst, Src>);

impl<Dst, Src> Transform1D<Dst, Src> for RFftMag<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        self.0.rfft_mag(output, input);
    }

    fn output_len(&self, _input_samples: usize) -> usize {
        self.0.rfft_len()
    }
}

#[cfg(test)]
mod test {
    use super::FilterState;
    use itertools::Itertools;

    #[test]
    fn test_filter() {
        // Note: input is carefully chosen so that boundary conditions are not annoying...
        let input: Vec<f32> = vec![0., 0., 0., 10., 5., 8., 3., 1., 7., 8., 9., 0., 0., 0.];
        let kernel: Vec<f32> = vec![1., 2., 3.];
        let mut output = vec![0f32; input.len()];
        let mut s: FilterState<f32, f32> = FilterState::new(input.len());
        s.load_kernel(&kernel);

        s.filter_single_pass(&mut output, &input);
        let output_i32: Vec<i32> = output.iter().map(|x| x.round() as i32).collect();
        assert_eq!(
            output_i32,
            &[0, 0, 0, 10, 25, 48, 34, 31, 18, 25, 46, 42, 27, 0]
        );

        s.filter_two_pass(&mut output, &input);
        let output_i32: Vec<i32> = output.iter().map(|x| x.round() as i32).collect();
        assert_eq!(
            output_i32,
            &[0, 30, 95, 204, 223, 209, 150, 142, 206, 243, 211, 96, 27, 0,],
        );
    }

    #[test]
    fn test_phase_correlation() {
        let input: Vec<f32> = vec![0., 0., 1., 1., 1., 0., 0., 1.];
        let mut output = vec![0f32; input.len()];
        let kernel: Vec<f32> = vec![0., 1., 1., 1.];

        let mut s: FilterState<f32, f32> = FilterState::new(input.len());
        s.load_kernel(&kernel);
        s.phase_correlation(&mut output, &input);

        let max_idx = output
            .iter()
            .cloned()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
            .unwrap();
        assert_eq!(max_idx, 1);
    }

    #[test]
    fn test_fft_correlation() {
        // Check the FFT correlation used in metrics works.
        let mut input: Vec<f32> = (1..6).map(|x| x as f32).collect();
        let fft_len = input.len() + 3 - 1;
        input.resize(fft_len, 0f32);
        eprintln!("{:?}", input);

        let mut kernel: Vec<f32> = vec![1., 2., 3.];
        kernel.reverse();
        kernel.resize(fft_len, 0f32);

        let mut output = vec![0f32; fft_len];
        let mut s: FilterState<f32, f32> = FilterState::new(input.len());
        s.load_kernel(&kernel);
        s.filter_single_pass(&mut output, &input);
        let output_i32 = output.iter().map(|x| x.round() as i32).collect_vec();
        assert_eq!(output_i32, &[3, 8, 14, 20, 26, 14, 5]);
    }
}