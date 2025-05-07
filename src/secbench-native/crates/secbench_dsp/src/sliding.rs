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

use crate::fft::FilterState;
use crate::traits::{DspFloat, Transform1D};
use itertools::{izip, Itertools};
use num_traits::AsPrimitive;
use std::iter::Sum;
use std::ops::AddAssign;
use std::{hint::black_box, marker::PhantomData};

#[derive(Clone)]
pub struct MovingSum<Dst, Src> {
    window_size: usize,
    scale: Dst,
    _src: PhantomData<Src>,
}

impl<Dst, Src> MovingSum<Dst, Src> {
    pub fn new(window_size: usize, scale: Dst) -> Self {
        MovingSum {
            window_size,
            scale,
            _src: Default::default(),
        }
    }
}

impl<Dst, Src> Transform1D<Dst, Src> for MovingSum<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        let window_size = self.window_size;
        assert!(window_size > 0);
        assert!(window_size <= output.len());
        assert_eq!(output.len(), input.len());
        // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
        // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
        let mut sum = Dst::zero();
        let mut error = Dst::zero();
        for j in 0..output.len() {
            let x = input[j].as_();
            let t = black_box(sum + x);
            if sum.abs() >= x.abs() {
                error += (sum - t) + x;
            } else {
                error += (x - t) + sum;
            }
            sum = t;
            output[j] = t + error;

            // Alternative version, much faster.
            // sum += src[j].as_();
            // dst[j] = sum;
        }

        // Compute windowed summation.
        let mut s_prev = Dst::zero();
        let j_last = output.len() - 1;
        let j_end = output.len() - window_size;
        for j in 0..output.len() {
            let s_curr = output[j];
            let s_end = if j <= j_end {
                output[j + window_size - 1]
            } else {
                output[j_last]
            };

            let mut tmp = s_end - s_prev;
            if self.scale != Dst::one() {
                tmp *= self.scale;
            }
            output[j] = tmp;
            s_prev = s_curr;
        }
    }
}

#[derive(Clone)]
pub struct MatchEuclidean<Dst, Src> {
    p_len: usize,
    p_square: Dst,
    tmp_x: Vec<Dst>,
    tmp_xx: Vec<Dst>,
    tmp_xp: Vec<Dst>,
    filter: FilterState<Dst, Src>,
}

impl<Dst, Src> MatchEuclidean<Dst, Src>
where
    Dst: DspFloat + Sum + 'static,
    Src: AsPrimitive<Dst> + Copy,
{
    pub fn new(pattern: &[Dst], seq_length: usize) -> Self {
        debug_assert!(pattern.len() > 0);
        debug_assert!(pattern.len() <= seq_length);
        let fft_len = pattern.len() + seq_length - 1;
        let mut filter: FilterState<Dst, Src> = FilterState::new(fft_len);
        let p_square = pattern.iter().map(|&x| x * x).sum();
        let pattern_padded = pattern.iter().cloned().rev().collect_vec();
        filter.load_kernel(&pattern_padded);
        MatchEuclidean {
            p_len: pattern.len(),
            p_square,
            tmp_x: vec![Dst::zero(); seq_length],
            tmp_xx: vec![Dst::zero(); seq_length],
            tmp_xp: vec![Dst::zero(); fft_len],
            filter,
        }
    }
}

impl<Dst, Src> Transform1D<Dst, Src> for MatchEuclidean<Dst, Src>
where
    Dst: DspFloat + 'static + AsPrimitive<Dst> + From<u8>,
    Src: AsPrimitive<Dst> + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        debug_assert!(output.len() >= self.output_len(input.len()));
        debug_assert!(input.len() <= self.filter.fft_len());

        let mut ms: MovingSum<Dst, Dst> = MovingSum::new(self.p_len, Dst::one());
        for (dst, &x) in self.tmp_x.iter_mut().zip(input.iter()) {
            *dst = x.as_() * x.as_();
        }
        ms.apply_inplace(&mut self.tmp_xx, self.tmp_x.as_slice());

        self.filter.filter_single_pass(&mut self.tmp_xp, input);
        output
            .iter_mut()
            .zip(self.tmp_xx.iter().zip(&self.tmp_xp[self.p_len - 1..]))
            .for_each(|(dst, (&xx, &xp))| {
                *dst = xx - <Dst as From<u8>>::from(2) * xp + self.p_square;
            });
    }

    fn output_len(&self, input_samples: usize) -> usize {
        input_samples - (self.p_len - 1)
    }
}

#[derive(Clone)]
pub struct MatchCorrelation<Dst, Src> {
    p_len: usize,
    p_mean: Dst,
    p_std: Dst,
    tmp_x_ms: Vec<Dst>,
    tmp_x_std: Vec<Dst>,
    tmp_xp: Vec<Dst>,
    filter: FilterState<Dst, Src>,
    sliding_std: SlidingExecutor<Dst, Src>,
}

impl<Dst, Src> MatchCorrelation<Dst, Src>
where
    Dst: DspFloat + Sum + 'static,
    Src: AsPrimitive<Dst> + AddAssign + Copy,
    usize: AsPrimitive<Dst>,
{
    pub fn new(pattern: &[Dst], seq_length: usize) -> Self {
        assert!(pattern.len() <= seq_length);
        assert!(pattern.len() > 0);
        let fft_len = pattern.len() + seq_length - 1;
        let p_len: Dst = pattern.len().as_();
        let p_sum: Dst = pattern.iter().cloned().sum();
        let p_mean: Dst = p_sum / p_len;
        let p_square_sum: Dst = pattern.iter().map(|&x| x * x).sum();
        let p_std: Dst = (p_square_sum / pattern.len().as_() - p_mean * p_mean).sqrt();

        let mut filter: FilterState<Dst, Src> = FilterState::new(fft_len);
        let pattern_padded = pattern.iter().cloned().rev().collect_vec();
        filter.load_kernel(&pattern_padded);

        MatchCorrelation {
            p_len: pattern.len(),
            p_mean,
            p_std,
            tmp_x_ms: vec![Dst::zero(); seq_length],
            tmp_x_std: vec![Dst::zero(); seq_length],
            tmp_xp: vec![Dst::zero(); fft_len],
            filter,
            sliding_std: SlidingExecutor::new(SlidingType::Std, pattern.len(), Some(Dst::one())),
        }
    }
}

impl<Dst, Src> Transform1D<Dst, Src> for MatchCorrelation<Dst, Src>
where
    Dst: DspFloat + 'static + AsPrimitive<Dst> + From<u8>,
    Src: AsPrimitive<Dst> + AddAssign + Copy,
{
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]) {
        debug_assert!(input.len() >= self.p_len);
        debug_assert!(input.len() <= self.filter.fft_len());
        debug_assert!(input.len() <= self.tmp_x_ms.len());

        self.filter.filter_single_pass(&mut self.tmp_xp, input);
        let mut ms: MovingSum<Dst, Src> = MovingSum::new(self.p_len, Dst::one());
        ms.apply_inplace(&mut self.tmp_x_ms[..input.len()], input);

        self.sliding_std
            .apply_inplace(&mut self.tmp_x_std[..input.len()], input);

        let output_len = input.len() - (self.p_len - 1);
        izip!(
            &mut output[..output_len],
            &self.tmp_xp[self.p_len - 1..],
            self.tmp_x_ms.iter(),
            &self.tmp_x_std[self.p_len - 1..],
        )
        .for_each(|(dst, &xp, &x_ms, &x_std)| {
            *dst = (xp - x_ms * self.p_mean) / (x_std * self.p_std);
        });
    }

    fn output_len(&self, input_samples: usize) -> usize {
        input_samples - (self.p_len - 1)
    }
}

#[derive(Clone)]
pub enum SlidingType {
    Mean,
    Var,
    Std,
    Skew,
    Kurt,
}

#[derive(Clone)]
pub struct SlidingExecutor<Dst, Src> {
    sliding_type: SlidingType,
    window_size: usize,
    padding_value: Option<Dst>,

    win_sized_cache1: Vec<Dst>,

    // used in the case of skew or kurt calculation
    coef: Dst,
    subs: Dst,

    phantom: PhantomData<Src>,
}

impl<Dst, Src> SlidingExecutor<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy + AddAssign,
{
    pub fn new(sliding_type: SlidingType, window_size: usize, padding_value: Option<Dst>) -> Self {
        let (coef, subs) = match sliding_type {
            SlidingType::Mean => (Dst::zero(), Dst::zero()),
            SlidingType::Var | SlidingType::Std => (Dst::zero(), Dst::zero()),
            SlidingType::Skew => {
                // perform the unbiased calculation of skewness
                // https://en.wikipedia.org/wiki/Skewness
                let coef: Dst = Dst::from(
                    ((window_size * window_size) as f64)
                        / (((window_size - 1) * (window_size - 2)) as f64),
                )
                .unwrap();

                (coef, Dst::zero())
            }
            SlidingType::Kurt => {
                // calculation of the unbiased kurtosis
                // https://en.wikipedia.org/wiki/Kurtosis
                let coef = Dst::from(
                    (((window_size + 1) * window_size) as f64)
                        / (((window_size - 1) * (window_size - 2) * (window_size - 3)) as f64),
                )
                .unwrap();

                let subs = Dst::from(
                    3. * ((((window_size - 1) * (window_size - 1)) as f64)
                        / (((window_size - 2) * (window_size - 3)) as f64)),
                )
                .unwrap();

                (coef, subs)
            }
        };

        Self {
            sliding_type,
            window_size,
            padding_value,
            win_sized_cache1: vec![Dst::zero(); window_size],
            coef,
            subs,
            phantom: PhantomData,
        }
    }

    fn sliding_mean(&mut self, in_array: &[Src], out_array: &mut [Dst]) {
        let mut error = Dst::zero();
        let mut acc = Dst::zero();
        let o_win_size = Dst::from(self.window_size).unwrap();

        self.win_sized_cache1[self.window_size - 1] = Dst::zero();
        for (i, x) in in_array.iter().enumerate() {
            // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            {
                let x = x.as_();
                let t = black_box(acc + x);
                if acc.abs() >= x.abs() {
                    error += (acc - t) + x;
                } else {
                    error += (x - t) + acc;
                }
                acc = t;
            }
            if i >= self.window_size - 1 {
                out_array[i] = (acc - self.win_sized_cache1[i % self.window_size]) / o_win_size;
            }
            self.win_sized_cache1[i % self.window_size] = acc;
        }
    }

    fn sliding_var(&mut self, in_array: &[Src], out_array: &mut [Dst]) {
        let mut error = Dst::zero();
        let mut acc = Dst::zero();
        let o_win_size = Dst::from(self.window_size).unwrap();

        self.win_sized_cache1[self.window_size - 1] = Dst::zero();
        for i in 0..in_array.len() {
            let x = in_array[i];
            // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            {
                let x = x.as_();
                let t = black_box(acc + x);
                if acc.abs() >= x.abs() {
                    error += (acc - t) + x;
                } else {
                    error += (x - t) + acc;
                }
                acc = t;
            }
            let acc_x = acc + error;

            if i >= self.window_size - 1 {
                // mean of the window starting at idx i
                let mean_i = (acc_x - self.win_sized_cache1[i % self.window_size]) / o_win_size;

                let mut sum = Dst::zero();
                for j in 0..self.window_size {
                    // will perform for each of the element in the window
                    // perform (X - mu)^p of each element of the window
                    sum += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                }

                out_array[i] = sum / (o_win_size - Dst::one());
            }

            self.win_sized_cache1[i % self.window_size] = acc;
        }
    }

    fn sliding_std(&mut self, in_array: &[Src], out_array: &mut [Dst]) {
        let mut error = Dst::zero();
        let mut acc = Dst::zero();
        let o_win_size = Dst::from(self.window_size).unwrap();

        self.win_sized_cache1[self.window_size - 1] = Dst::zero();
        for i in 0..in_array.len() {
            let x = in_array[i];
            // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            {
                let x = x.as_();
                let t = black_box(acc + x);
                if acc.abs() >= x.abs() {
                    error += (acc - t) + x;
                } else {
                    error += (x - t) + acc;
                }
                acc = t;
            }
            let acc_x = acc + error;

            if i >= self.window_size - 1 {
                // mean of the window starting at idx i
                let mean_i = (acc_x - self.win_sized_cache1[i % self.window_size]) / o_win_size;

                let mut sum = Dst::zero();
                for j in 0..self.window_size {
                    // will perform for each of the element in the window
                    // perform (X - mu)^p of each element of the window
                    sum += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                }

                out_array[i] = (sum / (o_win_size - Dst::one())).sqrt();
            }

            self.win_sized_cache1[i % self.window_size] = acc;
        }
    }

    /// calculation of the unbiased skewness
    /// https://en.wikipedia.org/wiki/Skewness
    fn sliding_skew(&mut self, in_array: &[Src], out_array: &mut [Dst]) {
        let mut error = Dst::zero();
        let mut acc = Dst::zero();
        let o_win_size = Dst::from(self.window_size).unwrap();

        self.win_sized_cache1[self.window_size - 1] = Dst::zero();
        for i in 0..in_array.len() {
            let x = in_array[i];
            // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            {
                let x = x.as_();
                let t = black_box(acc + x);
                if acc.abs() >= x.abs() {
                    error += (acc - t) + x;
                } else {
                    error += (x - t) + acc;
                }

                acc = t;
            }
            let acc_x = acc + error;

            if i >= self.window_size - 1 {
                // mean of the window starting at idx i
                let mean_i = (acc_x - self.win_sized_cache1[i % self.window_size]) / o_win_size;
                // calculate the e_x_mu_rank_k
                // will perform for each of the element in the window
                // perform (X - mu)^p of each element of the window
                let (mut sum1, mut sum2) = (Dst::zero(), Dst::zero());
                for j in 0..self.window_size {
                    //uppser
                    sum1 += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                    //lower
                    sum2 += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                }

                let upper = sum1 / o_win_size;
                let lower = sum2 / (o_win_size - Dst::one());

                out_array[i] = (upper / lower.powf(Dst::from(3. / 2.).unwrap())) * self.coef;
            }

            self.win_sized_cache1[i % self.window_size] = acc;
        }
    }

    /// calculation of the unbiased kurtosis
    /// https://en.wikipedia.org/wiki/Kurtosis
    fn sliding_kurt(&mut self, in_array: &[Src], out_array: &mut [Dst]) {
        let mut error = Dst::zero();
        let mut acc = Dst::zero();
        let o_win_size = Dst::from(self.window_size).unwrap();

        self.win_sized_cache1[self.window_size - 1] = Dst::zero();
        for i in 0..in_array.len() {
            let x = in_array[i];
            // Compute cumulative sum using Kahan Babushka NeumaierSum summation, see
            // https://en.wikipedia.org/wiki/Kahan_summation_algorithm
            {
                let x = x.as_();
                let t = black_box(acc + x);
                if acc.abs() >= x.abs() {
                    error += (acc - t) + x;
                } else {
                    error += (x - t) + acc;
                }

                acc = t;
            }
            let acc_x = acc + error;

            if i >= self.window_size - 1 {
                // mean of the window starting at idx i
                let mean_i = (acc_x - self.win_sized_cache1[i % self.window_size]) / o_win_size;
                // calculate the e_x_mu_rank_k
                // will perform for each of the element in the window
                // perform (X - mu)^p of each element of the window
                let mut sum1 = Dst::zero();
                let mut sum2 = Dst::zero();

                for j in 0..self.window_size {
                    //upper
                    sum1 += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                    //lower
                    sum2 += (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i)
                        * (in_array[(j + i) - (self.window_size - 1)].as_() - mean_i);
                }

                let upper = sum1;
                let lower = sum2 / (o_win_size - Dst::one());

                out_array[i] = self.coef * (upper / (lower * lower)) - self.subs;
            }

            self.win_sized_cache1[i % self.window_size] = acc;
        }
    }
}

impl<Dst, Src> Transform1D<Dst, Src> for SlidingExecutor<Dst, Src>
where
    Dst: DspFloat + 'static,
    Src: AsPrimitive<Dst> + Copy + AddAssign,
{
    /// Returns the sliding mean/std/var/... of the vector with a window size of window_size.
    /// The resulting vector is of size : vector.len()  => padding is performed
    /// at the start of the result with value padding_value
    /// If None is given => will be Dst::zero()
    fn apply_inplace(&mut self, out_array: &mut [Dst], in_array: &[Src]) {
        let padding_value = self.padding_value.unwrap_or(Dst::zero());

        assert!(!in_array.is_empty());
        assert!(self.window_size > 1);
        assert!(in_array.len() >= self.window_size);
        assert!(out_array.len() == in_array.len());

        let number_of_window = in_array.len() - self.window_size + 1;

        let start_idx = in_array.len() - number_of_window;
        out_array[..start_idx].fill(padding_value);

        match self.sliding_type {
            SlidingType::Mean => self.sliding_mean(in_array, out_array),
            SlidingType::Var => self.sliding_var(in_array, out_array),
            SlidingType::Std => self.sliding_std(in_array, out_array),
            SlidingType::Skew => self.sliding_skew(in_array, out_array),
            SlidingType::Kurt => self.sliding_kurt(in_array, out_array),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Transform2D;
    use ndarray::{array, Array2};

    #[test]
    fn moving_sum_basic() {
        let t0 = Array2::from_shape_fn((3, 8), |(i, j)| i as i16 + j as i16);
        let expected = array![
            [3.0f32, 6.0, 9.0, 12.0, 15.0, 18.0, 13.0, 7.0],
            [6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 15.0, 8.0],
            [9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 17.0, 9.0],
        ];
        // let actual = moving_sum(t0.view(), 3, 1f32);
        // assert_eq!(actual, expected);
        let mut s: MovingSum<f32, i16> = MovingSum {
            window_size: 3,
            scale: 1f32,
            _src: Default::default(),
        };
        let actual = s.apply_2d(t0.view());
        assert_eq!(actual, expected);

        let actual = s.apply_2d_parallel(t0.view(), None);
        assert_eq!(actual, expected);
    }
}