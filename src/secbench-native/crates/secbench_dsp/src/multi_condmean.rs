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

use crate::{DspFloat, IntoFloat};
use ndarray::{
    s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, ArrayViewMut3,
    Axis, Zip,
};
use num_traits::AsPrimitive;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

pub type Label = u16;

#[derive(Clone)]
pub struct CondMeanVar<I> {
    // samples_per_class[target][class][sample_idx] -> mean accumulator value at instant sample_idx
    mean_per_class: Array3<I>,
    // samples_per_class[target][class][sample_idx] -> variance accumulator value at instant sample_idx
    var_per_class: Array3<I>,
    // samples_per_class[target][class] -> number of items in the class.
    samples_per_class: Array2<u32>,
}

impl<I> CondMeanVar<I>
where
    I: DspFloat + 'static,
    u32: AsPrimitive<I>,
{
    pub fn new(targets: usize, samples: usize, classes: usize) -> Self {
        debug_assert_ne!(samples, 0);
        debug_assert_ne!(classes, 0);
        debug_assert_ne!(targets, 0);
        CondMeanVar {
            mean_per_class: Array3::zeros([targets, classes, samples]),
            var_per_class: Array3::zeros([targets, classes, samples]),
            samples_per_class: Array2::zeros([targets, classes]),
        }
    }

    pub fn load_state(
        &mut self,
        mean_per_class: ArrayView3<I>,
        var_per_class: ArrayView3<I>,
        samples_per_class: ArrayView2<u32>,
    ) {
        self.mean_per_class.assign(&mean_per_class);
        self.var_per_class.assign(&var_per_class);
        self.samples_per_class.assign(&samples_per_class);
    }

    pub fn dump_state(&self) -> (Array3<I>, Array3<I>, Array2<u32>) {
        (
            self.mean_per_class.clone(),
            self.var_per_class.clone(),
            self.samples_per_class.clone(),
        )
    }

    /// Create arrays with correct shapes to be passed to [`CondMeanVar::process_into`].
    pub fn create_output_arrays(&self) -> (Array3<I>, Array3<I>) {
        let mean = Array3::zeros(self.mean_per_class.raw_dim());
        let var = mean.clone();
        (mean, var)
    }

    pub fn num_classes(&self) -> usize {
        self.mean_per_class.shape()[1]
    }

    pub fn process<S>(&mut self, data: ArrayView1<S>, labels: ArrayView1<Label>)
    where
        S: IntoFloat<I> + Copy,
    {
        debug_assert_eq!(data.len(), self.mean_per_class.shape()[2]);
        debug_assert_eq!(labels.shape()[0], self.samples_per_class.shape()[0]);

        Zip::from(labels)
            .and(self.samples_per_class.axis_iter_mut(Axis(0)))
            .and(self.mean_per_class.axis_iter_mut(Axis(0)))
            .and(self.var_per_class.axis_iter_mut(Axis(0)))
            .for_each(|&label, mut sx, mut mx, mut vx| {
                let label = label as usize;
                let samples = sx[label] + 1;
                sx[label] = samples;

                Zip::from(mx.slice_mut(s![label, ..]))
                    .and(vx.slice_mut(s![label, ..]))
                    .and(data)
                    .for_each(|m, v, &x| {
                        let x: I = x.into_float();
                        let delta = x - *m;
                        let new_mean = *m + delta / samples.as_();
                        *m = new_mean;
                        *v += delta * (x - new_mean);
                    });
            });
    }

    pub fn process_block<S>(&mut self, data: ArrayView2<S>, labels: ArrayView2<Label>)
    where
        S: IntoFloat<I> + Copy,
    {
        Zip::from(data.outer_iter())
            .and(labels.outer_iter())
            .for_each(|d, l| self.process(d, l));
    }

    fn freeze_single_class(mut dst: ArrayViewMut2<I>, samples_per_class: ArrayView1<u32>) {
        Zip::from(dst.axis_iter_mut(Axis(0)))
            .and(samples_per_class)
            .for_each(|mut row, &n| {
                if n == 0 {
                    row.map_inplace(|x| {
                        *x = I::zero();
                    });
                } else {
                    row.map_inplace(|x| {
                        *x /= n.as_();
                    });
                }
            });
    }

    fn freeze_var_into(&self, mut var: ArrayViewMut3<I>) {
        Zip::from(var.axis_iter_mut(Axis(0)))
            .and(self.samples_per_class.axis_iter(Axis(0)))
            .for_each(|var, samples| Self::freeze_single_class(var, samples))
    }

    pub fn freeze_into(&self, mut mean: ArrayViewMut3<I>, mut var: ArrayViewMut3<I>) {
        mean.assign(&self.mean_per_class.view());
        var.assign(&self.var_per_class.view());
        self.freeze_var_into(var);
    }

    pub fn freeze(&self) -> (Array3<I>, Array3<I>) {
        let mean = self.mean_per_class.clone();
        let mut var = self.var_per_class.clone();
        self.freeze_var_into(var.view_mut());
        (mean, var)
    }

    /// Compute the global mean and variance of the accumulator.
    pub fn freeze_global_mean_var(&self) -> (Array1<I>, Array1<I>, u32) {
        // NOTE: we implement the merging algorithm here. Currently we do not have 2D accumulators.
        let mut m_acc = self.mean_per_class.slice(s![0, 0, ..]).to_owned();
        let mut v_acc = self.var_per_class.slice(s![0, 0, ..]).to_owned();
        let mut samples = self.samples_per_class[(0, 0)];

        Zip::from(self.var_per_class.slice(s![0, 1.., ..]).axis_iter(Axis(0)))
            .and(self.mean_per_class.slice(s![0, 1.., ..]).axis_iter(Axis(0)))
            .and(self.samples_per_class.slice(s![0, 1..]))
            .for_each(|v2_row, m2_row, &s| {
                let n_1 = samples;
                let n_2 = s;
                let n = n_1 + n_2;
                Zip::from(v_acc.view_mut())
                    .and(m_acc.view_mut())
                    .and(v2_row)
                    .and(m2_row)
                    .for_each(|v1, m1, &v2, &m2| {
                        let delta = m2 - *m1;
                        *m1 += n_2.as_() * delta / n.as_();
                        *v1 += v2 + (n_1 * n_2).as_() * delta * delta / n.as_();
                    });
                // Update sample count.
                samples += n_2;
            });

        if samples > 1 {
            let denum = samples.as_() - 1.as_();
            v_acc.map_inplace(|x| *x /= denum);
        } else {
            v_acc.map_inplace(|x| *x = I::zero());
        }
        (m_acc, v_acc, samples)
    }

    pub fn samples_per_class(&self) -> Array2<u32> {
        self.samples_per_class.clone()
    }

    /// Snapshot of the signal to noise ratio.
    pub fn freeze_snr(&self) -> Array2<I> {
        let num = self.mean_per_class.var_axis(Axis(1), I::one());
        let mut var = self.var_per_class.clone();
        self.freeze_var_into(var.view_mut());
        let denum = var.mean_axis(Axis(1)).unwrap();
        num / denum
    }
}

pub struct CondMeanVarP<I> {
    workers: Box<[CondMeanVar<I>]>,
    chunks: Box<[(u32, u32)]>,
    targets: usize,
    samples: usize,
    classes: usize,
}

impl<I> CondMeanVarP<I>
where
    I: DspFloat + Sync + Send + 'static,
    u32: AsPrimitive<I>,
{
    pub fn new(chunk_size: usize, targets: usize, samples: usize, classes: usize) -> Self {
        let tmp = CondMeanVar::new(targets, samples, classes);
        CondMeanVarP::split(&tmp, chunk_size)
    }

    pub fn split(accum: &CondMeanVar<I>, chunk_size: usize) -> Self {
        let sh = accum.var_per_class.shape();
        let samples = sh[2];
        let classes = sh[1];
        let targets = sh[0];

        let chunk_count = (samples + chunk_size - 1) / chunk_size;
        let mut workers = Vec::with_capacity(chunk_count);
        let mut indices = Vec::with_capacity(chunk_count);
        for start in (0..samples).step_by(chunk_size) {
            let end = start + chunk_size;
            let end = end.min(samples);
            debug_assert_ne!(start, end);
            workers.push(CondMeanVar {
                mean_per_class: accum
                    .mean_per_class
                    .slice(s![.., .., start..end])
                    .to_owned(),
                var_per_class: accum.var_per_class.slice(s![.., .., start..end]).to_owned(),
                samples_per_class: accum.samples_per_class.clone(),
            });
            indices.push((start as u32, end as u32))
        }
        CondMeanVarP {
            workers: workers.into_boxed_slice(),
            chunks: indices.into_boxed_slice(),
            targets,
            samples,
            classes,
        }
    }

    pub fn merge(&self) -> CondMeanVar<I> {
        let mut m = Array3::zeros([self.targets, self.classes, self.samples]);
        let mut v = Array3::zeros([self.targets, self.classes, self.samples]);
        self.workers
            .iter()
            .zip(self.chunks.iter())
            .for_each(|(x, &(start, end))| {
                let (start, end) = (start as usize, end as usize);
                m.slice_mut(s![.., .., start..end])
                    .assign(&x.mean_per_class);
                v.slice_mut(s![.., .., start..end]).assign(&x.var_per_class);
            });

        CondMeanVar {
            mean_per_class: m,
            var_per_class: v,
            samples_per_class: self.workers[0].samples_per_class.clone(),
        }
    }

    pub fn process_block<S>(&mut self, data: ArrayView2<S>, labels: ArrayView2<Label>)
    where
        S: IntoFloat<I> + Copy + Sync + Send,
    {
        debug_assert_eq!(data.shape()[1], self.samples);
        self.workers
            .as_mut()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, w)| {
                let (start, end) = self.chunks[i];
                let (start, end) = (start as usize, end as usize);
                let d = data.slice(s![.., start..end]);
                w.process_block(d, labels);
            });
    }
}

#[cfg(test)]
mod test {
    use super::{CondMeanVar, CondMeanVarP};
    use ndarray::{Array2, Axis};
    use rand::distributions::Uniform;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_cond_mean_var() {
        // Note: this is more a sanity check that a real test.
        // There are more interesting tests done through the Python bindings.
        let t0 = Array2::from_shape_fn((10, 16), |(i, j)| (i % (j + 2)) as f32);
        let t1 = Array2::from_shape_fn((8, 16), |(i, j)| (10 + i % (j + 2)) as f32);

        let zeros = Array2::<u16>::zeros((10, 1));
        let ones = Array2::<u16>::ones((8, 1));

        let mut acc = CondMeanVar::<f32>::new(1, t0.shape()[1], 2);
        acc.process_block(t0.view(), zeros.view());
        acc.process_block(t1.view(), ones.view());
        let snr = acc.freeze_snr();
        assert_eq!(snr.shape(), [1, 16]);
        assert!(snr[[0, 0]] > snr[[0, 15]]);

        let mut par_acc = CondMeanVarP::<f32>::new(4, 1, t0.shape()[1], 2);
        par_acc.process_block(t0.view(), zeros.view());
        par_acc.process_block(t1.view(), ones.view());
        let acc_2 = par_acc.merge();

        let (mean, var) = acc.freeze();
        let (mean_2, var_2) = acc_2.freeze();
        assert_eq!(mean, mean_2, "mean accumulators are the same");
        assert_eq!(var, var_2, "mean accumulators are the same");
    }

    #[test]
    fn test_cond_mean_global() {
        let mut rng = StdRng::seed_from_u64(0xAAABBB);
        const N: usize = 20000;
        let t0 = Array2::from_shape_fn((N, 4), |(_i, j)| {
            rng.sample(Uniform::new_inclusive(0u32, 23)) as f32 + j as f32
        });
        let labels = Array2::from_shape_fn((N, 2), |(_i, _j)| rng.sample(Uniform::new(0u16, 15)));

        let mut acc = CondMeanVar::<f32>::new(2, t0.shape()[1], 16);
        acc.process_block(t0.view(), labels.view());
        let (mean, var, samples) = acc.freeze_global_mean_var();
        assert_eq!(samples, N as u32);
        assert_eq!(var.shape(), &[4]);

        let mean_expected = t0.mean_axis(Axis(0)).unwrap();
        let var_expected = t0.var_axis(Axis(0), 1f32);
        assert!(mean.abs_diff_eq(&mean_expected, 1e-3));
        assert!(var.abs_diff_eq(&var_expected, 1e-3));
    }
}