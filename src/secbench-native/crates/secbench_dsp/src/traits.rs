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
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Axis, Zip};
use num_traits::{Float, FromPrimitive, NumAssignOps, Zero};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use realfft::FftNum;

/// Types that support trivial conversion to float.
pub trait IntoFloat<F> {
    fn into_float(self) -> F;
}

macro_rules! impl_into_float {
    ($Src:ty => $($Dst:ty),*) => { $(
        impl IntoFloat<$Dst> for $Src {
            #[inline(always)]
            fn into_float(self) -> $Dst {
                self as $Dst
            }
        }
    )* };
}

impl_into_float!(i8 => f32, f64);
impl_into_float!(i16 => f32, f64);
impl_into_float!(f32 => f32, f64);
impl_into_float!(f64 => f32, f64);

/// Type of float used through the secbench_dsp crate.
///
/// In our case, this is just an alias to f16, f32, f64...
pub trait DspFloat: NumAssignOps + Float + FromPrimitive + FftNum {}

impl<T> DspFloat for T where T: NumAssignOps + Float + FromPrimitive + FftNum {}

pub trait Transform1D<Dst, Src> {
    fn apply_inplace(&mut self, output: &mut [Dst], input: &[Src]);

    /// Number of expected samples in the output array.
    ///
    /// Default implementation assume output and input have
    /// the same number of samples.
    fn output_len(&self, input_samples: usize) -> usize {
        input_samples
    }
}

pub trait Transform2D<Dst, Src> {
    fn apply_2d_inplace(&mut self, output: ArrayViewMut2<Dst>, input: ArrayView2<Src>);
    fn apply_2d_inplace_parallel(
        &mut self,
        output: ArrayViewMut2<Dst>,
        input: ArrayView2<Src>,
        chunk_size: Option<usize>,
    );

    fn apply_2d(&mut self, input: ArrayView2<Src>) -> Array2<Dst>;
    fn apply_2d_parallel(
        &mut self,
        input: ArrayView2<Src>,
        chunk_size: Option<usize>,
    ) -> Array2<Dst>;
}

impl<T, Dst, Src> Transform2D<Dst, Src> for T
where
    Dst: Clone + Send + Sync + Zero,
    Src: Send + Sync,
    T: Transform1D<Dst, Src> + Clone + Send + Sync,
{
    fn apply_2d_inplace(&mut self, mut output: ArrayViewMut2<Dst>, input: ArrayView2<Src>) {
        Zip::from(output.axis_iter_mut(Axis(0)))
            .and(input.axis_iter(Axis(0)))
            .for_each(|mut x, y| {
                self.apply_inplace(x.as_slice_mut().unwrap(), y.as_slice().unwrap())
            });
    }

    fn apply_2d_inplace_parallel(
        &mut self,
        mut output: ArrayViewMut2<Dst>,
        input: ArrayView2<Src>,
        chunk_size: Option<usize>,
    ) {
        if let Some(chunk_size) = chunk_size {
            (
                output.axis_chunks_iter_mut(Axis(0), chunk_size),
                input.axis_chunks_iter(Axis(0), chunk_size),
            )
                .into_par_iter()
                .for_each_init(
                    || self.clone(),
                    |state, (x, y)| state.apply_2d_inplace(x, y),
                );
        } else {
            (output.axis_iter_mut(Axis(0)), input.axis_iter(Axis(0)))
                .into_par_iter()
                .for_each_init(
                    || self.clone(),
                    |state, (mut x, y)| {
                        state.apply_inplace(x.as_slice_mut().unwrap(), y.as_slice().unwrap())
                    },
                );
        }
    }

    fn apply_2d(&mut self, input: ArrayView2<Src>) -> Array2<Dst> {
        let olen = self.output_len(input.ncols());
        let mut result = Array2::zeros([input.nrows(), olen]);
        self.apply_2d_inplace(result.view_mut(), input);
        result
    }

    fn apply_2d_parallel(
        &mut self,
        input: ArrayView2<Src>,
        chunk_size: Option<usize>,
    ) -> Array2<Dst> {
        let olen = self.output_len(input.ncols());
        let mut result = Array2::zeros([input.nrows(), olen]);
        self.apply_2d_inplace_parallel(result.view_mut(), input, chunk_size);
        result
    }
}