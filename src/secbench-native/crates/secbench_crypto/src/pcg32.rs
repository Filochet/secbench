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

use core::convert::TryInto;
use core::num::Wrapping;

use rand_core::{impls, Error, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};

type W64 = Wrapping<u64>;

/// Seed for the `Pcg32` PRNG.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Pcg32Seed([u8; 16]);

impl Pcg32Seed {
    /// Create a seed from an initial state (`state`) and a sequence
    /// index (`inc`).
    ///
    /// This corresponds to the seeding approach used
    /// in [PCG github](https://github.com/imneme/pcg-c-basic/blob/master/pcg32-demo.c) repository.
    ///
    /// Beware that this is specific to `Pcg32`, we recommend using the
    /// `rand` crate seeding instead (e.g., `SeedableRng::from_entropy`).
    pub fn from_state_inc(state: u64, inc: u64) -> Self {
        let mut w = [0u8; 16];
        w[0..8].copy_from_slice(&state.to_le_bytes());
        w[8..16].copy_from_slice(&inc.to_le_bytes());
        Pcg32Seed(w)
    }
}

impl AsMut<[u8]> for Pcg32Seed {
    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.0
    }
}

/// PCG32 Pseudo Random Number Generator (PRNG).                          
///
/// [PCG](https://www.pcg-random.org/index.html) is a family of lightweight PRNG.
/// The `Pcg32` implementation is *Rustified* from the [Minimal C Version](http://www.pcg-random.org/).
/// It implements traits from the [rand_core](https://crates.io/crates/rand_core),
/// so it is fully compatible with `rand` APIs.
///
/// **Caution notes**:
/// - This PRNG is supposed to generate random-looking output (i.e., with good
///   statistical properties). We do not endorse any flaws that may be
///   discovered in this PRNG.
/// - This PRNG is not **crypto-safe**: please, **never** use it to generate
///   cryptographic keys.
///
/// # Examples
///
/// ```
/// use secbench_crypto::{Pcg32, Pcg32Seed};
///
/// let mut rng : Pcg32 = Pcg32::new(Pcg32Seed::from_state_inc(0x42, 1));
/// let w1 = rng.generate();
/// let w2 = rng.generate();
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Create a new `Pcg32` instance using a given seed.
    pub fn new(seed: Pcg32Seed) -> Self {
        let mut rng = Pcg32 { state: 0, inc: 0 };
        rng.reset(seed);
        rng
    }

    /// Reset the PRNG instance using a given seed.
    pub fn reset(&mut self, seed: Pcg32Seed) {
        let inc = u64::from_le_bytes((&seed.0[0..8]).try_into().unwrap());
        let state = u64::from_le_bytes((&seed.0[8..16]).try_into().unwrap());
        self.state = 0;
        self.inc = (Wrapping(inc) << 1).0 | 1;
        self.generate();
        self.state = self.state.wrapping_add(state);
        self.generate();
    }

    /// Generate a random output.
    pub fn generate(&mut self) -> u64 {
        const DEFAULT_MULT: W64 = Wrapping(0x5851_f42d_4c95_7f2d);
        let old_state = Wrapping(self.state);
        self.state = (old_state * DEFAULT_MULT + Wrapping(self.inc)).0;
        let xor_shifted = ((old_state >> 18) ^ old_state) >> 27;
        let rot = old_state >> 59;
        let shift = (!rot + Wrapping(1)) & Wrapping(31);
        let result = (xor_shifted >> (rot.0 as usize)) | (xor_shifted << (shift.0 as usize));
        result.0
    }
}

impl From<Pcg32Seed> for Pcg32 {
    fn from(seed: Pcg32Seed) -> Self {
        Pcg32::new(seed)
    }
}

impl RngCore for Pcg32 {
    fn next_u32(&mut self) -> u32 {
        self.generate() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.generate()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl SeedableRng for Pcg32 {
    type Seed = Pcg32Seed;

    fn from_seed(seed: Pcg32Seed) -> Self {
        Pcg32::new(seed)
    }
}