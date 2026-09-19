//! Fixed-seed input generation, shared by every runtime and by the reference.

use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::config::*;

pub struct Input {
  pub pos_x: Vec<f32>,
  pub pos_y: Vec<f32>,
  pub mass: Vec<f32>,
  /// 0 is liquid (simulated); anything else is vapor and is skipped, the same
  /// test the C++ kernels make on `sph.state[i]`.
  pub state: Vec<u32>,
}

impl Input {
  pub fn generate(seed: u64) -> Self {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = NUM_PARTICLES as usize;
    let mut pos_x = Vec::with_capacity(n);
    let mut pos_y = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    let mut state = Vec::with_capacity(n);

    for _ in 0..n {
      // Kept strictly inside the world so that `pos / cell_size` never
      // truncates to the out-of-range cell at the far edge.
      pos_x.push(rng.random_range(0.001f32..BOUNDS_X - 0.001));
      pos_y.push(rng.random_range(0.001f32..BOUNDS_Y - 0.001));
      // Varying the mass catches a kernel that gathers the wrong
      // particle id but the right distance.
      mass.push(rng.random_range(0.9f32..1.1));
      // A tenth of the particles are vapor, so the state guard is live.
      state.push(if rng.random_range(0u32..10) == 0 {
        1
      } else {
        0
      });
    }

    Self {
      pos_x,
      pos_y,
      mass,
      state,
    }
  }
}
