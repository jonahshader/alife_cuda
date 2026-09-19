//! The fixed problem configuration for the spike.
//!
//! Mirrors the C++ defaults in `src/config/sim_params.h` (smoothing radius
//! 0.2, cell size = smoothing radius, 4 particles per cell) at the C++ world
//! size of 32 x 16 m, so the particle count matches the headless baseline in
//! `docs/perf.md`.

use core::hash::{Hash, Hasher};

pub const GRID_W: u32 = 160;
pub const GRID_H: u32 = 80;
pub const CELL_SIZE: f32 = 0.2;
pub const SMOOTHING_RADIUS: f32 = 0.2;
pub const PARTICLES_PER_CELL: u32 = 4;
/// The C++ default is 128; 32 is plenty at 4 particles per cell on average and
/// keeps the grid-index buffer at 1.6 MB, which matters for the CPU runtime.
pub const MAX_PER_CELL: u32 = 32;

pub const NUM_CELLS: u32 = GRID_W * GRID_H;
pub const NUM_PARTICLES: u32 = NUM_CELLS * PARTICLES_PER_CELL;
pub const BOUNDS_X: f32 = GRID_W as f32 * CELL_SIZE;
pub const BOUNDS_Y: f32 = GRID_H as f32 * CELL_SIZE;

/// Comptime kernel configuration. Every field is a constant baked into the
/// kernel at JIT time, so the kernels take no runtime scalar arguments.
///
/// `f32` is neither `Eq` nor `Hash`, and CubeCL keys a compiled kernel on the
/// hash of its comptime arguments, so both are implemented over the bit
/// patterns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cfg {
  pub grid_w: i32,
  pub grid_h: i32,
  pub max_per_cell: u32,
  pub max_per_cell_us: usize,
  pub cell_size: f32,
  pub radius: f32,
  pub bounds_x: f32,
  /// `6 / (pi * radius^4)`, the 2D normalization of the density kernel.
  pub norm_density: f32,
  /// `20 / (pi * radius^5)`, the 2D normalization of the near-density kernel.
  pub norm_near: f32,
}

impl Eq for Cfg {}

impl Hash for Cfg {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.grid_w.hash(state);
    self.grid_h.hash(state);
    self.max_per_cell.hash(state);
    self.cell_size.to_bits().hash(state);
    self.radius.to_bits().hash(state);
    self.bounds_x.to_bits().hash(state);
  }
}

impl Default for Cfg {
  fn default() -> Self {
    let r = SMOOTHING_RADIUS;
    Self {
      grid_w: GRID_W as i32,
      grid_h: GRID_H as i32,
      max_per_cell: MAX_PER_CELL,
      max_per_cell_us: MAX_PER_CELL as usize,
      cell_size: CELL_SIZE,
      radius: r,
      bounds_x: BOUNDS_X,
      norm_density: 6.0 / (core::f32::consts::PI * r * r * r * r),
      norm_near: 20.0 / (core::f32::consts::PI * r * r * r * r * r),
    }
  }
}
