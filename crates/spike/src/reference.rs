//! Plain-Rust reference for the three kernels, in the same f32 arithmetic.
//!
//! This is the oracle every runtime is compared against. It builds the grid
//! sequentially, so the slot order inside a cell is deterministic, which the
//! atomic kernels' order is not; `compare` accounts for that.

use crate::config::*;
use crate::input::Input;

pub struct Output {
  pub counts: Vec<u32>,
  pub grid_indices: Vec<u32>,
  pub density: Vec<f32>,
  pub near_density: Vec<f32>,
}

pub fn density_w(cfg: &Cfg, dst: f32) -> f32 {
  if dst >= cfg.radius {
    return 0.0;
  }
  let v = cfg.radius - dst;
  cfg.norm_density * v * v
}

pub fn near_density_w(cfg: &Cfg, dst: f32) -> f32 {
  if dst >= cfg.radius {
    return 0.0;
  }
  let v = cfg.radius - dst;
  cfg.norm_near * v * v * v
}

pub fn run(cfg: &Cfg, input: &Input) -> Output {
  let n = input.pos_x.len();
  let num_cells = NUM_CELLS as usize;
  let mpc = cfg.max_per_cell_us;

  let mut counts = vec![0u32; num_cells];
  let mut grid_indices = vec![u32::MAX; num_cells * mpc];

  for i in 0..n {
    if input.state[i] != 0 {
      continue;
    }
    let cid = cell_of(cfg, input.pos_x[i], input.pos_y[i]);
    let slot = counts[cid];
    counts[cid] += 1;
    if slot < cfg.max_per_cell {
      grid_indices[cid * mpc + slot as usize] = i as u32;
    }
  }

  let mut density = vec![0.0f32; n];
  let mut near_density = vec![0.0f32; n];

  for i in 0..n {
    if input.state[i] != 0 {
      continue;
    }
    let px = input.pos_x[i];
    let py = input.pos_y[i];
    let cell_x = (px / cfg.cell_size) as i32;
    let cell_y = (py / cfg.cell_size) as i32;

    let xi_neg = if cell_x == 0 { 2 } else { 1 };
    let xi_pos = if cell_x >= cfg.grid_w - 2 { 2 } else { 1 };

    let mut dens = 0.0f32;
    let mut near = 0.0f32;

    for yi in (cell_y - 1)..=(cell_y + 1) {
      if yi < 0 || yi >= cfg.grid_h {
        continue;
      }
      for xi in (cell_x - xi_neg)..=(cell_x + xi_pos) {
        let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
        let neighbour = (yi * cfg.grid_w + wrapped_x) as usize;
        let cnt = counts[neighbour].min(cfg.max_per_cell);
        for k in 0..cnt as usize {
          let pid = grid_indices[neighbour * mpc + k] as usize;
          let mut ox = input.pos_x[pid];
          let oy = input.pos_y[pid];
          if xi < 0 {
            ox -= cfg.bounds_x;
          } else if xi >= cfg.grid_w {
            ox += cfg.bounds_x;
          }
          let dx = px - ox;
          let dy = py - oy;
          let dist = (dx * dx + dy * dy).sqrt();
          dens += input.mass[pid] * density_w(cfg, dist);
          near += input.mass[pid] * near_density_w(cfg, dist);
        }
      }
    }

    density[i] = dens;
    near_density[i] = near;
  }

  Output {
    counts,
    grid_indices,
    density,
    near_density,
  }
}

fn cell_of(cfg: &Cfg, px: f32, py: f32) -> usize {
  let gx = (px / cfg.cell_size) as i32;
  let gy = (py / cfg.cell_size) as i32;
  (gy * cfg.grid_w + gx) as usize
}

pub struct Diff {
  pub counts_equal: bool,
  /// Per-cell sets of particle ids, compared after sorting: the atomic
  /// kernel's slot order inside a cell is not deterministic.
  pub grid_sets_equal: bool,
  pub max_abs_density: f32,
  pub max_abs_near_density: f32,
}

impl Diff {
  pub fn ok(&self) -> bool {
    self.counts_equal
      && self.grid_sets_equal
      && self.max_abs_density < 1e-2
      && self.max_abs_near_density < 1e-2
  }
}

pub fn compare(cfg: &Cfg, want: &Output, got: &Output) -> Diff {
  let mpc = cfg.max_per_cell_us;
  let counts_equal = want.counts == got.counts;

  let mut grid_sets_equal = true;
  for cell in 0..want.counts.len() {
    let used = want.counts[cell].min(cfg.max_per_cell) as usize;
    let mut a: Vec<u32> = want.grid_indices[cell * mpc..cell * mpc + used].to_vec();
    let mut b: Vec<u32> = got.grid_indices[cell * mpc..cell * mpc + used].to_vec();
    a.sort_unstable();
    b.sort_unstable();
    if a != b {
      grid_sets_equal = false;
      break;
    }
  }

  Diff {
    counts_equal,
    grid_sets_equal,
    max_abs_density: max_abs(&want.density, &got.density),
    max_abs_near_density: max_abs(&want.near_density, &got.near_density),
  }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}
