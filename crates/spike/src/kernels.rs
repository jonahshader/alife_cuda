//! The kernel under test: a port of `reset_particles_per_cell`,
//! `populate_grid_indices` and the soil-free `calculate_particle_density` from
//! `src/systems/particle_fluid2.cu`.
//!
//! One `#[cube]` body, launched unchanged on every runtime.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::config::*;
use crate::input::Input;
use crate::reference::Output;

#[cube]
fn density_w(dst: f32, #[comptime] cfg: Cfg) -> f32 {
  let mut out = 0.0f32;
  if dst < cfg.radius {
    let v = cfg.radius - dst;
    out = cfg.norm_density * v * v;
  }
  out
}

#[cube]
fn near_density_w(dst: f32, #[comptime] cfg: Cfg) -> f32 {
  let mut out = 0.0f32;
  if dst < cfg.radius {
    let v = cfg.radius - dst;
    out = cfg.norm_near * v * v * v;
  }
  out
}

#[cube(launch)]
pub fn reset_particles_per_cell(counts: &mut [u32]) {
  if ABSOLUTE_POS < counts.len() {
    counts[ABSOLUTE_POS] = 0u32;
  }
}

#[cube(launch)]
pub fn populate_grid_indices(
  pos_x: &[f32],
  pos_y: &[f32],
  state: &[u32],
  counts: &[Atomic<u32>],
  grid_indices: &mut [u32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= pos_x.len() {
    terminate!();
  }
  if state[i] != 0u32 {
    terminate!();
  }

  let cell_x = (pos_x[i] / cfg.cell_size) as i32;
  let cell_y = (pos_y[i] / cfg.cell_size) as i32;
  let cid = (cell_y * cfg.grid_w + cell_x) as usize;

  let slot = counts[cid].fetch_add(1u32);
  if slot < cfg.max_per_cell {
    grid_indices[cid * cfg.max_per_cell_us + slot as usize] = i as u32;
  }
}

#[cube(launch)]
pub fn calculate_particle_density(
  pos_x: &[f32],
  pos_y: &[f32],
  mass: &[f32],
  state: &[u32],
  counts: &[u32],
  grid_indices: &[u32],
  density: &mut [f32],
  near_density: &mut [f32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= pos_x.len() {
    terminate!();
  }
  if state[i] != 0u32 {
    terminate!();
  }

  let px = pos_x[i];
  let py = pos_y[i];
  let cell_x = (px / cfg.cell_size) as i32;
  let cell_y = (py / cfg.cell_size) as i32;

  // The C++ kernel widens the neighbourhood by one cell at the wrap seam
  // instead of wrapping the index arithmetic; keep that exactly, it changes
  // which particles contribute.
  let mut xi_neg = 1i32;
  if cell_x == 0 {
    xi_neg = 2i32;
  }
  let mut xi_pos = 1i32;
  if cell_x >= cfg.grid_w - 2 {
    xi_pos = 2i32;
  }
  let x_start = cell_x - xi_neg;
  let x_count = (xi_neg + xi_pos + 1) as u32;

  let mut dens = 0.0f32;
  let mut near = 0.0f32;

  for dy in 0..3u32 {
    let yi = cell_y + dy as i32 - 1;
    if yi >= 0 && yi < cfg.grid_h {
      for dx in 0..x_count {
        let xi = x_start + dx as i32;
        let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
        let neighbour = (yi * cfg.grid_w + wrapped_x) as usize;

        let mut cnt = counts[neighbour];
        if cnt > cfg.max_per_cell {
          cnt = cfg.max_per_cell;
        }

        for k in 0..cnt {
          let pid = grid_indices[neighbour * cfg.max_per_cell_us + k as usize] as usize;
          let mut ox = pos_x[pid];
          let oy = pos_y[pid];
          if xi < 0 {
            ox -= cfg.bounds_x;
          } else if xi >= cfg.grid_w {
            ox += cfg.bounds_x;
          }
          let ddx = px - ox;
          let ddy = py - oy;
          let dist = f32::sqrt(ddx * ddx + ddy * ddy);
          dens += mass[pid] * density_w(dist, cfg);
          near += mass[pid] * near_density_w(dist, cfg);
        }
      }
    }
  }

  density[i] = dens;
  near_density[i] = near;
}

/// Device-side buffers for one run.
pub struct Buffers {
  pub pos_x: Handle,
  pub pos_y: Handle,
  pub mass: Handle,
  pub state: Handle,
  pub counts: Handle,
  pub grid_indices: Handle,
  pub density: Handle,
  pub near_density: Handle,
}

pub fn upload<R: Runtime>(client: &ComputeClient<R>, input: &Input) -> Buffers {
  let n = input.pos_x.len();
  let cells = NUM_CELLS as usize;
  Buffers {
    pos_x: client.create_from_slice(f32::as_bytes(&input.pos_x)),
    pos_y: client.create_from_slice(f32::as_bytes(&input.pos_y)),
    mass: client.create_from_slice(f32::as_bytes(&input.mass)),
    state: client.create_from_slice(u32::as_bytes(&input.state)),
    counts: client.empty(cells * size_of::<u32>()),
    // `u32::MAX` in every slot, so a slot the kernel fails to write reads
    // back as an obviously invalid particle id rather than as 0.
    grid_indices: client.create_from_slice(u32::as_bytes(&vec![
      u32::MAX;
      cells * MAX_PER_CELL as usize
    ])),
    density: client.empty(n * size_of::<f32>()),
    near_density: client.empty(n * size_of::<f32>()),
  }
}

fn cube_count(work: usize, dim: u32) -> CubeCount {
  CubeCount::Static(work.div_ceil(dim as usize) as u32, 1, 1)
}

pub const CUBE_DIM: u32 = 256;

/// One full step: reset the counters, rebuild the grid, gather the densities.
pub fn launch_all<R: Runtime>(client: &ComputeClient<R>, bufs: &Buffers, cfg: Cfg) {
  let n = NUM_PARTICLES as usize;
  let cells = NUM_CELLS as usize;
  let mpc = MAX_PER_CELL as usize;

  reset_particles_per_cell::launch::<R>(
    client,
    cube_count(cells, CUBE_DIM),
    CubeDim::new_1d(CUBE_DIM),
    unsafe { BufferArg::from_raw_parts(bufs.counts.clone(), cells) },
  );

  populate_grid_indices::launch::<R>(
    client,
    cube_count(n, CUBE_DIM),
    CubeDim::new_1d(CUBE_DIM),
    unsafe { BufferArg::from_raw_parts(bufs.pos_x.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.pos_y.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.state.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.counts.clone(), cells) },
    unsafe { BufferArg::from_raw_parts(bufs.grid_indices.clone(), cells * mpc) },
    cfg,
  );

  calculate_particle_density::launch::<R>(
    client,
    cube_count(n, CUBE_DIM),
    CubeDim::new_1d(CUBE_DIM),
    unsafe { BufferArg::from_raw_parts(bufs.pos_x.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.pos_y.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.mass.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.state.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.counts.clone(), cells) },
    unsafe { BufferArg::from_raw_parts(bufs.grid_indices.clone(), cells * mpc) },
    unsafe { BufferArg::from_raw_parts(bufs.density.clone(), n) },
    unsafe { BufferArg::from_raw_parts(bufs.near_density.clone(), n) },
    cfg,
  );
}

pub fn download<R: Runtime>(client: &ComputeClient<R>, bufs: &Buffers) -> Output {
  let counts = u32::from_bytes(&client.read_one_unchecked(bufs.counts.clone())).to_vec();
  let grid_indices =
    u32::from_bytes(&client.read_one_unchecked(bufs.grid_indices.clone())).to_vec();
  let density = f32::from_bytes(&client.read_one_unchecked(bufs.density.clone())).to_vec();
  let near_density =
    f32::from_bytes(&client.read_one_unchecked(bufs.near_density.clone())).to_vec();
  Output {
    counts,
    grid_indices,
    density,
    near_density,
  }
}
