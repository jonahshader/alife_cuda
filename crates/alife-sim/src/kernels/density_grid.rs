//! `calculate_density_grid_kernel`: the GUI's density overlay. The fluid's
//! SPH density sampled on a regular lattice and coloured into one RGBA8 texel
//! per sample, packed into a `u32` (red in the low byte, as WGSL's
//! `unpack4x8unorm` reads it).
//!
//! A diagnostic, not part of the step: the window computes it once a frame
//! while the overlay is on, and nothing else ever launches it.
//!
//! The colours are diagnostics too:
//!
//! - **red** — the sample's cell holds more than `max_particles_per_cell`.
//!   Every neighbour kernel reads at most that many entries of a cell
//!   (`grid::neighbour_cell`), so the rest are silently dropped from every
//!   density and force sum; red is the only place that shows.
//! - **blue** — the density is above the display scale (300 in the window,
//!   as in the C++ screen), so greyscale would saturate.
//! - otherwise greyscale, `density / max_density`.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::grid::{GridDevice, GridRef, neighbour_cell, neighbours, unwrap_x};
use super::{
  Cfg, GridArgs, P_BOUNDS_X, P_CELL_SIZE, P_SMOOTHING_RADIUS, SphArgs, density_kernel,
  density_kernel_ref, particle_to_cid,
};
use crate::particles::{SphDevice, SphHost};

/// The texels of one density overlay, on the device: `width x height`
/// samples, row-major from the world's lower-left corner, each one
/// `sample_interval` metres from the next. Sample `(x, y)` is taken at
/// `(x, y) * sample_interval`, the texel's lower-left corner.
#[derive(Debug, Clone)]
pub struct DensityGrid {
  pub texels: Handle,
  pub width: u32,
  pub height: u32,
  pub sample_interval: f32,
}

impl DensityGrid {
  /// Room for a `width x height` overlay across a world `bounds_x` wide.
  /// The samples span the width exactly; the height is what it is, so the
  /// top row can sit a little above or below the world's ceiling.
  pub fn alloc<R: Runtime>(
    client: &ComputeClient<R>,
    width: u32,
    height: u32,
    bounds_x: f32,
  ) -> Self {
    Self {
      texels: client.empty((width * height).max(1) as usize * size_of::<u32>()),
      width,
      height,
      sample_interval: bounds_x / width.max(1) as f32,
    }
  }
}

/// Pack one texel as the shader unpacks it.
#[cube]
fn pack_rgba(r: u32, g: u32, b: u32) -> u32 {
  r | (g << 8u32) | (b << 16u32) | (255u32 << 24u32)
}

/// Host twin of [`pack_rgba`].
pub fn pack_rgba_ref(r: u32, g: u32, b: u32) -> u32 {
  r | (g << 8) | (b << 16) | (255 << 24)
}

#[allow(clippy::too_many_arguments)]
#[cube(launch)]
pub fn calculate_density_grid(
  sph: &SphArgs,
  grid: &GridArgs,
  params: &[f32],
  texels: &mut [u32],
  width: u32,
  height: u32,
  sample_interval: f32,
  max_density: f32,
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= (width * height) as usize {
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let smoothing_radius = params[P_SMOOTHING_RADIUS as usize];
  let bounds_x = params[P_BOUNDS_X as usize];

  let sample_x = (i as u32 % width) as f32 * sample_interval;
  let sample_y = (i as u32 / width) as f32 * sample_interval;
  let grid_index = particle_to_cid(sample_x, sample_y, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  // The C++ took this from `calculate_density_at_pos`, which returns the near
  // density as well and threw it away here.
  // TODO: also report near density
  let mut density = 0.0f32;
  for dy in 0..3u32 {
    for dx in 0..3u32 {
      let cell = neighbour_cell(grid, cell_x, cell_y, dx, dy, bounds_x, cfg);
      for k in 0..cell.count {
        let particle_id = grid.sorted_ids[cell.start as usize + k as usize] as usize;
        let other_x = unwrap_x(sph.pos[2 * particle_id], cell.x_shift);
        let other_y = sph.pos[2 * particle_id + 1];
        let diff_x = sample_x - other_x;
        let diff_y = sample_y - other_y;
        let distance = f32::sqrt(diff_x * diff_x + diff_y * diff_y);
        density += sph.mass[particle_id] * density_kernel(smoothing_radius, distance);
      }
    }
  }

  // The full count, not the clamped one the gather above read: the excess is
  // exactly what red reports.
  let num_particles = grid.cell_counts[grid_index as usize];

  // normalize density
  let normalized = density / max_density;

  if num_particles > cfg.max_per_cell {
    // red if too many particles per cell
    texels[i] = pack_rgba(255u32, 0u32, 0u32);
  } else if normalized > 1.0f32 {
    // blue if density over max
    texels[i] = pack_rgba(0u32, 0u32, 255u32);
  } else {
    // normal, grayscale rendering
    let grey = (255.0f32 * normalized) as u32;
    texels[i] = pack_rgba(grey, grey, grey);
  }
}

/// Plain-Rust reference for [`calculate_density_grid`]: the texels, in the
/// order the kernel writes them.
#[allow(clippy::too_many_arguments)]
pub fn calculate_density_grid_ref(
  particles: &SphHost,
  grid: &GridRef,
  cell_size: f32,
  smoothing_radius: f32,
  bounds_x: f32,
  cfg: &Cfg,
  width: u32,
  height: u32,
  sample_interval: f32,
  max_density: f32,
) -> Vec<u32> {
  (0..width * height)
    .map(|i| {
      let sample = glam::Vec2::new(
        (i % width) as f32 * sample_interval,
        (i / width) as f32 * sample_interval,
      );
      let grid_index = super::particle_to_cid_ref(sample, cell_size, cfg);
      let cell_x = (grid_index % cfg.grid_w as u32) as i32;
      let cell_y = (grid_index / cfg.grid_w as u32) as i32;

      let mut density = 0.0f32;
      for (pid, x_shift) in neighbours(grid, cell_x, cell_y, bounds_x, cfg) {
        let mut other = particles.pos[pid];
        other.x += x_shift;
        let distance = (sample - other).length();
        density += particles.mass[pid] * density_kernel_ref(smoothing_radius, distance);
      }

      let normalized = density / max_density;
      if grid.cell_counts[grid_index as usize] > cfg.max_per_cell {
        pack_rgba_ref(255, 0, 0)
      } else if normalized > 1.0 {
        pack_rgba_ref(0, 0, 255)
      } else {
        let grey = (255.0 * normalized) as u32;
        pack_rgba_ref(grey, grey, grey)
      }
    })
    .collect()
}

/// Launch helper. Reads the grid the last step built, as the C++ did after
/// `update_fluid`.
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  grid: &GridDevice,
  params: &Handle,
  cfg: Cfg,
  out: &DensityGrid,
  max_density: f32,
) {
  let samples = (out.width * out.height) as usize;
  calculate_density_grid::launch::<R>(
    client,
    super::cube_count(samples),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::grid_args(grid, &cfg),
    super::whole(params, super::PARAM_COUNT),
    super::whole(&out.texels, samples.max(1)),
    out.width,
    out.height,
    out.sample_interval,
    max_density,
    cfg,
  );
}

#[cfg(test)]
mod tests {
  use super::super::test_support::Harness;
  use super::{DensityGrid, pack_rgba_ref};

  /// One fixed input, with the overflow threshold and the display scale set
  /// so that all three colours turn up: the harness averages three particles
  /// a cell, so a threshold of three flags the crowded ones red.
  #[test]
  fn matches_reference() {
    let h = Harness::new();
    h.build_grid();
    let cfg = super::Cfg {
      max_per_cell: 3,
      ..h.cfg
    };
    // 10 samples per metre, as the window takes them.
    let width = (h.geom.bounds.x * 10.0).round() as u32;
    let height = (h.geom.bounds.y * 10.0).round() as u32;
    let out = DensityGrid::alloc(&h.client, width, height, h.geom.bounds.x);
    let max_density = 40.0;
    super::launch(
      &h.client,
      &h.sph,
      &h.grid,
      &h.params_buf,
      cfg,
      &out,
      max_density,
    );
    let bytes = h.client.read_one_unchecked(out.texels.clone());
    let actual = bytemuck::cast_slice::<u8, u32>(&bytes)[..(width * height) as usize].to_vec();

    let expected = super::calculate_density_grid_ref(
      &h.particles,
      &h.grid_ref(),
      h.geom.cell_size,
      h.params.smoothing_radius,
      h.geom.bounds.x,
      &cfg,
      width,
      height,
      out.sample_interval,
      max_density,
    );

    // Greyscale truncates a float to a byte, so one ulp of difference in the
    // density can move a texel by one level; the classification cannot move.
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
      let channels = |t: u32| t.to_le_bytes();
      let (a, e) = (channels(*a), channels(*e));
      for c in 0..4 {
        assert!(
          a[c].abs_diff(e[c]) <= 1,
          "texel {i}: kernel {a:?}, reference {e:?}"
        );
      }
    }

    let red = pack_rgba_ref(255, 0, 0);
    let blue = pack_rgba_ref(0, 0, 255);
    assert!(expected.contains(&red), "no overflowing cell");
    assert!(expected.contains(&blue), "no density over the scale");
    assert!(
      expected
        .iter()
        .any(|t| *t != red && *t != blue && t & 0xFF > 0),
      "no greyscale texel"
    );
  }
}
