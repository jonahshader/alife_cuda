//! `calculate_particle_density`: SPH density at each liquid particle, plus the
//! solid fraction of the soil underneath it.

use cubecl::prelude::*;

use super::grid::{GridDevice, neighbour_cell, neighbours, unwrap_x};
use super::soil_sample::{solid_density_at_pos, solid_density_at_pos_ref};
use super::{
  Cfg, GridArgs, P_BOUNDS_X, P_CELL_SIZE, P_SMOOTHING_RADIUS, P_SOIL_SIZE, P_TARGET_DENSITY,
  SoilArgs, SphArgs, density_kernel, density_kernel_ref, near_density_kernel,
  near_density_kernel_ref, particle_to_cid,
};
use crate::particles::{ParticleKind, SphHost};
use crate::soil::SoilHost;

/// Density and near-density at a position, gathered over the 3x3 cell
/// neighbourhood. x wraps; y is clipped at the world floor and ceiling.
#[cube(launch)]
pub fn calculate_particle_density(
  sph: &mut SphArgs,
  grid: &GridArgs,
  soil: &SoilArgs,
  params: &[f32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != 0u32 {
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let smoothing_radius = params[P_SMOOTHING_RADIUS as usize];
  let bounds_x = params[P_BOUNDS_X as usize];

  let px = sph.pos[2 * i];
  let py = sph.pos[2 * i + 1];
  let grid_index = particle_to_cid(px, py, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  let mut density = 0.0f32;
  let mut near_density = 0.0f32;

  // iterate through cell neighborhood
  for dy in 0..3u32 {
    for dx in 0..3u32 {
      let cell = neighbour_cell(grid, cell_x, cell_y, dx, dy, bounds_x, cfg);
      // iterate through particles within the cell
      for k in 0..cell.count {
        let particle_id = grid.sorted_ids[cell.start as usize + k as usize] as usize;
        let other_x = unwrap_x(sph.pos[2 * particle_id], cell.x_shift);
        let other_y = sph.pos[2 * particle_id + 1];
        let diff_x = px - other_x;
        let diff_y = py - other_y;
        let distance = f32::sqrt(diff_x * diff_x + diff_y * diff_y);
        let mass = sph.mass[particle_id];
        density += mass * density_kernel(smoothing_radius, distance);
        near_density += mass * near_density_kernel(smoothing_radius, distance);
      }
    }
  }

  let soil_density = solid_density_at_pos(
    px,
    py,
    soil,
    params[P_SOIL_SIZE as usize],
    params[P_TARGET_DENSITY as usize],
    cfg,
  );

  sph.density[i] = density + soil_density;
  // The soil contributes no near-density: the C++ returns `make_float2(d, 0)`.
  sph.near_density[i] = near_density;
}

/// Plain-Rust reference for [`calculate_particle_density`].
pub fn calculate_particle_density_ref(
  particles: &mut SphHost,
  grid: &super::grid::GridRef,
  soil: &SoilHost,
  geom: &crate::world::WorldGeometry,
  params: &crate::SimParams,
  cfg: &Cfg,
) {
  let bounds_x = geom.bounds.x;
  for i in 0..particles.len() {
    if particles.state[i] != ParticleKind::Liquid {
      continue;
    }
    let pos = particles.pos[i];
    let grid_index = super::particle_to_cid_ref(pos, geom.cell_size, cfg);
    let cell_x = (grid_index % cfg.grid_w as u32) as i32;
    let cell_y = (grid_index / cfg.grid_w as u32) as i32;

    let mut density = 0.0f32;
    let mut near_density = 0.0f32;

    for (pid, x_shift) in neighbours(grid, cell_x, cell_y, bounds_x, cfg) {
      let mut other = particles.pos[pid];
      other.x += x_shift;
      let distance = (pos - other).length();
      let mass = particles.mass[pid];
      density += mass * density_kernel_ref(params.smoothing_radius, distance);
      near_density += mass * near_density_kernel_ref(params.smoothing_radius, distance);
    }

    particles.density[i] = density
      + solid_density_at_pos_ref(pos, soil, geom.soil_cell_size, params.target_density, cfg);
    particles.near_density[i] = near_density;
  }
}

/// Launch helper; see `sim::Sim::step` for the order within a step.
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &crate::particles::SphDevice,
  grid: &GridDevice,
  soil: &crate::soil::SoilDevice,
  params: &cubecl_runtime::server::Handle,
  cfg: Cfg,
) {
  calculate_particle_density::launch::<R>(
    client,
    super::cube_count(cfg.num_particles as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::grid_args(grid, &cfg),
    super::soil_args(soil, &cfg),
    super::whole(params, super::PARAM_COUNT),
    cfg,
  );
}

#[cfg(test)]
mod tests {
  use super::super::test_support::{Harness, assert_close};

  #[test]
  fn matches_reference() {
    let h = Harness::new();
    h.build_grid();
    super::launch(
      &h.client,
      &h.sph,
      &h.grid,
      &h.soil_dev,
      &h.params_buf,
      h.cfg,
    );
    let actual = h.read_particles();

    let mut expected = h.particles.clone();
    super::calculate_particle_density_ref(
      &mut expected,
      &h.grid_ref(),
      &h.soil.cells,
      &h.geom,
      &h.params,
      &h.cfg,
    );

    assert_close(&actual.density, &expected.density, 1e-6, "density");
    assert_close(
      &actual.near_density,
      &expected.near_density,
      1e-6,
      "near_density",
    );
  }

  #[test]
  fn soil_raises_density_inside_a_column() {
    let h = Harness::new();
    h.build_grid();
    super::launch(
      &h.client,
      &h.sph,
      &h.grid,
      &h.soil_dev,
      &h.params_buf,
      h.cfg,
    );
    let actual = h.read_particles();
    // The capillary-test terrain is solid above the pool, so at least one
    // liquid particle must be carrying a solid fraction.
    assert!(
      actual
        .density
        .iter()
        .any(|d| *d > h.params.target_density * 0.2),
      "no particle picked up a soil solid fraction"
    );
  }
}
