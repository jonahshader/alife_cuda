//! `calculate_evap_prob`: the vertical density gradient per liquid particle,
//! suppressed where the soil is solid.

use cubecl::prelude::*;

use super::grid::{GridDevice, neighbour_cell, neighbours, unwrap_x};
use super::soil_sample::{presence_at_pos, presence_at_pos_ref};
use super::{
  Cfg, GridArgs, P_BOUNDS_X, P_CELL_SIZE, P_SMOOTHING_RADIUS, P_SOIL_SIZE, SoilArgs, SphArgs,
  density_kernel_gradient_component, density_kernel_gradient_component_ref, particle_to_cid,
};
use crate::particles::{ParticleKind, SphHost};
use crate::soil::SoilHost;

/// Compute the vertical density gradient per liquid particle → `evap_prob`
/// (raw surface signal).
#[cube(launch)]
pub fn calculate_evap_prob(
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
    sph.evap_prob[i] = 0.0f32;
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let smoothing_radius = params[P_SMOOTHING_RADIUS as usize];
  let bounds_x = params[P_BOUNDS_X as usize];

  let pos_x = sph.pos[2 * i];
  let pos_y = sph.pos[2 * i + 1];

  // suppress evaporation inside soil — check total soil material
  // (sand+silt+clay), which is 1.0 in soil cells and 0.0 in air, with a
  // smooth bilinear transition at the surface
  let soil_factor = 1.0f32 - presence_at_pos(pos_x, pos_y, soil, params[P_SOIL_SIZE as usize], cfg);

  let grid_index = particle_to_cid(pos_x, pos_y, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  // sum m_j * density_kernel_gradient().y over neighbors
  let mut drho_dy = 0.0f32;

  for dy in 0..3u32 {
    for dx in 0..3u32 {
      let cell = neighbour_cell(grid, cell_x, cell_y, dx, dy, bounds_x, cfg);
      for k in 0..cell.count {
        let pid = grid.sorted_ids[cell.start as usize + k as usize] as usize;
        let other_x = unwrap_x(sph.pos[2 * pid], cell.x_shift);
        let other_y = sph.pos[2 * pid + 1];
        let diff_x = pos_x - other_x;
        let diff_y = pos_y - other_y;
        let dst = f32::sqrt(diff_x * diff_x + diff_y * diff_y);
        let grad_y = density_kernel_gradient_component(smoothing_radius, diff_y, dst);
        drho_dy += sph.mass[pid] * grad_y;
      }
    }
  }

  // Surface particles have neighbors below → grad.y < 0 → drho_dy < 0
  // Negate so surface gets positive probability.
  // Store raw surface signal (no evap_rate) — evap_rate * dt applied in evaporate kernel.
  let mut denom = sph.density[i];
  if denom < 1e-6f32 {
    denom = 1e-6f32;
  }
  let normalized_grad = -drho_dy / denom;
  let mut prob = normalized_grad;
  if prob < 0.0f32 {
    prob = 0.0f32;
  }
  sph.evap_prob[i] = prob * soil_factor;
}

/// Plain-Rust reference for [`calculate_evap_prob`].
pub fn calculate_evap_prob_ref(
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
      particles.evap_prob[i] = 0.0;
      continue;
    }
    let pos = particles.pos[i];
    let soil_factor = 1.0 - presence_at_pos_ref(pos, soil, geom.soil_cell_size, cfg);

    let grid_index = super::particle_to_cid_ref(pos, geom.cell_size, cfg);
    let cell_x = (grid_index % cfg.grid_w as u32) as i32;
    let cell_y = (grid_index / cfg.grid_w as u32) as i32;

    let mut drho_dy = 0.0f32;
    for (pid, x_shift) in neighbours(grid, cell_x, cell_y, bounds_x, cfg) {
      let mut other = particles.pos[pid];
      other.x += x_shift;
      let diff = pos - other;
      let grad_y =
        density_kernel_gradient_component_ref(params.smoothing_radius, diff.y, diff.length());
      drho_dy += particles.mass[pid] * grad_y;
    }

    let normalized_grad = -drho_dy / particles.density[i].max(1e-6);
    particles.evap_prob[i] = normalized_grad.max(0.0) * soil_factor;
  }
}

pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &crate::particles::SphDevice,
  grid: &GridDevice,
  soil: &crate::soil::SoilDevice,
  params: &cubecl_runtime::server::Handle,
  cfg: Cfg,
) {
  calculate_evap_prob::launch::<R>(
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
  use crate::particles::SphDevice;

  #[test]
  fn matches_reference() {
    let mut h = Harness::new();
    h.build_grid();
    super::super::density::launch(
      &h.client,
      &h.sph,
      &h.grid,
      &h.soil_dev,
      &h.params_buf,
      h.cfg,
    );
    h.particles = h.read_particles();
    h.sph = SphDevice::upload(&h.client, &h.particles);

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
    super::calculate_evap_prob_ref(
      &mut expected,
      &h.grid_ref(),
      &h.soil.cells,
      &h.geom,
      &h.params,
      &h.cfg,
    );

    assert_close(&actual.evap_prob, &expected.evap_prob, 1e-5, "evap_prob");
    assert!(
      actual.evap_prob.iter().any(|p| *p > 0.0),
      "no particle got a surface signal, so the test proves nothing"
    );
  }
}
