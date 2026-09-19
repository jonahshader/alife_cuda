//! `calculate_accel`: pressure, near-pressure, viscosity, capillary suction
//! and soil friction, integrated into the velocity.

use cubecl::prelude::*;

use super::grid::GridDevice;
use super::soil_sample::{friction, properties_at_pos, properties_at_pos_ref};
use super::{
  Cfg, GridArgs, P_BOUNDS_X, P_CAPILLARY_MULT, P_CELL_SIZE, P_DT, P_GRAVITY, P_NEAR_PRESSURE_MULT,
  P_PRESSURE_MULT, P_SMOOTHING_RADIUS, P_SOIL_SIZE, P_TARGET_DENSITY, P_VISCOSITY_STRENGTH,
  SoilArgs, SphArgs, clamp_i32, density_kernel_gradient_scale, density_kernel_gradient_scale_ref,
  particle_to_cid, viscosity_kernel, viscosity_kernel_ref,
};
use crate::particles::{ParticleKind, SphHost};
use crate::soil::SoilHost;

/// Writes into `vel_next` rather than into `sph.vel`, which the caller then
/// swaps in.
///
/// The C++ writes `sph.vel[pid]` at the end of the same kernel whose viscosity
/// term reads `sph.vel[particle_id]` for every neighbour, so a thread sees a
/// mix of this step's and last step's velocities depending on scheduling. That
/// is a data race, and one of the reasons two C++ runs at one seed diverge.
/// Reading only last step's velocities is deterministic and is what the
/// viscosity term means.
#[cube(launch)]
pub fn calculate_accel(
  sph: &SphArgs,
  vel_next: &mut [f32],
  grid: &GridArgs,
  soil: &SoilArgs,
  params: &[f32],
  #[comptime] cfg: Cfg,
) {
  let pid = ABSOLUTE_POS;
  if pid >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[pid] != 0u32 {
    // Vapor is not accelerated here, but `vel_next` becomes `vel`, so it
    // still has to carry the value forward.
    vel_next[2 * pid] = sph.vel[2 * pid];
    vel_next[2 * pid + 1] = sph.vel[2 * pid + 1];
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let smoothing_radius = params[P_SMOOTHING_RADIUS as usize];
  let bounds_x = params[P_BOUNDS_X as usize];
  let target_density = params[P_TARGET_DENSITY as usize];
  let pressure_mult = params[P_PRESSURE_MULT as usize];
  let near_pressure_mult = params[P_NEAR_PRESSURE_MULT as usize];
  let soil_size = params[P_SOIL_SIZE as usize];

  let pos_x = sph.pos[2 * pid];
  let pos_y = sph.pos[2 * pid + 1];
  let vel_x = sph.vel[2 * pid];
  let vel_y = sph.vel[2 * pid + 1];
  let density = sph.density[pid];
  let near_density = sph.near_density[pid];

  let grid_index = particle_to_cid(pos_x, pos_y, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  // capillary action: modify effective target density based on local saturation.
  // In unsaturated soil, raise the target so pressure becomes negative (suction),
  // which draws water in via SPH pressure gradients. No grid feedback loop.
  let soil_props = properties_at_pos(pos_x, pos_y, soil, soil_size, target_density, cfg);
  let local_water_density = density - soil_props.solid_density;
  let mut local_sat = 1.0f32;
  if soil_props.pore_capacity > 1e-6f32 {
    local_sat = local_water_density / soil_props.pore_capacity;
  }
  let mut unsaturated = 1.0f32 - local_sat;
  if unsaturated < 0.0f32 {
    unsaturated = 0.0f32;
  }
  let capillary_bonus =
    soil_props.capillary_strength * unsaturated * params[P_CAPILLARY_MULT as usize];
  let effective_target = target_density + capillary_bonus;

  let pressure = pressure_mult * (density - effective_target);
  let near_pressure = near_pressure_mult * near_density;
  let total_pressure = pressure + near_pressure;

  let mut pressure_force_x = 0.0f32;
  let mut pressure_force_y = 0.0f32;
  let mut viscosity_force_x = 0.0f32;
  let mut viscosity_force_y = 0.0f32;

  // iterate through cell neighborhood
  for dy in 0..3u32 {
    let yi = cell_y + dy as i32 - 1;
    // skip if cell is out of vertical bounds
    if yi >= 0 && yi < cfg.grid_h {
      for dx in 0..3u32 {
        let xi = cell_x + dx as i32 - 1;
        // wrap x if out of horizontal bounds
        let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
        let neighbour_index = (yi * cfg.grid_w + wrapped_x) as usize;

        let start = grid.cell_start[neighbour_index] as usize;
        let mut num_particles = grid.cell_counts[neighbour_index];
        if num_particles > cfg.max_per_cell {
          num_particles = cfg.max_per_cell;
        }

        // iterate through particles within the cell
        for k in 0..num_particles {
          let particle_id = grid.sorted_ids[start + k as usize] as usize;
          let mut other_x = sph.pos[2 * particle_id];
          let other_y = sph.pos[2 * particle_id + 1];
          if xi < 0 {
            other_x -= bounds_x;
          } else if xi >= cfg.grid_w {
            other_x += bounds_x;
          }
          let other_density = sph.density[particle_id];
          let other_pressure = pressure_mult * (other_density - target_density);
          let other_near_pressure = near_pressure_mult * sph.near_density[particle_id];
          let other_mass = sph.mass[particle_id];

          let offset_x = pos_x - other_x;
          let offset_y = pos_y - other_y;
          let dst2 = offset_x * offset_x + offset_y * offset_y;
          let dst = f32::sqrt(dst2);

          let weight = other_mass
            * ((total_pressure + other_pressure + other_near_pressure) / (4.0f32 * other_density))
            * density_kernel_gradient_scale(smoothing_radius, dst);
          pressure_force_x -= weight * offset_x;
          pressure_force_y -= weight * offset_y;

          // viscosity
          if particle_id != pid {
            let influence = viscosity_kernel(smoothing_radius, dst2);
            // scale with mass?
            viscosity_force_x += influence * (sph.vel[2 * particle_id] - vel_x);
            viscosity_force_y += influence * (sph.vel[2 * particle_id + 1] - vel_y);
          }
        }
      }
    }
  }

  pressure_force_x -= total_pressure * soil_props.solid_density_gradient_x * 0.5f32 / density;
  pressure_force_y -= total_pressure * soil_props.solid_density_gradient_y * 0.5f32 / density;

  let viscosity_strength = params[P_VISCOSITY_STRENGTH as usize];
  let mut acc_x = (pressure_force_x + viscosity_force_x * viscosity_strength) / density;
  let mut acc_y = params[P_GRAVITY as usize]
    + (pressure_force_y + viscosity_force_y * viscosity_strength) / density;

  // The C++ indexes the soil with an unclamped `floor(pos / soil_size)`, so a
  // particle sitting exactly on the ceiling reads one row past the grid.
  let soil_x = clamp_i32(f32::floor(pos_x / soil_size) as i32, 0, cfg.soil_w - 1);
  let soil_y = clamp_i32(f32::floor(pos_y / soil_size) as i32, 0, cfg.soil_h - 1);
  let soil_friction = friction(soil, (soil_y * cfg.soil_w + soil_x) as usize);
  acc_x -= vel_x * soil_friction;
  acc_y -= vel_y * soil_friction;

  let dt = params[P_DT as usize];
  vel_next[2 * pid] = vel_x + acc_x * dt;
  vel_next[2 * pid + 1] = vel_y + acc_y * dt;
}

/// Plain-Rust reference for [`calculate_accel`].
pub fn calculate_accel_ref(
  particles: &mut SphHost,
  grid: &super::grid::GridRef,
  soil: &SoilHost,
  geom: &crate::world::WorldGeometry,
  params: &crate::SimParams,
  cfg: &Cfg,
) {
  let bounds_x = geom.bounds.x;
  let positions = particles.pos.clone();
  let velocities = particles.vel.clone();

  for pid in 0..particles.len() {
    if particles.state[pid] != ParticleKind::Liquid {
      continue;
    }
    let pos = positions[pid];
    let vel = velocities[pid];
    let density = particles.density[pid];
    let near_density = particles.near_density[pid];

    let grid_index = super::particle_to_cid_ref(pos, geom.cell_size, cfg);
    let cell_x = (grid_index % cfg.grid_w as u32) as i32;
    let cell_y = (grid_index / cfg.grid_w as u32) as i32;

    let soil_props =
      properties_at_pos_ref(pos, soil, geom.soil_cell_size, params.target_density, cfg);
    let local_water_density = density - soil_props.solid_density;
    let local_sat = if soil_props.pore_capacity > 1e-6 {
      local_water_density / soil_props.pore_capacity
    } else {
      1.0
    };
    let capillary_bonus =
      soil_props.capillary_strength * (1.0 - local_sat).max(0.0) * params.capillary_mult;
    let effective_target = params.target_density + capillary_bonus;

    let pressure = params.pressure_mult * (density - effective_target);
    let near_pressure = params.near_pressure_mult * near_density;
    let total_pressure = pressure + near_pressure;

    let mut pressure_force = glam::Vec2::ZERO;
    let mut viscosity_force = glam::Vec2::ZERO;

    for dy in 0..3i32 {
      let yi = cell_y + dy - 1;
      if yi < 0 || yi >= cfg.grid_h {
        continue;
      }
      for dx in 0..3i32 {
        let xi = cell_x + dx - 1;
        let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
        let neighbour_index = (yi * cfg.grid_w + wrapped_x) as usize;
        let start = grid.cell_start[neighbour_index] as usize;
        let count = grid.cell_counts[neighbour_index].min(cfg.max_per_cell);

        for k in 0..count as usize {
          let particle_id = grid.sorted_ids[start + k] as usize;
          let mut other = positions[particle_id];
          if xi < 0 {
            other.x -= bounds_x;
          } else if xi >= cfg.grid_w {
            other.x += bounds_x;
          }
          let other_density = particles.density[particle_id];
          let other_pressure = params.pressure_mult * (other_density - params.target_density);
          let other_near_pressure = params.near_pressure_mult * particles.near_density[particle_id];
          let other_mass = particles.mass[particle_id];

          let offset = pos - other;
          let dst2 = offset.length_squared();
          let dst = dst2.sqrt();

          let weight = other_mass
            * ((total_pressure + other_pressure + other_near_pressure) / (4.0 * other_density))
            * density_kernel_gradient_scale_ref(params.smoothing_radius, dst);
          pressure_force -= offset * weight;

          if particle_id != pid {
            let influence = viscosity_kernel_ref(params.smoothing_radius, dst2);
            viscosity_force += (velocities[particle_id] - vel) * influence;
          }
        }
      }
    }

    pressure_force -= soil_props.solid_density_gradient * (total_pressure * 0.5 / density);

    let mut acc = glam::Vec2::new(0.0, params.gravity)
      + (pressure_force + viscosity_force * params.viscosity_strength) / density;

    let soil_x = ((pos.x / geom.soil_cell_size).floor() as i32).clamp(0, cfg.soil_w - 1);
    let soil_y = ((pos.y / geom.soil_cell_size).floor() as i32).clamp(0, cfg.soil_h - 1);
    acc -= vel * crate::soil::friction(soil, (soil_y * cfg.soil_w + soil_x) as usize);

    particles.vel[pid] = vel + acc * params.dt;
  }
}

#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &crate::particles::SphDevice,
  vel_next: &cubecl_runtime::server::Handle,
  grid: &GridDevice,
  soil: &crate::soil::SoilDevice,
  params: &cubecl_runtime::server::Handle,
  cfg: Cfg,
) {
  calculate_accel::launch::<R>(
    client,
    super::cube_count(cfg.num_particles as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::whole(vel_next, cfg.num_particles as usize * 2),
    super::grid_args(grid, &cfg),
    super::soil_args(soil, &cfg),
    super::whole(params, super::PARAM_COUNT),
    cfg,
  );
}

#[cfg(test)]
mod tests {
  use super::super::test_support::{Harness, assert_close_vec2};
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
    // The reference needs the same densities the kernel just wrote.
    let seeded = h.read_particles();
    h.particles = seeded;
    h.sph = SphDevice::upload(&h.client, &h.particles);

    super::launch(
      &h.client,
      &h.sph,
      &h.vel_next,
      &h.grid,
      &h.soil_dev,
      &h.params_buf,
      h.cfg,
    );
    let actual: Vec<glam::Vec2> =
      crate::soa::download_field(&h.client, &h.vel_next, h.particles.len());

    let mut expected = h.particles.clone();
    super::calculate_accel_ref(
      &mut expected,
      &h.grid_ref(),
      &h.soil.cells,
      &h.geom,
      &h.params,
      &h.cfg,
    );

    // Vapor is not accelerated; the kernel copies it forward so the
    // swapped-in buffer stays whole.
    assert_close_vec2(&actual, &expected.vel, 1e-5, "vel after accel");
  }
}
