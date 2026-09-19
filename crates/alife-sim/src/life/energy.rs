//! The energy budget: one unit per organism, serial over its particles.
//!
//! `docs/organism.md`, *World coupling and life cycle*: "Leaf particles gain
//! energy from light... Root particles draw water from their soil cell's
//! saturation. Every particle costs energy per tick."
//!
//! ```text
//! gain   = SUM_leaf light_gain * light(cell of particle)
//!        + SUM_root water_gain * max(0, water - 1)
//! upkeep = upkeep_per_particle * particles
//! energy += (gain - upkeep) * life_interval
//! ```
//!
//! **Why `water - 1`.** `water` is the same quantity the brain's sensor
//! reads: `sph.density / target_density`. `kernels::density` adds the soil's
//! solid-density offset to every particle's density, and that offset is
//! exactly `target_density` in air, so the reading sits near 1 where there is
//! nothing to drink and rises above it where liquid is present. `water - 1`
//! is therefore the wetness, and it is negative in dry soil (whose offset is
//! `(1 - porosity) * target_density`, below the air value), which is why the
//! `max(0, ...)` is there rather than a clamp on `water` itself.
//!
//! **Why the defaults break a founder about even.** The founder plant
//! (`Genome::seed_plant`) is a 2-particle root, a 3-particle stem and a
//! 1-particle leaf: 6 particles. At `light_gain` 0.012, `water_gain` 0.004
//! and `upkeep_per_particle` 0.001 it gains
//! `0.012 * light + 0.004 * 2 * wet - 0.006` per step, so in open light with
//! dry roots it makes +0.006 and a fully shaded one loses 0.006. A seedling
//! that has germinated but not yet sprouted a leaf is pure upkeep and dies
//! unless its brain sprouts one, which is the selection pressure the sprout
//! head is under.
//!
//! One unit per organism and a serial walk over its particles: no float
//! atomic, so the sum order is the slot order and the result is the same on
//! every run (`docs/organism.md`, decisions).

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::light::{LightGrid, soil_cell_index, soil_cell_index_ref};
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE};
use crate::genome::PartType;
use crate::genome::population::{LimbArgs, Population, limb_args};
use crate::kernels::constraints::{limb_len, particle_at};
use crate::kernels::{
  CUBE_DIM, Cfg, P_LIFE_INTERVAL, P_LIGHT_GAIN, P_SOIL_SIZE, P_TARGET_DENSITY,
  P_UPKEEP_PER_PARTICLE, P_WATER_GAIN, PARAM_COUNT, cube_count, whole,
};
use crate::particles::{SphDevice, SphHost};

const TYPE_ROOT: u32 = PartType::Root as u32;
const TYPE_LEAF: u32 = PartType::Leaf as u32;

/// One unit per organism slot. A free slot is left alone.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn organism_energy(
  ppos: &[f32],
  density: &[f32],
  limbs: &LimbArgs,
  map: &[u32],
  alive: &[u32],
  energy: &mut [f32],
  light: &[f32],
  params: &[f32],
  #[comptime] body: BodyCfg,
  #[comptime] world: Cfg,
) {
  let o = ABSOLUTE_POS as u32;
  if o >= comptime!(body.max_organisms) {
    terminate!();
  }
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let ml = comptime!(body.max_limbs);
  let mp = comptime!(body.max_particles_per_limb);
  let base = o * ml;
  let pbase = o * ml * mp;

  let soil_size = params[P_SOIL_SIZE as usize];
  let target_density = params[P_TARGET_DENSITY as usize];
  let light_gain = params[P_LIGHT_GAIN as usize];
  let water_gain = params[P_WATER_GAIN as usize];

  let mut gain = 0.0f32;
  let mut particles = 0u32;
  for l in 0..ml {
    let n = limb_len(limbs, base + l, body);
    let part_type = limbs.part_type[(base + l) as usize];
    for i in 0..n {
      let id = particle_at(map, pbase, l, i, body);
      if id != NO_PARTICLE {
        let id = id as usize;
        particles += 1u32;
        if part_type == TYPE_LEAF {
          let cell = soil_cell_index(ppos[2 * id], ppos[2 * id + 1], soil_size, world);
          gain += light_gain * light[cell as usize];
        } else if part_type == TYPE_ROOT {
          let wetness = density[id] / target_density - 1.0f32;
          if wetness > 0.0f32 {
            gain += water_gain * wetness;
          }
        }
      }
    }
  }

  let upkeep = params[P_UPKEEP_PER_PARTICLE as usize] * particles as f32;
  energy[o as usize] += (gain - upkeep) * params[P_LIFE_INTERVAL as usize];
}

/// A seed particle is not in the limb map, so it neither gains nor costs.
/// That is deliberate: a seed is a package in flight, and its budget is the
/// `seed_lifetime` clock rather than energy.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  bodies: &BodyState,
  pop: &Population,
  light: &LightGrid,
  params: &Handle,
  body: BodyCfg,
  world: Cfg,
) {
  let n = world.num_particles as usize;
  organism_energy::launch::<R>(
    client,
    cube_count(body.max_organisms as usize),
    CubeDim::new_1d(CUBE_DIM),
    whole(&sph.ppos, n * 2),
    whole(&sph.density, n),
    limb_args(&pop.device.limbs),
    whole(&bodies.device.limb_particles, body.particle_map_len()),
    whole(&pop.device.organisms.alive, body.max_organisms as usize),
    whole(&pop.device.organisms.energy, body.max_organisms as usize),
    whole(&light.light, light.cells),
    whole(params, PARAM_COUNT),
    body,
    world,
  );
}

/// Plain-Rust twin of [`organism_energy`], over the host copies.
#[allow(clippy::needless_range_loop)]
pub fn organism_energy_ref(
  particles: &SphHost,
  pop: &Population,
  bodies: &BodyState,
  light: &[f32],
  params: &SimParams,
  geom: &crate::world::WorldGeometry,
) -> Vec<f32> {
  let cfg = bodies.cfg;
  let world = geom.cfg();
  let ml = cfg.max_limbs as usize;
  let mut energy = pop.organisms.energy.clone();

  for o in 0..cfg.max_organisms as usize {
    if pop.organisms.alive[o] == 0 {
      continue;
    }
    let mut gain = 0.0f32;
    let mut count = 0u32;
    for l in 0..ml {
      let record = o * ml + l;
      let part_type = pop.limbs.part_type[record];
      if !part_type.is_present() {
        continue;
      }
      let n = (pop.limbs.length[record] as usize).min(cfg.max_particles_per_limb as usize);
      for i in 0..n {
        let id = bodies.particle(o, l, i);
        if id == NO_PARTICLE {
          continue;
        }
        let id = id as usize;
        count += 1;
        match part_type {
          PartType::Leaf => {
            let cell = soil_cell_index_ref(particles.ppos[id], geom.soil_cell_size, &world);
            gain += params.light_gain * light[cell];
          }
          PartType::Root => {
            let wetness = particles.density[id] / params.target_density - 1.0;
            if wetness > 0.0 {
              gain += params.water_gain * wetness;
            }
          }
          _ => {}
        }
      }
    }
    let upkeep = params.upkeep_per_particle * count as f32;
    energy[o] += (gain - upkeep) * params.life_interval as f32;
  }
  energy
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  fn run(h: &OrganismHarness, light: &LightGrid) -> Vec<f32> {
    launch(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      light,
      &h.params_buf,
      h.body_cfg,
      h.cfg,
    );
    crate::genome::population::read_f32(
      &h.client,
      &h.pop.device.organisms.energy,
      h.pop.max_organisms,
    )
  }

  #[test]
  fn matches_reference() {
    let mut h = OrganismHarness::new();
    // A non-zero starting energy, so a kernel that overwrote rather than
    // accumulated would show up.
    h.pop.organisms.energy[0] = 0.75;
    h.pop.upload(&h.client);
    // The fixture never runs the density kernel, so every particle reads 0 and
    // the root branch would be dead code. Wet the first root particle and
    // leave the second dry, which is also the `max(0, water - 1)` guard.
    let (wet, dry) = (h.bodies.particle(0, 0, 0), h.bodies.particle(0, 0, 1));
    h.particles.density[wet as usize] = h.params.target_density * 1.6;
    h.particles.density[dry as usize] = h.params.target_density * 0.55;
    h.sph = crate::particles::SphDevice::upload(&h.client, &h.particles);

    let light = LightGrid::new(&h.client, &h.soil, h.params.light_top);
    super::super::light::launch(
      &h.client,
      &h.sph,
      &light,
      &h.params_buf,
      h.cfg,
      crate::kernels::LiveParticles(h.geom.num_particles),
    );
    let light_host = light.read(&h.client);

    let actual = run(&h, &light);
    let expected = organism_energy_ref(
      &h.particles,
      &h.pop,
      &h.bodies,
      &light_host,
      &h.params,
      &h.geom,
    );
    assert_close(&actual, &expected, 1e-6, "energy");
    assert_ne!(actual[0], 0.75, "the organism's budget moved");
  }

  /// A free slot is never touched, and the whole budget scales with the
  /// interval it covers.
  #[test]
  fn a_free_slot_is_left_alone_and_the_interval_scales_the_budget() {
    let mut h = OrganismHarness::new();
    h.pop.organisms.energy[1] = -12.5;
    h.pop.upload(&h.client);
    let light = LightGrid::new(&h.client, &h.soil, h.params.light_top);
    let one = run(&h, &light);
    assert_eq!(one[1], -12.5, "slot 1 is free");

    h.params.life_interval *= 3;
    h.params_buf = h
      .client
      .create_from_slice(bytemuck::cast_slice(&crate::kernels::pack_params(
        &h.params, &h.geom,
      )));
    h.pop.organisms.energy[0] = 0.0;
    h.pop.upload(&h.client);
    let three = run(&h, &light);
    // `one[0]` started from zero too, so its delta is one interval's worth.
    assert!((three[0] - 3.0 * one[0]).abs() <= 1e-5 * three[0].abs().max(1.0));
  }

  /// The fixture's leaf is in the open, so light is what pays for it: drop
  /// the light to zero and the organism only loses.
  #[test]
  fn no_light_means_no_gain() {
    let h = OrganismHarness::new();
    let dark = LightGrid::new(&h.client, &h.soil, 0.0);
    let energy = run(&h, &dark);
    assert!(
      energy[0] < 0.0,
      "upkeep with nothing coming in: {}",
      energy[0]
    );
  }
}
