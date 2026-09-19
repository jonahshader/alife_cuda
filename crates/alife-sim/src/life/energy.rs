//! The energy budget: one unit per organism, serial over its particles.
//!
//! `docs/organism.md`, *World coupling and life cycle*: "Leaf particles gain
//! energy from light... Root particles draw water from their soil cell's
//! saturation." What a body costs is its volume.
//!
//! ```text
//! gain   = SUM_leaf light_gain * light(cell of particle)
//!        + SUM_root water_gain * max(0, water - 1)
//! upkeep = upkeep_per_volume * SUM_limb volume(limb)
//! energy += (gain - upkeep) * life_interval
//! ```
//!
//! **Why volume rather than particles.** A limb's volume is
//! `grown * limb_segment_length * thickness^2` with the thickness tied to the
//! square root of its progress (`crate::bodies::limb_volume`), so a tall thin
//! plant is cheaper than a tall thick one and a half-grown limb costs a
//! quarter of what it will (`docs/organism.md`, *Growth*). Gain is still per
//! particle: what a leaf catches and what a root drinks are surface
//! phenomena, and a particle is the only place either is sampled. A seed in
//! flight is in no limb, so it has no volume and costs nothing.
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
//! 1-particle leaf. Fully grown, at `limb_segment_length` 0.15 and the
//! `--thickness-*` defaults (root 0.04, stem 0.05, leaf 0.03), its volume is
//! `2*0.15*0.04^2 + 3*0.15*0.05^2 + 1*0.15*0.03^2` = 0.00174, so at
//! `upkeep_per_volume` 10 it costs 0.0174 a step — the 0.018 six particles
//! cost at the `upkeep_per_particle` 0.003 this replaced, which is what that
//! default was chosen to reproduce. At `light_gain` 0.012 and `water_gain`
//! 0.004 it gains `0.012 * light + 0.008 * wet - 0.0174` per step, so it
//! breaks even in open light with a wet root (wetness about 0.75, which
//! saturated soil reaches) and loses about 0.006 with a dry one or in shade.
//! The per-particle figure was the one constant the previous chunk moved off
//! the brief's, and the run is why: at 0.001 open light alone paid for the
//! whole body, nothing ever starved, and a 3000-step run at `--founders 64`
//! recorded 3 deaths against 99 births and filled every organism slot by step
//! 400 — a population with no selection in it. At 0.003 the same run recorded
//! 78 deaths against 68 births, ended at 93 of 256 slots, and 45 of the 64
//! founder lineages were already extinct (`docs/perf.md`).
//!
//! A seedling that has germinated but not yet grown a leaf is pure upkeep
//! plus whatever its root drinks, so it lives about as long as its
//! `seed_energy` lasts unless it grows one. Growth is funded by what is
//! left over `seed_cost`, and at these constants a seedling's root barely
//! covers its own upkeep, so it mostly does not: `docs/organism.md`,
//! decisions, and the `TODO.md` item it is the other half of.
//!
//! One unit per organism and a serial walk over its particles: no float
//! atomic, so the sum order is the slot order and the result is the same on
//! every run (`docs/organism.md`, decisions).

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::light::{LightGrid, soil_cell_index, soil_cell_index_ref};
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE, limb_volume};
use crate::genome::PartType;
use crate::genome::population::{LimbArgs, Population, limb_args};
use crate::kernels::constraints::{limb_len, particle_at};
use crate::kernels::{
  CUBE_DIM, Cfg, P_LIFE_INTERVAL, P_LIGHT_GAIN, P_LIMB_SEGMENT_LENGTH, P_SOIL_SIZE,
  P_TARGET_DENSITY, P_UPKEEP_PER_VOLUME, P_WATER_GAIN, PARAM_COUNT, cube_count, whole,
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
  grown: &[f32],
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
  let segment_length = params[P_LIMB_SEGMENT_LENGTH as usize];

  let mut gain = 0.0f32;
  let mut volume = 0.0f32;
  for l in 0..ml {
    let n = limb_len(map, pbase, l, body);
    let record = (base + l) as usize;
    let part_type = limbs.part_type[record];

    // `crate::bodies::limb_volume`, inlined: the record's length capped as
    // growth caps it, the thickness from the progress, the volume from both.
    if part_type != 0u32 {
      let mut target = limbs.length[record] as f32;
      if target > comptime!(mp as f32) {
        target = comptime!(mp as f32);
      }
      if target > 0.0f32 {
        let progress = f32::clamp(grown[record] / target, 0.0f32, 1.0f32);
        let thickness = limbs.target_thickness[record] * f32::sqrt(progress);
        volume += grown[record] * segment_length * thickness * thickness;
      }
    }

    for i in 0..n {
      let id = particle_at(map, pbase, l, i, body);
      if id != NO_PARTICLE {
        let id = id as usize;
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

  let upkeep = params[P_UPKEEP_PER_VOLUME as usize] * volume;
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
    whole(&bodies.device.grown, body.limb_count()),
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
    let mut volume = 0.0f32;
    for l in 0..ml {
      let record = o * ml + l;
      let part_type = pop.limbs.part_type[record];
      if !part_type.is_present() {
        continue;
      }
      let target = (pop.limbs.length[record] as usize).min(cfg.max_particles_per_limb as usize);
      volume += limb_volume(
        params.limb_segment_length,
        pop.limbs.target_thickness[record],
        target as f32,
        bodies.grown[cfg.limb_index(o, l)],
      );
      let n = bodies.limb_len(o, l);
      for i in 0..n {
        let id = bodies.particle(o, l, i);
        if id == NO_PARTICLE {
          continue;
        }
        let id = id as usize;
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
    let upkeep = params.upkeep_per_volume * volume;
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
      &h.soil_dev,
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
