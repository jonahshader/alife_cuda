//! The energy and water budget: one unit per organism, serial over its
//! particles.
//!
//! `docs/organism.md`, *World coupling and life cycle* and *Water*. Leaves
//! earn from light, but only as far as the organism's water pool pays for it;
//! what a body costs is its volume.
//!
//! ```text
//! water     += drinks                                  (this tick's, whole particles)
//! income     = SUM_leaf light_gain * light(cell) * segment_fraction(grown, k)
//!              * life_interval
//! earned     = min(income, water / water_per_energy)
//! water     -= earned * water_per_energy
//! transpired += earned * water_per_energy
//! energy    += earned - upkeep_per_volume * SUM_limb volume(limb) * life_interval
//! ```
//!
//! **Why roots earn nothing.** A root's job is the water the drink kernel
//! takes out of the fluid ([`super::drink`]); the energy is the leaf's. When
//! roots earned energy from wetness, a root-only plant on a wet spot paid for
//! itself forever and nothing ever needed a leaf. Now a plant with no leaf
//! has no income, and a plant with leaves and no water has none either — the
//! pool is what ties the two halves of a body together.
//!
//! **Why the water is spent here and not on the host.** The pool is credited
//! and spent by the same unit that computes the income, so the cap is exact
//! and no float crosses the bus to be clamped there. What was spent goes to
//! `transpired`, which the host empties a whole particle at a time as vapor
//! at the organism's top (`life::Sim::apply`).
//!
//! **Why volume rather than particles.** A limb's volume is
//! `grown * limb_segment_length * thickness^2` with the thickness tied to the
//! square root of its progress (`crate::bodies::limb_volume`), so a tall thin
//! plant is cheaper than a tall thick one and a half-grown limb costs a
//! quarter of what it will (`docs/organism.md`, *Growth*). Income is still per
//! particle: what a leaf catches is a surface phenomenon, and a particle is
//! the only place light is sampled. A leaf particle earns in proportion to
//! how far its segment has grown — `segment_fraction(grown, k)`, the same
//! share the constraint pass gives its rest length and the light grid its
//! shade — so a limb's last particle does not earn a whole particle's light
//! the moment it is claimed. A seed in flight is in no limb, so it has no
//! volume and costs nothing.
//!
//! **The balance at the defaults.** The founder plant
//! (`Genome::seed_plant`) is a 2-particle root, a 3-particle stem and a
//! 1-particle leaf; fully grown at `limb_segment_length` 0.15 and the
//! `--thickness-*` defaults its volume is 0.00174, so at `upkeep_per_volume`
//! 10 it costs 0.0174 a step. Its one leaf in open light earns
//! `light_gain` 0.012 a step, so a founder-shaped body runs at -0.0054 a step
//! even with a full pool and starves from `seed_energy` 3.0 in about 550
//! steps: on light alone, a plant has to carry more leaf per unit of volume
//! than the founder does to pay for itself. Its pool holds
//! `water_capacity_per_volume` 3000 times its volume, 5.2 particles, and the
//! leaf spends 0.12 of them a tick at `water_per_energy` 1, so a full pool
//! lasts about 43 ticks without a drink.
//!
//! One unit per organism and a serial walk over its particles: no float
//! atomic, so the sum order is the slot order and the result is the same on
//! every run (`docs/organism.md`, decisions). The drink count is an integer
//! the pack kernel reads and clears ([`super::pack`]); this only adds it.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::light::{LightGrid, soil_cell_index, soil_cell_index_ref};
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE, limb_volume};
use crate::genome::PartType;
use crate::genome::population::{LimbArgs, Population, limb_args};
use crate::kernels::constraints::{limb_len, particle_at, segment_fraction, segment_fraction_ref};
use crate::kernels::{
  CUBE_DIM, Cfg, P_LIFE_INTERVAL, P_LIGHT_GAIN, P_LIMB_SEGMENT_LENGTH, P_SOIL_SIZE,
  P_UPKEEP_PER_VOLUME, P_WATER_PER_ENERGY, PARAM_COUNT, cube_count, whole,
};
use crate::particles::{SphDevice, SphHost};

const TYPE_LEAF: u32 = PartType::Leaf as u32;

/// One unit per organism slot. A free slot is left alone.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn organism_energy(
  ppos: &[f32],
  limbs: &LimbArgs,
  map: &[u32],
  grown: &[f32],
  alive: &[u32],
  drinks: &[u32],
  energy: &mut [f32],
  water: &mut [f32],
  transpired: &mut [f32],
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
  let light_gain = params[P_LIGHT_GAIN as usize];
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

    if part_type == TYPE_LEAF {
      for i in 0..n {
        let id = particle_at(map, pbase, l, i, body);
        if id != NO_PARTICLE {
          let id = id as usize;
          let cell = soil_cell_index(ppos[2 * id], ppos[2 * id + 1], soil_size, world);
          gain += light_gain * light[cell as usize] * segment_fraction(grown[record], i);
        }
      }
    }
  }

  let interval = params[P_LIFE_INTERVAL as usize];
  let water_per_energy = params[P_WATER_PER_ENERGY as usize];
  let mut pool = water[o as usize] + drinks[o as usize] as f32;
  let mut earned = gain * interval;
  let affordable = pool / water_per_energy;
  if earned > affordable {
    earned = affordable;
  }
  let spent = earned * water_per_energy;
  // `earned` is at most `pool / water_per_energy`, so this is at least zero
  // up to a rounding; the floor keeps a rounding from becoming a debt.
  pool = f32::max(pool - spent, 0.0f32);
  water[o as usize] = pool;
  transpired[o as usize] += spent;

  let upkeep = params[P_UPKEEP_PER_VOLUME as usize] * volume;
  energy[o as usize] += earned - upkeep * interval;
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
  drinks: &Handle,
  params: &Handle,
  body: BodyCfg,
  world: Cfg,
) {
  let n = world.num_particles as usize;
  let organisms = body.max_organisms as usize;
  organism_energy::launch::<R>(
    client,
    cube_count(organisms),
    CubeDim::new_1d(CUBE_DIM),
    whole(&sph.ppos, n * 2),
    limb_args(&pop.device.limbs),
    whole(&bodies.device.limb_particles, body.particle_map_len()),
    whole(&bodies.device.grown, body.limb_count()),
    whole(&pop.device.organisms.alive, organisms),
    whole(drinks, organisms),
    whole(&pop.device.organisms.energy, organisms),
    whole(&pop.device.organisms.water, organisms),
    whole(&pop.device.organisms.transpired, organisms),
    whole(&light.light, light.cells),
    whole(params, PARAM_COUNT),
    body,
    world,
  );
}

/// What [`organism_energy`] leaves behind, per organism slot.
#[derive(Debug, Clone, PartialEq)]
pub struct Budget {
  pub energy: Vec<f32>,
  pub water: Vec<f32>,
  pub transpired: Vec<f32>,
}

/// Plain-Rust twin of [`organism_energy`], over the host copies.
#[allow(clippy::needless_range_loop)]
pub fn organism_energy_ref(
  particles: &SphHost,
  pop: &Population,
  bodies: &BodyState,
  drinks: &[u32],
  light: &[f32],
  params: &SimParams,
  geom: &crate::world::WorldGeometry,
) -> Budget {
  let cfg = bodies.cfg;
  let world = geom.cfg();
  let ml = cfg.max_limbs as usize;
  let mut out = Budget {
    energy: pop.organisms.energy.clone(),
    water: pop.organisms.water.clone(),
    transpired: pop.organisms.transpired.clone(),
  };

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
      if part_type != PartType::Leaf {
        continue;
      }
      for i in 0..bodies.limb_len(o, l) {
        let id = bodies.particle(o, l, i);
        if id == NO_PARTICLE {
          continue;
        }
        let cell = soil_cell_index_ref(particles.ppos[id as usize], geom.soil_cell_size, &world);
        let share = segment_fraction_ref(bodies.grown[cfg.limb_index(o, l)], i);
        gain += params.light_gain * light[cell] * share;
      }
    }
    let interval = params.life_interval as f32;
    let pool = out.water[o] + drinks[o] as f32;
    let earned = (gain * interval).min(pool / params.water_per_energy);
    let spent = earned * params.water_per_energy;
    out.water[o] = (pool - spent).max(0.0);
    out.transpired[o] += spent;
    let upkeep = params.upkeep_per_volume * volume;
    out.energy[o] += earned - upkeep * interval;
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  fn run(h: &OrganismHarness, light: &LightGrid, drinks: &[u32]) -> Budget {
    let drinks = h.client.create_from_slice(bytemuck::cast_slice(drinks));
    launch(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      light,
      &drinks,
      &h.params_buf,
      h.body_cfg,
      h.cfg,
    );
    let n = h.pop.max_organisms;
    let read = |handle| crate::genome::population::read_f32(&h.client, handle, n);
    let organisms = &h.pop.device.organisms;
    Budget {
      energy: read(&organisms.energy),
      water: read(&organisms.water),
      transpired: read(&organisms.transpired),
    }
  }

  /// Full light in every cell, as a grid is before its first rebuild. The
  /// fixture's capillary terrain buries its leaf, and a rebuilt grid would
  /// read it zero; what is under test here is the budget, not the occlusion
  /// (`super::super::light` tests that).
  fn lit(h: &OrganismHarness) -> (LightGrid, Vec<f32>) {
    let light = LightGrid::new(&h.client, &h.soil, h.params.light_top);
    let host = light.read(&h.client);
    (light, host)
  }

  /// Twice: once with water to spare, where light alone sets the income,
  /// and once with a pool too small for it, where the pool does.
  #[test]
  fn matches_reference() {
    for (pool, drunk) in [(50.0f32, 2u32), (0.05, 0)] {
      let mut h = OrganismHarness::new();
      // Non-zero starting values, so a kernel that overwrote rather than
      // accumulated would show up.
      h.pop.organisms.energy[0] = 0.75;
      h.pop.organisms.water[0] = pool;
      h.pop.organisms.transpired[0] = 0.5;
      h.pop.upload(&h.client);
      let (light, light_host) = lit(&h);
      let drinks = [drunk, 0];

      let actual = run(&h, &light, &drinks);
      let expected = organism_energy_ref(
        &h.particles,
        &h.pop,
        &h.bodies,
        &drinks,
        &light_host,
        &h.params,
        &h.geom,
      );
      assert_close(&actual.energy, &expected.energy, 1e-6, "energy");
      assert_close(&actual.water, &expected.water, 1e-6, "water");
      assert_close(&actual.transpired, &expected.transpired, 1e-6, "transpired");
      assert_ne!(actual.energy[0], 0.75, "the organism's budget moved");
      assert!(actual.transpired[0] > 0.5, "the leaf spent no water");
      // Water is conserved through the kernel: what the pool lost is what
      // was transpired.
      let lost = pool + drunk as f32 - actual.water[0];
      assert!((lost - (actual.transpired[0] - 0.5)).abs() < 1e-5);
    }
  }

  /// A free slot is never touched, and the whole budget scales with the
  /// interval it covers.
  #[test]
  fn a_free_slot_is_left_alone_and_the_interval_scales_the_budget() {
    let mut h = OrganismHarness::new();
    h.pop.organisms.energy[1] = -12.5;
    h.pop.organisms.water[0] = 1000.0;
    h.pop.upload(&h.client);
    let (light, _) = lit(&h);
    let one = run(&h, &light, &[0, 0]);
    assert_eq!(one.energy[1], -12.5, "slot 1 is free");

    h.params.life_interval *= 3;
    h.params_buf = h
      .client
      .create_from_slice(bytemuck::cast_slice(&crate::kernels::pack_params(
        &h.params, &h.geom,
      )));
    h.pop.organisms.energy[0] = 0.0;
    h.pop.organisms.water[0] = 1000.0;
    h.pop.upload(&h.client);
    let three = run(&h, &light, &[0, 0]);
    // `one` started from zero energy too, so its delta is one interval's
    // worth.
    assert!((three.energy[0] - 3.0 * one.energy[0]).abs() <= 1e-5 * three.energy[0].abs().max(1.0));
  }

  /// A leaf particle earns as much of a particle's light as its segment has
  /// grown: the fixture's one-particle leaf half way into its segment
  /// transpires — earns — half of what it does whole, on the device and in
  /// the reference alike.
  #[test]
  fn a_half_grown_leaf_earns_half() {
    let earned = |leaf: f32| {
      let mut h = OrganismHarness::new();
      h.bodies.grown[h.body_cfg.limb_index(0, 2)] = leaf;
      h.bodies.upload(&h.client);
      h.pop.organisms.water[0] = 1000.0;
      h.pop.upload(&h.client);
      let (light, light_host) = lit(&h);
      let actual = run(&h, &light, &[0, 0]);
      let expected = organism_energy_ref(
        &h.particles,
        &h.pop,
        &h.bodies,
        &[0, 0],
        &light_host,
        &h.params,
        &h.geom,
      );
      assert_close(&actual.energy, &expected.energy, 1e-6, "energy");
      assert_close(&actual.transpired, &expected.transpired, 1e-6, "transpired");
      actual.transpired[0]
    };
    let (whole, half) = (earned(1.0), earned(0.5));
    assert!(whole > 0.0);
    assert!((half - 0.5 * whole).abs() < 1e-6, "{half} against {whole}");
  }

  /// No light, or no water, and the organism only loses: upkeep is all that
  /// is left. With no water at all, the leaf earns nothing however bright it
  /// is, and nothing is transpired.
  #[test]
  fn no_light_or_no_water_means_no_gain() {
    let mut h = OrganismHarness::new();
    h.pop.organisms.water[0] = 1000.0;
    h.pop.upload(&h.client);
    let dark = LightGrid::new(&h.client, &h.soil, 0.0);
    let budget = run(&h, &dark, &[0, 0]);
    assert!(budget.energy[0] < 0.0, "upkeep with nothing coming in");

    let h = OrganismHarness::new();
    let (light, _) = lit(&h);
    let dry = run(&h, &light, &[0, 0]);
    assert!(dry.energy[0] < 0.0, "a dry plant earned {}", dry.energy[0]);
    assert_eq!(dry.transpired[0], 0.0);
    assert_eq!(dry.energy[0], budget.energy[0], "dry is as poor as dark");
  }
}
