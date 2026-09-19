//! The sensor buffer: what a limb feels of the world this tick.
//!
//! `[max_organisms x max_limbs x SENSOR_DIM]`, in the spec's order — light,
//! water, soil solid fraction, contact, energy (`docs/organism.md`,
//! *Implementation layout*, the brain-shape defaults).
//!
//! Every reading is taken at the limb's **first** particle, the one its base
//! joint holds, and from `ppos` rather than `pos`: `ppos` is where the
//! constraint pass actually left the particle, and it is what
//! [`crate::kernels::limb_geometry`] reports positions against, so the
//! spatial features and the sensors describe the same body.
//!
//! An absent limb, or a free organism slot, is written as zeros rather than
//! left alone — a stale reading from a dead organism would otherwise be fed
//! to whatever slot comes next.
//!
//! Light comes from the grid [`crate::life::light`] rebuilds every
//! `life_interval` steps, so between rebuilds it is that many steps stale;
//! energy is the organism's budget divided by `seed_threshold`, so 1 means
//! "ready to seed".

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use super::BrainCfg;
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE};
use crate::genome::Population;
use crate::genome::shape::SENSOR_DIM;
use crate::kernels::soil_sample::{solid_fraction_at_pos, solid_fraction_at_pos_ref};
use crate::kernels::{
  Cfg, P_SEED_THRESHOLD, P_SOIL_SIZE, P_TARGET_DENSITY, SoilArgs, cube_count, soil_args, whole,
};
use crate::life::light::{LightGrid, soil_cell_index, soil_cell_index_ref};
use crate::particles::{SphDevice, SphHost};
use crate::soil::{SoilDevice, SoilHost};

/// Slots of a limb's sensor vector.
pub const S_LIGHT: usize = 0;
pub const S_WATER: usize = 1;
pub const S_SOIL: usize = 2;
pub const S_CONTACT: usize = 3;
pub const S_ENERGY: usize = 4;

/// Solid fraction above which a limb counts as in contact with soil.
///
/// Pure clay's solid fraction is exactly 0.50 (`1 - CLAY_POROSITY`), so a limb
/// buried in undiluted clay reads *no* contact while the same limb in sand
/// (0.62) or silt (0.55) reads contact. That is the spec's threshold taken
/// literally; whether contact should instead come from soil presence, which
/// would not have the edge, is still open and belongs to the first milestone
/// that acts on the channel — the life cycle reads soil through
/// `solid_fraction_at_pos` directly rather than through this flag.
pub const CONTACT_THRESHOLD: f32 = 0.5;

/// One unit per `(organism, limb)`.
// The water clamp is spelled out: `f32::clamp` is a Rust method, not
// something the kernel language offers — the same reason
// `kernels::soil_sample` spells its clamps out.
#[cube(launch)]
#[allow(clippy::too_many_arguments, clippy::manual_clamp)]
pub fn write_sensors(
  ppos: &[f32],
  density: &[f32],
  part_type: &[u32],
  map: &[u32],
  alive: &[u32],
  energy: &[f32],
  soil: &SoilArgs,
  light: &[f32],
  sensors: &mut [f32],
  params: &[f32],
  #[comptime] body: BodyCfg,
  #[comptime] world: Cfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(body.max_organisms * body.max_limbs) {
    terminate!();
  }
  let sd = comptime!(SENSOR_DIM as u32);
  let base = (t * sd) as usize;
  for k in 0..sd {
    sensors[base + k as usize] = 0.0f32;
  }

  let ml = comptime!(body.max_limbs);
  let mp = comptime!(body.max_particles_per_limb);
  let o = t / ml;
  let l = t % ml;
  if alive[o as usize] == 0u32 || part_type[t as usize] == 0u32 {
    terminate!();
  }

  let first = map[(o * ml * mp + l * mp) as usize];
  if first == NO_PARTICLE {
    terminate!();
  }
  let first = first as usize;
  let px = ppos[2 * first];
  let py = ppos[2 * first + 1];

  // Water, as the SPH density the fluid just computed, in units of the rest
  // density. That density carries the soil's solid contribution too
  // (`kernels::density`), so the reading sits near 1 in free air rather than
  // near 0; the clamp is what keeps a crushed particle from saturating the
  // token.
  let mut water = density[first] / params[P_TARGET_DENSITY as usize];
  if water < 0.0f32 {
    water = 0.0f32;
  }
  if water > 2.0f32 {
    water = 2.0f32;
  }

  let solid = solid_fraction_at_pos(px, py, soil, params[P_SOIL_SIZE as usize], world);
  let mut contact = 0.0f32;
  if solid > CONTACT_THRESHOLD {
    contact = 1.0f32;
  }

  // Light at the limb's own cell, from the grid the life tick rebuilt. It is
  // up to `life_interval` steps old between rebuilds, which is what a canopy
  // that moves at a plant's pace is worth.
  let cell = soil_cell_index(px, py, params[P_SOIL_SIZE as usize], world);

  sensors[base + S_LIGHT] = light[cell as usize];
  sensors[base + S_WATER] = water;
  sensors[base + S_SOIL] = solid;
  sensors[base + S_CONTACT] = contact;
  // Normalized by the energy at which the organism reproduces, so 1 is "ready
  // to seed" whatever the budget's absolute scale.
  sensors[base + S_ENERGY] = energy[o as usize] / params[P_SEED_THRESHOLD as usize];
}

#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  soil: &SoilDevice,
  bodies: &BodyState,
  pop: &Population,
  light: &LightGrid,
  brain: &super::BrainDevice,
  params: &Handle,
  cfg: BrainCfg,
  world: Cfg,
) {
  let body = bodies.cfg;
  let n = world.num_particles as usize;
  write_sensors::launch::<R>(
    client,
    cube_count(cfg.limbs()),
    CubeDim::new_1d(crate::kernels::CUBE_DIM),
    whole(&sph.ppos, n * 2),
    whole(&sph.density, n),
    whole(&pop.device.limbs.part_type, cfg.limbs()),
    whole(&bodies.device.limb_particles, body.particle_map_len()),
    whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
    whole(&pop.device.organisms.energy, cfg.max_organisms as usize),
    soil_args(soil, &world),
    whole(&light.light, light.cells),
    whole(&brain.sensors, cfg.sensors_len()),
    whole(params, crate::kernels::PARAM_COUNT),
    body,
    world,
  );
}

/// Plain-Rust twin of [`write_sensors`].
pub fn write_sensors_ref(
  particles: &SphHost,
  pop: &Population,
  bodies: &BodyState,
  soil: &SoilHost,
  light: &[f32],
  params: &SimParams,
  geom: &crate::world::WorldGeometry,
) -> Vec<f32> {
  let cfg = bodies.cfg;
  let world = geom.cfg();
  let ml = cfg.max_limbs as usize;
  let mut out = vec![0.0f32; cfg.limb_count() * SENSOR_DIM];

  for o in 0..cfg.max_organisms as usize {
    for l in 0..ml {
      let t = o * ml + l;
      if pop.organisms.alive[o] == 0 || !pop.limbs.part_type[t].is_present() {
        continue;
      }
      let first = bodies.particle(o, l, 0);
      if first == NO_PARTICLE {
        continue;
      }
      let pos: Vec2 = particles.ppos[first as usize];
      let water = (particles.density[first as usize] / params.target_density).clamp(0.0, 2.0);
      let solid = solid_fraction_at_pos_ref(pos, soil, geom.soil_cell_size, &world);

      let cell = soil_cell_index_ref(pos, geom.soil_cell_size, &world);

      let base = t * SENSOR_DIM;
      out[base + S_LIGHT] = light[cell];
      out[base + S_WATER] = water;
      out[base + S_SOIL] = solid;
      out[base + S_CONTACT] = if solid > CONTACT_THRESHOLD { 1.0 } else { 0.0 };
      out[base + S_ENERGY] = pop.organisms.energy[o] / params.seed_threshold;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  fn reference(h: &OrganismHarness) -> Vec<f32> {
    write_sensors_ref(
      &h.particles,
      &h.pop,
      &h.bodies,
      &h.soil.cells,
      &h.light.read(&h.client),
      &h.params,
      &h.geom,
    )
  }

  #[test]
  fn matches_reference() {
    let mut h = OrganismHarness::new();
    // Energy and light are the two channels that come from the life tick, so
    // the fixture gives the organism a budget rather than leaving it at zero.
    h.pop.organisms.energy[0] = 1.25;
    h.pop.upload(&h.client);
    h.run_sense();
    let actual = h.brain.read_sensors(&h.client);
    let expected = reference(&h);
    assert_eq!(actual.len(), expected.len());
    assert_close(&actual, &expected, 1e-6, "sensors");
    assert_eq!(
      actual[S_ENERGY],
      1.25 / h.params.seed_threshold,
      "the root limb's energy channel is the organism's budget, normalized"
    );
  }

  /// The fixture's plant stands on the capillary-test terrain, so its root is
  /// in soil and its leaf is not: the soil and contact channels have to tell
  /// them apart, and every channel has to stay in its range.
  #[test]
  fn every_channel_stays_in_range() {
    let h = OrganismHarness::new();
    let expected = reference(&h);
    for t in 0..h.body_cfg.limb_count() {
      let s = &expected[t * SENSOR_DIM..(t + 1) * SENSOR_DIM];
      assert!(s[S_LIGHT] >= 0.0 && s[S_LIGHT] <= h.params.light_top);
      assert!(s[S_WATER] >= 0.0 && s[S_WATER] <= 2.0);
      assert!(s[S_SOIL] >= 0.0 && s[S_SOIL] <= 1.0);
      assert!(s[S_CONTACT] == 0.0 || s[S_CONTACT] == 1.0);
      assert_eq!(s[S_CONTACT] == 1.0, s[S_SOIL] > CONTACT_THRESHOLD);
    }
    // The fixture's plant is root, stem, leaf in records 0..3 of organism 0,
    // and organism 1's slot is free: everything else reads as zeros.
    for t in 3..h.body_cfg.limb_count() {
      assert!(
        expected[t * SENSOR_DIM..(t + 1) * SENSOR_DIM]
          .iter()
          .all(|v| *v == 0.0),
        "limb {t} is absent but was written"
      );
    }
  }
}
