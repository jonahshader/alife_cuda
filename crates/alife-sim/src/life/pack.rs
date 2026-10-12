//! Everything the host decision pass needs, in one buffer and one readback.
//!
//! The life tick is decided on the host — which organisms grow, seed,
//! germinate and die — and the host is not allowed to be expensive about it.
//! Rather than pulling the heads buffer (`max_organisms x max_limbs x
//! HEAD_DIM` floats) and several particle fields back, one unit per organism
//! packs what the decisions actually read into `[max_organisms x (LIFE_FIXED
//! + 2 * max_limbs)]` floats — 41 KB at the defaults, one `read`:
//!
//! - the organism's energy, water pool and transpired-but-not-emitted water,
//!   which the energy kernel just wrote, and how many particles its roots
//!   drank this tick;
//! - its seed particle's position and speed, for germination;
//! - the top of its body, where a seed is emitted from;
//! - per limb, the worst its base joint has been bent since the last tick,
//!   which is what a limb breaks on (`docs/organism.md`, *Growth*);
//! - per limb, the brain's growth logit, which the allocation adds to the
//!   record's `growth_bias` before the softmax.
//!
//! The two per-limb runs are separate slices rather than interleaved, so a
//! host walk over either one is contiguous.
//!
//! The per-limb bend maximum is *reset* here, for every slot alive or not:
//! this is the one reader of it, and the interval it covers is the one that
//! just ended. The drink count is reset the same way and for the same reason:
//! the energy kernel has already added it to the pool, and the host counts it
//! from here. The logit is only read; the brain rewrites it every step. A
//! free slot packs as zeros, and so reads as "no energy, no seed, no bend, no
//! logit", which is what the host does nothing about.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE};
use crate::brain::{BrainState, HEAD_DIM, HEAD_GROWTH};
use crate::genome::PartType;
use crate::genome::Population;
use crate::kernels::constraints::{limb_len, particle_at};
use crate::kernels::{CUBE_DIM, cube_count, whole};
use crate::particles::SphDevice;

/// Slots of one organism's packed entry, before the two per-limb runs.
pub const LIFE_ENERGY: usize = 0;
pub const LIFE_SEED_X: usize = 1;
pub const LIFE_SEED_Y: usize = 2;
pub const LIFE_SEED_SPEED: usize = 3;
pub const LIFE_TOP_X: usize = 4;
pub const LIFE_TOP_Y: usize = 5;
pub const LIFE_WATER: usize = 6;
pub const LIFE_TRANSPIRED: usize = 7;
/// Particles drunk this tick, as a float: exact, since a tick's drinks are
/// far below 2^24.
pub const LIFE_DRINKS: usize = 8;
pub const LIFE_FIXED: usize = 9;

/// `x` of an organism that holds no seed particle. Outside the world, as
/// [`crate::particles::PARKED_POS`] is, so it cannot be mistaken for a place.
pub const NO_SEED_X: f32 = -1.0;

const TYPE_STEM: u32 = PartType::Stem as u32;
const TYPE_LEAF: u32 = PartType::Leaf as u32;

/// Floats per organism in the packed buffer: the fixed slots, then one bend
/// maximum per limb, then one growth logit per limb.
pub fn stride(body: BodyCfg) -> usize {
  LIFE_FIXED + 2 * body.max_limbs as usize
}

/// Where a limb's worst base-joint bend sits in an organism's entry.
pub fn bend_slot(limb: usize) -> usize {
  LIFE_FIXED + limb
}

/// Where a limb's growth logit sits in an organism's entry.
pub fn logit_slot(body: BodyCfg, limb: usize) -> usize {
  LIFE_FIXED + body.max_limbs as usize + limb
}

/// One unit per organism slot, serial over its limbs.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn pack_life(
  ppos: &[f32],
  vel: &[f32],
  limbs: &crate::genome::population::LimbArgs,
  map: &[u32],
  alive: &[u32],
  energy: &[f32],
  water: &[f32],
  transpired: &[f32],
  drinks: &mut [u32],
  seed_particles: &[u32],
  anchors: &[f32],
  heads: &[f32],
  bend_max: &mut [f32],
  out: &mut [f32],
  #[comptime] body: BodyCfg,
) {
  let o = ABSOLUTE_POS as u32;
  if o >= comptime!(body.max_organisms) {
    terminate!();
  }
  let ml = comptime!(body.max_limbs);
  let mp = comptime!(body.max_particles_per_limb);
  let width = comptime!((LIFE_FIXED as u32) + 2u32 * body.max_limbs);
  let base = (o * width) as usize;
  let base_limb = o * ml;
  for k in 0..width {
    out[base + k as usize] = 0.0f32;
  }
  // Read and reset before the `alive` guard: a slot that died during the
  // interval must not hand its bend on to whatever germinates into it next.
  // A dead slot's maxima are already zero — the constraint pass skips it —
  // so the packed value is the same zero the loop above wrote, and so is its
  // logit, which the heads kernel zeroes for a slot it skips.
  for l in 0..ml {
    let slot = (base_limb + l) as usize;
    out[base + LIFE_FIXED + l as usize] = bend_max[slot];
    bend_max[slot] = 0.0f32;
    out[base + LIFE_FIXED + (ml + l) as usize] =
      heads[slot * comptime!(HEAD_DIM) + comptime!(HEAD_GROWTH)];
  }
  out[base + LIFE_DRINKS] = drinks[o as usize] as f32;
  drinks[o as usize] = 0u32;
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  out[base + LIFE_ENERGY] = energy[o as usize];
  out[base + LIFE_WATER] = water[o as usize];
  out[base + LIFE_TRANSPIRED] = transpired[o as usize];

  let seed = seed_particles[o as usize];
  if seed == NO_PARTICLE {
    out[base + LIFE_SEED_X] = NO_SEED_X;
  } else {
    let s = seed as usize;
    out[base + LIFE_SEED_X] = ppos[2 * s];
    out[base + LIFE_SEED_Y] = ppos[2 * s + 1];
    out[base + LIFE_SEED_SPEED] =
      f32::sqrt(vel[2 * s] * vel[2 * s] + vel[2 * s + 1] * vel[2 * s + 1]);
  }

  // Where a seed is emitted from: the highest stem or leaf particle, or the
  // anchor for a body that has neither yet.
  let mut top_x = anchors[2 * o as usize];
  let mut top_y = anchors[2 * o as usize + 1];
  let mut found = false;
  let pbase = o * ml * mp;
  for l in 0..ml {
    let n = limb_len(map, pbase, l, body);
    let part_type = limbs.part_type[(base_limb + l) as usize];
    if part_type == TYPE_STEM || part_type == TYPE_LEAF {
      for i in 0..n {
        let id = particle_at(map, pbase, l, i, body);
        if id != NO_PARTICLE {
          let y = ppos[2 * id as usize + 1];
          if !found || y > top_y {
            top_x = ppos[2 * id as usize];
            top_y = y;
            found = true;
          }
        }
      }
    }
  }
  out[base + LIFE_TOP_X] = top_x;
  out[base + LIFE_TOP_Y] = top_y;
}

/// Scratch for the packed readback, allocated once.
#[derive(Debug, Clone)]
pub struct LifePack {
  pub buffer: Handle,
  pub len: usize,
}

impl LifePack {
  pub fn new<R: Runtime>(client: &ComputeClient<R>, body: BodyCfg) -> Self {
    let len = body.max_organisms as usize * stride(body);
    Self {
      buffer: client.empty(len * size_of::<f32>()),
      len,
    }
  }
}

/// Pack and read back. One `read`, and the only one the decision pass takes.
///
/// Resets the per-limb bend maxima and the per-organism drink counts on the
/// device as it goes, so the interval the next tick reads starts here.
#[allow(clippy::too_many_arguments)]
pub fn read<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  bodies: &BodyState,
  pop: &Population,
  brain: &BrainState,
  drinks: &Handle,
  pack: &LifePack,
  num_particles: usize,
) -> Vec<f32> {
  let body = bodies.cfg;
  pack_life::launch::<R>(
    client,
    cube_count(body.max_organisms as usize),
    CubeDim::new_1d(CUBE_DIM),
    whole(&sph.ppos, num_particles * 2),
    whole(&sph.vel, num_particles * 2),
    crate::genome::population::limb_args(&pop.device.limbs),
    whole(&bodies.device.limb_particles, body.particle_map_len()),
    whole(&pop.device.organisms.alive, body.max_organisms as usize),
    whole(&pop.device.organisms.energy, body.max_organisms as usize),
    whole(&pop.device.organisms.water, body.max_organisms as usize),
    whole(
      &pop.device.organisms.transpired,
      body.max_organisms as usize,
    ),
    whole(drinks, body.max_organisms as usize),
    whole(&bodies.device.seed_particles, body.max_organisms as usize),
    whole(&bodies.device.anchors, body.max_organisms as usize * 2),
    whole(&brain.device.heads, brain.cfg.heads_len()),
    whole(&bodies.device.bend_max, body.limb_count()),
    whole(&pack.buffer, pack.len),
    body,
  );
  crate::genome::population::read_f32(client, &pack.buffer, pack.len)
}

/// Plain-Rust twin of [`pack_life`], resetting `bend_max` and `drinks` as the
/// kernel does.
pub fn pack_life_ref(
  particles: &crate::particles::SphHost,
  pop: &Population,
  bodies: &BodyState,
  heads: &[f32],
  bend_max: &mut [f32],
  drinks: &mut [u32],
) -> Vec<f32> {
  let body = bodies.cfg;
  let ml = body.max_limbs as usize;
  let width = stride(body);
  let mut out = vec![0.0f32; body.max_organisms as usize * width];

  for o in 0..body.max_organisms as usize {
    let base = o * width;
    for l in 0..ml {
      out[base + bend_slot(l)] = std::mem::take(&mut bend_max[o * ml + l]);
      out[base + logit_slot(body, l)] = heads[(o * ml + l) * HEAD_DIM + HEAD_GROWTH];
    }
    out[base + LIFE_DRINKS] = std::mem::take(&mut drinks[o]) as f32;
    if pop.organisms.alive[o] == 0 {
      continue;
    }
    out[base + LIFE_ENERGY] = pop.organisms.energy[o];
    out[base + LIFE_WATER] = pop.organisms.water[o];
    out[base + LIFE_TRANSPIRED] = pop.organisms.transpired[o];

    let seed = bodies.seed_particles[o];
    if seed == NO_PARTICLE {
      out[base + LIFE_SEED_X] = NO_SEED_X;
    } else {
      let s = seed as usize;
      out[base + LIFE_SEED_X] = particles.ppos[s].x;
      out[base + LIFE_SEED_Y] = particles.ppos[s].y;
      out[base + LIFE_SEED_SPEED] = particles.vel[s].length();
    }

    let mut top = bodies.anchors[o];
    let mut found = false;
    for l in 0..ml {
      let record = o * ml + l;
      let part_type = pop.limbs.part_type[record];
      if !matches!(part_type, PartType::Stem | PartType::Leaf) {
        continue;
      }
      let n = bodies.limb_len(o, l);
      for i in 0..n {
        let id = bodies.particle(o, l, i);
        if id == NO_PARTICLE {
          continue;
        }
        let p = particles.ppos[id as usize];
        if !found || p.y > top.y {
          top = p;
          found = true;
        }
      }
    }
    out[base + LIFE_TOP_X] = top.x;
    out[base + LIFE_TOP_Y] = top.y;
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  /// Distinct bend maxima, as if the constraint pass had accumulated them.
  fn fixture_bend_max(h: &OrganismHarness) -> Vec<f32> {
    (0..h.body_cfg.limb_count())
      .map(|i| 0.05 * (i as f32 + 1.0))
      .collect()
  }

  /// One brain tick, so the heads hold something other than the zeros the
  /// buffer was allocated with.
  fn run_brain(h: &OrganismHarness) -> Vec<f32> {
    h.run_brain();
    h.brain.read_heads(&h.client)
  }

  /// A drink counter for the fixture's two slots, uploaded.
  fn drinks(h: &OrganismHarness, counts: &[u32]) -> Handle {
    h.client.create_from_slice(bytemuck::cast_slice(counts))
  }

  #[test]
  fn matches_reference() {
    let mut h = OrganismHarness::new();
    h.pop.organisms.energy[0] = 1.75;
    h.pop.organisms.water[0] = 4.5;
    h.pop.organisms.transpired[0] = 0.25;
    // Give the organism a seed particle of its own, so both branches run.
    let seed = (h.geom.num_particles - 1) as u32;
    h.bodies.seed_particles[0] = seed;
    h.particles.ppos[seed as usize] = glam::Vec2::new(1.5, 0.75);
    h.particles.vel[seed as usize] = glam::Vec2::new(0.3, -0.4);
    h.sph = crate::particles::SphDevice::upload(&h.client, &h.particles);
    h.pop.upload(&h.client);
    h.bodies.upload(&h.client);
    let mut bend = fixture_bend_max(&h);
    h.bodies.device.bend_max = h.client.create_from_slice(bytemuck::cast_slice(&bend));
    let heads = run_brain(&h);

    let pack = LifePack::new(&h.client, h.body_cfg);
    let mut counts = vec![3u32, 0];
    let drunk = drinks(&h, &counts);
    let actual = read(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.brain,
      &drunk,
      &pack,
      h.geom.num_particles,
    );
    let expected = pack_life_ref(
      &h.particles,
      &h.pop,
      &h.bodies,
      &heads,
      &mut bend,
      &mut counts,
    );
    assert_eq!(actual.len(), expected.len());
    assert_close(&actual, &expected, 1e-6, "life pack");

    let width = stride(h.body_cfg);
    assert_eq!(actual[LIFE_ENERGY], 1.75);
    assert_eq!(actual[LIFE_WATER], 4.5);
    assert_eq!(actual[LIFE_TRANSPIRED], 0.25);
    assert_eq!(actual[LIFE_DRINKS], 3.0);
    // Read once, gone: the count is the tick's, and the host has it now.
    let left: Vec<u32> = crate::soa::download_field(&h.client, &drunk, 2);
    assert_eq!(left, vec![0, 0]);
    assert_eq!(counts, vec![0, 0]);
    assert_eq!(actual[LIFE_SEED_X], 1.5);
    assert!((actual[LIFE_SEED_SPEED] - 0.5).abs() < 1e-6);
    // The free slot packs as zeros — no energy, no seed — but its bend slots
    // are read and cleared like everyone else's.
    assert!(actual[width..width + LIFE_FIXED].iter().all(|v| *v == 0.0));
  }

  /// The per-limb slots carry the bend maxima, and reading them clears them:
  /// the next tick measures its own interval, not the sum of two.
  #[test]
  fn the_bend_maxima_are_read_and_reset() {
    let mut h = OrganismHarness::new();
    let ml = h.body_cfg.max_limbs as usize;
    let bend = fixture_bend_max(&h);
    h.bodies.device.bend_max = h.client.create_from_slice(bytemuck::cast_slice(&bend));
    let heads = run_brain(&h);

    let pack = LifePack::new(&h.client, h.body_cfg);
    let drunk = drinks(&h, &[0, 0]);
    let first = read(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.brain,
      &drunk,
      &pack,
      h.geom.num_particles,
    );
    let width = stride(h.body_cfg);
    assert_eq!(width, LIFE_FIXED + 2 * ml);
    for l in 0..ml {
      assert_eq!(first[bend_slot(l)], bend[l], "limb {l}");
    }

    let second = read(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.brain,
      &drunk,
      &pack,
      h.geom.num_particles,
    );
    assert!(
      second[bend_slot(0)..bend_slot(ml)]
        .iter()
        .all(|v| *v == 0.0),
      "the maxima survived the tick that read them: {:?}",
      &second[bend_slot(0)..bend_slot(ml)]
    );

    // And the host twin resets its own copy the same way.
    let mut host = bend.clone();
    let packed = pack_life_ref(
      &h.particles,
      &h.pop,
      &h.bodies,
      &heads,
      &mut host,
      &mut [0, 0],
    );
    assert_eq!(&packed[bend_slot(0)..bend_slot(ml)], &bend[..ml]);
    assert!(host.iter().all(|v| *v == 0.0));
  }

  /// The growth logits ride in a per-limb slice of their own, straight out of
  /// the heads buffer, and — unlike the bend maxima — reading them leaves
  /// them alone: the brain writes them afresh every step.
  #[test]
  fn the_growth_logits_come_from_the_heads_buffer() {
    let h = OrganismHarness::new();
    let ml = h.body_cfg.max_limbs as usize;
    let heads = run_brain(&h);

    let pack = LifePack::new(&h.client, h.body_cfg);
    let drunk = drinks(&h, &[0, 0]);
    let packed = read(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.brain,
      &drunk,
      &pack,
      h.geom.num_particles,
    );

    let mut nonzero = 0;
    for o in 0..h.body_cfg.max_organisms as usize {
      for l in 0..ml {
        let logit = heads[(o * ml + l) * HEAD_DIM + HEAD_GROWTH];
        assert_eq!(
          packed[o * stride(h.body_cfg) + logit_slot(h.body_cfg, l)],
          logit,
          "organism {o} limb {l}"
        );
        nonzero += (logit != 0.0) as usize;
      }
    }
    assert!(
      nonzero > 0,
      "no limb had a logit, so the test proves nothing"
    );

    let again = read(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.brain,
      &drunk,
      &pack,
      h.geom.num_particles,
    );
    let logits = |p: &[f32]| p[logit_slot(h.body_cfg, 0)..logit_slot(h.body_cfg, ml)].to_vec();
    assert_eq!(logits(&again), logits(&packed), "a read cleared the logits");
  }
}
