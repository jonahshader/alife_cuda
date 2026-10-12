//! Roots drink: a liquid particle next to a thirsty root is taken out of the
//! fluid whole and credited to the root's organism.
//!
//! `docs/organism.md`, *Water*. One unit per liquid particle, on the life
//! tick, before the energy kernel: gather the 3x3 neighbour cells for the
//! nearest `Body` particle of part type `Root` within `smoothing_radius`
//! (the lowest particle id breaks a tie, so exactly one organism is ever in
//! the running), and if that root's organism is *thirsty* — a flag the host
//! uploads every tick, 1 while the organism's pool is below its capacity —
//! roll a Threefry draw keyed by the particle id, the step and
//! [`STREAM_DRINK`]. With probability `drink_rate` times the root particle's
//! `segment_fraction(grown, k)` the particle is drunk: its slot becomes
//! `Free`, exactly as `free_particles` leaves one, and `drinks[organism]`
//! goes up by one. The fraction is the same share a leaf particle earns light
//! by and a stem particle shades by: a root's newest particle drinks as much
//! as its segment has grown, not a whole particle's worth the moment it is
//! claimed.
//!
//! **Why the liquid particle decides.** A root reaching for the nearest
//! water would have two roots grab one particle, and settling that is a slot
//! grab — an atomic whose return value decides who wins, which the run's
//! reproducibility rules out. Turned around, every particle has exactly one
//! nearest root and decides its own fate alone, so no two units ever write
//! the same slot. The only shared write is the organism's drink count, an
//! integer `fetch_add`: exact and associative, so the order the units arrive
//! in cannot change it (`docs/organism.md`, decisions, 2026-09-19, "an
//! integer atomic is not the atomic the spec rules out"). The SPH kernels are
//! untouched; a drunk particle is simply not there next step.
//!
//! **Which grid.** The neighbour grid the step built at its start, read at
//! the positions the step ended on (`ppos`, which for a body is where the
//! constraint pass put it). A particle that crossed a cell boundary since the
//! build can be missed for one tick; that is deterministic and costs nothing.
//! Neighbours are tested for `Body` before their position is read, and a
//! body particle is written by no unit here, so a liquid neighbour being
//! freed by its own unit at the same time is never read.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::bodies::{BodyCfg, BodyState};
use crate::genome::PartType;
use crate::kernels::constraints::{segment_fraction, segment_fraction_ref};
use crate::kernels::grid::{GridDevice, GridRef, neighbour_cell, neighbours, unwrap_x};
use crate::kernels::spawn::free_slot;
use crate::kernels::{
  BodyArgs, CUBE_DIM, Cfg, GridArgs, KIND_BODY, KIND_LIQUID, LiveParticles, P_CELL_SIZE,
  P_DRINK_RATE, P_SMOOTHING_RADIUS, PARAM_COUNT, body_args, cube_count, grid_args, particle_to_cid,
  particle_to_cid_ref, whole,
};
use crate::particles::{ParticleKind, SphDevice, SphHost};
use crate::rng::{threefry4x32_20, threefry4x32_20_ref, u01, u01_ref};

/// The Threefry stream word a drink draws from. `evaporate_particles` keys
/// its draw with 1 and `move_vapor_particles` with 2 in the same position.
pub const STREAM_DRINK: u32 = 3;

const TYPE_ROOT: u32 = PartType::Root as u32;

/// One unit per particle slot; only a liquid one does anything.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn drink_particles(
  sph: &mut BodyArgs,
  grid: &GridArgs,
  thirsty: &[u32],
  grown: &[f32],
  drinks: &[Atomic<u32>],
  params: &[f32],
  step: u32,
  seed: u32,
  #[comptime] body: BodyCfg,
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != KIND_LIQUID {
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let radius = params[P_SMOOTHING_RADIUS as usize];
  let bounds_x = params[crate::kernels::P_BOUNDS_X as usize];
  let px = sph.ppos[2 * i];
  let py = sph.ppos[2 * i + 1];
  let grid_index = particle_to_cid(px, py, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  // Nearest root within the radius, by squared distance; a tie goes to the
  // lower particle id, so the winner does not depend on the visiting order.
  let mut best_d2 = radius * radius;
  let mut best = 0u32;
  let mut found = false;
  for dy in 0..3u32 {
    for dx in 0..3u32 {
      let cell = neighbour_cell(grid, cell_x, cell_y, dx, dy, bounds_x, cfg);
      for k in 0..cell.count {
        let pid = grid.sorted_ids[cell.start as usize + k as usize];
        let p = pid as usize;
        if sph.state[p] == KIND_BODY && sph.part_type[p] == TYPE_ROOT {
          let ox = unwrap_x(sph.ppos[2 * p], cell.x_shift);
          let oy = sph.ppos[2 * p + 1];
          let ddx = px - ox;
          let ddy = py - oy;
          let d2 = ddx * ddx + ddy * ddy;
          if d2 < best_d2 || (found && d2 == best_d2 && pid < best) {
            best_d2 = d2;
            best = pid;
            found = true;
          }
        }
      }
    }
  }
  if !found {
    terminate!();
  }
  let organism = sph.organism[best as usize];
  if thirsty[organism as usize] == 0u32 {
    terminate!();
  }

  let root = best as usize;
  let limb = organism * comptime!(body.max_limbs) + sph.limb[root];
  let share = segment_fraction(grown[limb as usize], sph.index_in_limb[root]);
  let r = threefry4x32_20(0u32, 0u32, 0u32, 0u32, i as u32, step, STREAM_DRINK, seed);
  if u01(r.x0) >= params[P_DRINK_RATE as usize] * share {
    terminate!();
  }

  drinks[organism as usize].fetch_add(1u32);
  // A drunk particle is a free slot like any other, and the next claim may
  // hand it to a body.
  free_slot(sph, i);
}

/// Launch over the live prefix. `thirsty` is one `u32` per organism slot and
/// `drinks` the per-organism counter the pack kernel reads and clears.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  grid: &GridDevice,
  bodies: &BodyState,
  thirsty: &Handle,
  drinks: &Handle,
  params: &Handle,
  step: u32,
  seed: u64,
  cfg: Cfg,
  live: LiveParticles,
) {
  let body = bodies.cfg;
  let organisms = body.max_organisms as usize;
  drink_particles::launch::<R>(
    client,
    cube_count(live.get()),
    CubeDim::new_1d(CUBE_DIM),
    body_args(sph),
    grid_args(grid, &cfg),
    whole(thirsty, organisms),
    whole(&bodies.device.grown, body.limb_count()),
    whole(drinks, organisms),
    whole(params, PARAM_COUNT),
    step,
    seed as u32,
    body,
    cfg,
  );
}

/// Plain-Rust twin of [`drink_particles`]: frees the drunk particles in
/// `particles` and returns the drinks per organism.
///
/// Sequential, which is the same result: a unit's decision reads only body
/// particles, and nothing here writes one.
#[allow(clippy::too_many_arguments)]
pub fn drink_particles_ref(
  particles: &mut SphHost,
  grid: &GridRef,
  bodies: &BodyState,
  thirsty: &[u32],
  params: &crate::SimParams,
  geom: &crate::world::WorldGeometry,
  step: u32,
  seed: u64,
) -> Vec<u32> {
  let cfg = geom.cfg();
  let mut drinks = vec![0u32; thirsty.len()];
  for i in 0..particles.len() {
    if particles.state[i] != ParticleKind::Liquid {
      continue;
    }
    let at = particles.ppos[i];
    let grid_index = particle_to_cid_ref(at, geom.cell_size, &cfg);
    let cell_x = (grid_index % cfg.grid_w as u32) as i32;
    let cell_y = (grid_index / cfg.grid_w as u32) as i32;

    let mut best: Option<(f32, usize)> = None;
    for (p, x_shift) in neighbours(grid, cell_x, cell_y, geom.bounds.x, &cfg) {
      if particles.state[p] != ParticleKind::Body || particles.part_type[p] != PartType::Root {
        continue;
      }
      let mut other = particles.ppos[p];
      other.x += x_shift;
      let d = at - other;
      let d2 = d.x * d.x + d.y * d.y;
      let radius2 = params.smoothing_radius * params.smoothing_radius;
      let better = match best {
        None => d2 < radius2,
        Some((best_d2, best_id)) => d2 < best_d2 || (d2 == best_d2 && p < best_id),
      };
      if better {
        best = Some((d2, p));
      }
    }
    let Some((_, root)) = best else {
      continue;
    };
    let organism = particles.organism[root] as usize;
    if thirsty[organism] == 0 {
      continue;
    }
    let grown = bodies.grown[bodies
      .cfg
      .limb_index(organism, particles.limb[root] as usize)];
    let share = segment_fraction_ref(grown, particles.index_in_limb[root] as usize);
    let r = threefry4x32_20_ref([0, 0, 0, 0], [i as u32, step, STREAM_DRINK, seed as u32]);
    if u01_ref(r[0]) >= params.drink_rate * share {
      continue;
    }
    drinks[organism] += 1;
    crate::kernels::spawn::free_particles_ref(particles, &[i as u32]);
  }
  drinks
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::OrganismHarness;

  /// The fixture's organism with liquid all around its roots: every liquid
  /// particle sits a short way off some fixture particle, so the nearest-root
  /// search, the tie-break and the thirst flag all have work to do.
  fn wet_harness(drink_rate: f32) -> (OrganismHarness, GridDevice) {
    let mut h = OrganismHarness::new();
    h.params.drink_rate = drink_rate;
    h.params_buf = h
      .client
      .create_from_slice(bytemuck::cast_slice(&crate::kernels::pack_params(
        &h.params, &h.geom,
      )));
    // Liquid by every body particle, and two equidistant from both root
    // particles so the tie-break is exercised.
    let roots = [h.bodies.particle(0, 0, 0), h.bodies.particle(0, 0, 1)];
    let mut spare = (0..h.geom.fluid_particles)
      .filter(|i| h.particles.state[*i] == ParticleKind::Liquid)
      .skip(6);
    for (k, root) in roots.iter().enumerate() {
      let at = h.particles.ppos[*root as usize];
      for offset in [
        glam::Vec2::new(0.03, 0.0),
        glam::Vec2::new(-0.02, 0.04),
        glam::Vec2::new(0.0, -0.05 - 0.01 * k as f32),
      ] {
        let i = spare.next().unwrap();
        h.particles.pos[i] = at + offset;
        h.particles.ppos[i] = at + offset;
      }
    }
    let (a, b) = (
      h.particles.ppos[roots[0] as usize],
      h.particles.ppos[roots[1] as usize],
    );
    let mid = 0.5 * (a + b);
    for i in spare.by_ref().take(2) {
      h.particles.pos[i] = mid;
      h.particles.ppos[i] = mid;
    }
    h.sph = SphDevice::upload(&h.client, &h.particles);
    let grid = GridDevice::alloc(&h.client, &h.cfg);
    crate::kernels::grid::build(
      &h.client,
      &h.sph,
      &grid,
      &h.params_buf,
      h.cfg,
      LiveParticles(h.geom.num_particles),
      &mut |_, f| f(),
    );
    (h, grid)
  }

  fn run(h: &OrganismHarness, grid: &GridDevice, thirsty: &[u32], step: u32) -> Vec<u32> {
    let thirsty_buf = h.client.create_from_slice(bytemuck::cast_slice(thirsty));
    let drinks = h
      .client
      .create_from_slice(bytemuck::cast_slice(&vec![0u32; thirsty.len()]));
    launch(
      &h.client,
      &h.sph,
      grid,
      &h.bodies,
      &thirsty_buf,
      &drinks,
      &h.params_buf,
      step,
      0xABCD_1234_5678,
      h.cfg,
      LiveParticles(h.geom.num_particles),
    );
    crate::soa::download_field(&h.client, &drinks, thirsty.len())
  }

  #[test]
  fn matches_reference() {
    let (h, grid) = wet_harness(0.5);
    // The fixture's stray body particles claim organisms 0 to 2, so the flag
    // buffer covers all three.
    let thirsty = vec![1u32, 0, 0];
    let actual = run(&h, &grid, &thirsty, 17);

    let mut expected = h.particles.clone();
    let grid_ref = crate::kernels::grid::build_ref(&h.particles, h.geom.cell_size, &h.cfg);
    let drinks = drink_particles_ref(
      &mut expected,
      &grid_ref,
      &h.bodies,
      &thirsty,
      &h.params,
      &h.geom,
      17,
      0xABCD_1234_5678,
    );
    assert_eq!(actual, drinks);
    assert_eq!(h.read_particles(), expected);
    assert!(
      drinks[0] > 0,
      "nothing was drunk, so the test proves nothing"
    );
    assert!(
      drinks[0] < 8,
      "every candidate was drunk at a rate of a half: {drinks:?}"
    );
    // A drunk slot is exactly a free one.
    assert_eq!(
      expected.count_of(ParticleKind::Liquid) + drinks[0] as usize,
      h.particles.count_of(ParticleKind::Liquid)
    );
  }

  /// A root's newest particle drinks as much as its segment has grown: at a
  /// rate of one, with the root a thousandth of the way into its second
  /// segment, the water whose nearest root is that second particle stays,
  /// and the device agrees with the reference about which.
  #[test]
  fn a_root_particle_drinks_as_much_as_its_segment_has_grown() {
    let (whole_h, grid) = wet_harness(1.0);
    let whole = run(&whole_h, &grid, &[1, 0, 0], 5)[0];

    let (mut h, grid) = wet_harness(1.0);
    h.bodies.grown[h.body_cfg.limb_index(0, 0)] = 1.001;
    h.bodies.upload(&h.client);
    let actual = run(&h, &grid, &[1, 0, 0], 5);
    let mut expected = h.particles.clone();
    let grid_ref = crate::kernels::grid::build_ref(&h.particles, h.geom.cell_size, &h.cfg);
    let drinks = drink_particles_ref(
      &mut expected,
      &grid_ref,
      &h.bodies,
      &[1, 0, 0],
      &h.params,
      &h.geom,
      5,
      0xABCD_1234_5678,
    );
    assert_eq!(actual, drinks);
    assert_eq!(h.read_particles(), expected);
    assert!(
      drinks[0] > 0 && drinks[0] < whole,
      "{} drank against {whole} whole",
      drinks[0]
    );
  }

  /// A root whose organism is not thirsty drinks nothing, and at a rate of
  /// one every particle within reach of a thirsty one is drunk.
  #[test]
  fn thirst_gates_the_drink_and_the_rate_is_a_probability() {
    let (h, grid) = wet_harness(1.0);
    assert_eq!(run(&h, &grid, &[0, 0, 0], 3), vec![0, 0, 0]);
    assert_eq!(h.read_particles(), h.particles, "a sated root drank");

    let (h, grid) = wet_harness(1.0);
    let drinks = run(&h, &grid, &[1, 0, 0], 3);
    // Every liquid particle whose nearest root in reach is organism 0's,
    // brute force over every root in the world. The world is wider than two
    // grid cells, so a root in reach is always in the 3x3 neighbourhood.
    let reach2 = h.params.smoothing_radius * h.params.smoothing_radius;
    let roots: Vec<usize> = (0..h.particles.len())
      .filter(|i| {
        h.particles.state[*i] == ParticleKind::Body && h.particles.part_type[*i] == PartType::Root
      })
      .collect();
    let bounds_x = h.geom.bounds.x;
    let d2 = |a: glam::Vec2, b: glam::Vec2| {
      let mut d = a - b;
      if d.x > 0.5 * bounds_x {
        d.x -= bounds_x;
      } else if d.x < -0.5 * bounds_x {
        d.x += bounds_x;
      }
      d.length_squared()
    };
    let expected = (0..h.particles.len())
      .filter(|i| h.particles.state[*i] == ParticleKind::Liquid)
      .filter(|i| {
        let at = h.particles.ppos[*i];
        let nearest = roots
          .iter()
          .map(|r| (d2(at, h.particles.ppos[*r]), *r))
          .filter(|(d, _)| *d < reach2)
          .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        nearest.is_some_and(|(_, r)| h.particles.organism[r] == 0)
      })
      .count();
    assert!(expected > 0);
    assert_eq!(drinks[0] as usize, expected);
  }
}
