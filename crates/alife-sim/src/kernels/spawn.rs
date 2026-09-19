//! The kernels a body spawn needs: which particle slots are claimable, where
//! a limb's particles go, and turning claimed slots into body particles.
//!
//! Slot allocation is a prefix scan over an occupancy flag, never an atomic
//! claim (`docs/organism.md`, *Implementation layout*): the ids have to come
//! out ascending and in the same order every run, or the run stops being
//! reproducible. [`crate::genome::slots`] owns the scan; this only builds the
//! flag buffer it scans.
//!
//! Growth is two launches over a list of `(organism, limb)` requests.
//! [`layout_limbs`] needs the device's `ppos` — where the parent limb's last
//! particle actually ended up — so it runs on the device; [`place_body_particles`]
//! then writes the claimed slots from what it computed plus the per-particle
//! fields the host already knows. A request's parent limb must already be
//! grown, which is why [`crate::bodies::spawn`] lays a whole body out in
//! waves rather than one batch.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::constraints::{limb_axis, limb_len, particle_at, rotate, wrap_x};
use super::{
  BodyArgs, Cfg, KIND_BODY, KIND_FREE, P_BOUNDS_X, P_BOUNDS_Y, P_LIMB_SEGMENT_LENGTH, PARAM_COUNT,
  body_args, cube_count, whole,
};
use crate::bodies::BodyCfg;
use crate::genome::population::{LimbArgs, limb_args};
use crate::particles::{PARKED_POS, ParticleKind, SphDevice, SphHost};

/// [`crate::particles::PARKED_POS`] as two kernel-side literals.
const PARKED_X: f32 = PARKED_POS.x;
const PARKED_Y: f32 = PARKED_POS.y;
/// [`crate::particles::NO_ORGANISM`].
const NO_ORGANISM: u32 = crate::particles::NO_ORGANISM;

/// 1 where a particle slot is taken, 0 where it is claimable. A `Free` slot
/// is the only claimable one.
#[cube(launch)]
pub fn mark_body_occupancy(state: &[u32], flags: &mut [u32], #[comptime] cfg: Cfg) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  let mut occupied = 1u32;
  if state[i] == KIND_FREE {
    occupied = 0u32;
  }
  flags[i] = occupied;
}

/// Plain-Rust twin of [`mark_body_occupancy`].
pub fn mark_body_occupancy_ref(state: &[ParticleKind]) -> Vec<u32> {
  state
    .iter()
    .map(|k| u32::from(*k != ParticleKind::Free))
    .collect()
}

/// Where one limb's particles go: one unit per growth request.
///
/// The chain starts on the organism's anchor for the root limb, where the
/// pin holds it, and one rest length off its parent's last particle for a
/// child — along the parent's axis rotated by the record's `grow_angle`,
/// which is the rest shape the base-joint constraint asks for.
///
/// The map is only read — for the parent's last particle. The host is its
/// master and uploads it on both sides of this launch, so the claimed ids
/// themselves never have to reach this kernel.
///
/// `ppos` is taken mutably although it is only read: a CubeCL expand type has
/// no reborrow, so `limb_axis` cannot be handed a `&[f32]`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn layout_limbs(
  ppos: &mut [f32],
  limbs: &LimbArgs,
  map: &[u32],
  anchors: &[f32],
  req_organism: &[u32],
  req_limb: &[u32],
  req_first: &[u32],
  req_count: &[u32],
  positions: &mut [f32],
  params: &[f32],
  requests: u32,
  #[comptime] cfg: BodyCfg,
) {
  let r = ABSOLUTE_POS as u32;
  if r >= requests {
    terminate!();
  }
  let o = req_organism[r as usize];
  let l = req_limb[r as usize];
  let first = req_first[r as usize];
  let count = req_count[r as usize];

  let ml = comptime!(cfg.max_limbs);
  let mp = comptime!(cfg.max_particles_per_limb);
  let base = o * ml;
  let pbase = o * ml * mp;
  let bounds_x = params[P_BOUNDS_X as usize];
  let bounds_y = params[P_BOUNDS_Y as usize];
  let rest = params[P_LIMB_SEGMENT_LENGTH as usize];
  let angle = limbs.grow_angle[(base + l) as usize];
  let parent = limbs.parent[(base + l) as usize];

  let mut base_x = anchors[2 * o as usize];
  let mut base_y = anchors[2 * o as usize + 1];
  let mut dir_x = f32::cos(angle);
  let mut dir_y = f32::sin(angle);
  // The root's first particle sits *on* the anchor; a child's starts one rest
  // length off its parent's last particle.
  let mut first_step = 0.0f32;

  if parent != l {
    first_step = 1.0f32;
    let n = limb_len(limbs, base + parent, cfg);
    let last = particle_at(map, pbase, parent, n - 1u32, cfg);
    base_x = ppos[2 * last as usize];
    base_y = ppos[2 * last as usize + 1];
    let mut axis = limb_axis(ppos, limbs, map, base, pbase, parent, bounds_x, cfg);
    // A parent with no direction of its own — coincident particles — grows
    // its child along +x rather than nowhere.
    if axis.x == 0.0f32 && axis.y == 0.0f32 {
      axis.x = 1.0f32;
      axis.y = 0.0f32;
    }
    let d = rotate(axis, angle);
    dir_x = d.x;
    dir_y = d.y;
  }

  for i in 0..count {
    let k = (first + i) as usize;
    let t = rest * (i as f32 + first_step);
    let x = wrap_x(base_x + dir_x * t, bounds_x);
    let mut y = base_y + dir_y * t;
    if y < 0.0f32 {
      y = 0.0f32;
    } else if y > bounds_y {
      y = bounds_y;
    }
    positions[2 * k] = x;
    positions[2 * k + 1] = y;
  }
}

/// Turn claimed slots into body particles. One unit per entry of the list.
///
/// Everything the fluid kernels read is set here, not only the position: a
/// slot that has been free since startup carries whatever `client.empty` gave
/// it in `density`, and `calculate_accel` divides by that. A slot freed by a
/// death carries the dead organism's values, which is the same problem.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn place_body_particles(
  sph: &mut BodyArgs,
  ids: &[u32],
  positions: &[f32],
  organism: &[u32],
  limb: &[u32],
  index_in_limb: &[u32],
  part_type: &[u32],
  count: u32,
) {
  let k = ABSOLUTE_POS;
  if k >= count as usize {
    terminate!();
  }
  let id = ids[k] as usize;
  sph.pos[2 * id] = positions[2 * k];
  sph.pos[2 * id + 1] = positions[2 * k + 1];
  sph.ppos[2 * id] = positions[2 * k];
  sph.ppos[2 * id + 1] = positions[2 * k + 1];
  sph.vel[2 * id] = 0.0f32;
  sph.vel[2 * id + 1] = 0.0f32;
  sph.mass[id] = 1.0f32;
  sph.density[id] = 0.0f32;
  sph.near_density[id] = 0.0f32;
  sph.evap_prob[id] = 0.0f32;
  sph.state[id] = KIND_BODY;
  sph.organism[id] = organism[k];
  sph.limb[id] = limb[k];
  sph.index_in_limb[id] = index_in_limb[k];
  sph.part_type[id] = part_type[k];
}

/// Give a list of particle slots back: park them outside the world and mark
/// them claimable again. One unit per entry.
///
/// Each freed slot's `ppos` is copied out first, because a dying body's
/// particles become organic matter in the soil cell they were standing in and
/// the host does the depositing — a float add per cell is order-dependent on
/// the device and the host sum is in slot order, so it is the same every run.
///
/// The slot is left exactly as `SphHost::push_free_slots` leaves a slot that
/// has never been used, so a re-used slot and a fresh one are the same thing.
#[cube(launch)]
pub fn free_particles(sph: &mut BodyArgs, ids: &[u32], freed: &mut [f32], count: u32) {
  let k = ABSOLUTE_POS;
  if k >= count as usize {
    terminate!();
  }
  let id = ids[k] as usize;
  freed[2 * k] = sph.ppos[2 * id];
  freed[2 * k + 1] = sph.ppos[2 * id + 1];

  sph.pos[2 * id] = PARKED_X;
  sph.pos[2 * id + 1] = PARKED_Y;
  sph.ppos[2 * id] = PARKED_X;
  sph.ppos[2 * id + 1] = PARKED_Y;
  sph.vel[2 * id] = 0.0f32;
  sph.vel[2 * id + 1] = 0.0f32;
  sph.mass[id] = 0.0f32;
  sph.density[id] = 0.0f32;
  sph.near_density[id] = 0.0f32;
  sph.evap_prob[id] = 0.0f32;
  sph.state[id] = KIND_FREE;
  sph.organism[id] = NO_ORGANISM;
  sph.limb[id] = 0u32;
  sph.index_in_limb[id] = 0u32;
  sph.part_type[id] = 0u32;
}

/// Free a list of slots and return where each one was standing.
pub fn launch_free<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  ids: &[u32],
) -> Vec<glam::Vec2> {
  if ids.is_empty() {
    return Vec::new();
  }
  let n = ids.len();
  let ids_dev = client.create_from_slice(bytemuck::cast_slice(ids));
  let freed = client.empty(n * 2 * size_of::<f32>());
  free_particles::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    body_args(sph),
    whole(&ids_dev, n),
    whole(&freed, n * 2),
    n as u32,
  );
  crate::soa::download_field(client, &freed, n)
}

/// Plain-Rust twin of [`free_particles`].
pub fn free_particles_ref(particles: &mut SphHost, ids: &[u32]) -> Vec<glam::Vec2> {
  let mut freed = Vec::with_capacity(ids.len());
  for id in ids {
    let id = *id as usize;
    freed.push(particles.ppos[id]);
    particles.pos[id] = crate::particles::PARKED_POS;
    particles.ppos[id] = crate::particles::PARKED_POS;
    particles.vel[id] = glam::Vec2::ZERO;
    particles.mass[id] = 0.0;
    particles.density[id] = 0.0;
    particles.near_density[id] = 0.0;
    particles.evap_prob[id] = 0.0;
    particles.state[id] = ParticleKind::Free;
    particles.organism[id] = crate::particles::NO_ORGANISM;
    particles.limb[id] = 0;
    particles.index_in_limb[id] = 0;
    particles.part_type[id] = crate::genome::PartType::Absent;
  }
  freed
}

/// Fill a per-particle occupancy flag buffer from the particle kinds.
/// Over every slot, not the live prefix: a claim is looking for the free
/// slots above the high-water mark.
pub fn launch_mark_occupancy<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  flags: &Handle,
  cfg: Cfg,
) {
  let n = cfg.num_particles as usize;
  mark_body_occupancy::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(&sph.state, n),
    whole(flags, n),
    cfg,
  );
}

/// One batch of particles to write. `positions` is empty when
/// [`launch_layout`] is filling it on the device.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Placement {
  pub ids: Vec<u32>,
  /// Interleaved x, y, one pair per id.
  pub positions: Vec<f32>,
  pub organism: Vec<u32>,
  pub limb: Vec<u32>,
  pub index_in_limb: Vec<u32>,
  pub part_type: Vec<u32>,
}

impl Placement {
  pub fn len(&self) -> usize {
    self.ids.len()
  }

  pub fn is_empty(&self) -> bool {
    self.ids.is_empty()
  }
}

/// One limb to grow, as [`layout_limbs`] takes it: `first` indexes the
/// claimed-id list, which is also the [`Placement`] entry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowBatchEntry {
  pub organism: u32,
  pub limb: u32,
  pub first: u32,
  pub count: u32,
}

/// Run [`layout_limbs`] over a batch, into a freshly allocated positions
/// buffer. Every request's parent limb must already be grown.
#[allow(clippy::too_many_arguments)]
pub fn launch_layout<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  limbs: &crate::genome::LimbDevice,
  map: &Handle,
  anchors: &Handle,
  batch: &[GrowBatchEntry],
  particles: usize,
  params: &Handle,
  cfg: BodyCfg,
  world: Cfg,
) -> Handle {
  let positions = client.empty(particles * 2 * size_of::<f32>());
  let upload = |v: Vec<u32>| client.create_from_slice(bytemuck::cast_slice(&v));
  let organism = upload(batch.iter().map(|e| e.organism).collect());
  let limb = upload(batch.iter().map(|e| e.limb).collect());
  let first = upload(batch.iter().map(|e| e.first).collect());
  let count = upload(batch.iter().map(|e| e.count).collect());

  layout_limbs::launch::<R>(
    client,
    cube_count(batch.len()),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(&sph.ppos, world.num_particles as usize * 2),
    limb_args(limbs),
    whole(map, cfg.particle_map_len()),
    whole(anchors, cfg.max_organisms as usize * 2),
    whole(&organism, batch.len()),
    whole(&limb, batch.len()),
    whole(&first, batch.len()),
    whole(&count, batch.len()),
    whole(&positions, particles * 2),
    whole(params, PARAM_COUNT),
    batch.len() as u32,
    cfg,
  );
  positions
}

/// Write a batch of body particles. `positions` overrides
/// `placement.positions` when the device computed them.
pub fn launch_place<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  placement: &Placement,
  positions: Option<&Handle>,
) {
  let n = placement.len();
  if n == 0 {
    return;
  }
  let upload = |v: &[u32]| client.create_from_slice(bytemuck::cast_slice(v));
  let ids = upload(&placement.ids);
  let organism = upload(&placement.organism);
  let limb = upload(&placement.limb);
  let index_in_limb = upload(&placement.index_in_limb);
  let part_type = upload(&placement.part_type);
  let owned;
  let positions = match positions {
    Some(handle) => handle,
    None => {
      owned = client.create_from_slice(bytemuck::cast_slice(&placement.positions));
      &owned
    }
  };

  place_body_particles::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    body_args(sph),
    whole(&ids, n),
    whole(positions, n * 2),
    whole(&organism, n),
    whole(&limb, n),
    whole(&index_in_limb, n),
    whole(&part_type, n),
    n as u32,
  );
}

// --- Plain-Rust references ---

/// Plain-Rust twin of [`layout_limbs`]: the positions it writes.
pub fn layout_limbs_ref(
  particles: &SphHost,
  pop: &crate::genome::Population,
  bodies: &crate::bodies::BodyState,
  batch: &[GrowBatchEntry],
  params: &crate::SimParams,
  geom: &crate::world::WorldGeometry,
) -> Vec<f32> {
  use super::constraints::{limb_axis_ref, rotate_ref, wrap_x_ref};
  use glam::Vec2;

  let cfg = bodies.cfg;
  let mp = cfg.max_particles_per_limb as usize;
  let bounds = geom.bounds;
  let rest = params.limb_segment_length;
  let particle_count = batch
    .iter()
    .map(|e| (e.first + e.count) as usize)
    .max()
    .unwrap_or(0);
  let mut positions = vec![0.0f32; particle_count * 2];

  for entry in batch {
    let (o, l) = (entry.organism as usize, entry.limb as usize);
    let record = pop.limb_index(o, l);
    let angle = pop.limbs.grow_angle[record];
    let parent = pop.limbs.parent[record] as usize;

    let (base, direction, first_step) = if parent == l {
      (bodies.anchors[o], Vec2::from_angle(angle), 0.0)
    } else {
      let pcount = (pop.limbs.length[pop.limb_index(o, parent)] as usize).min(mp);
      let last = bodies.particle(o, parent, pcount - 1) as usize;
      let axis = limb_axis_ref(&particles.ppos, pop, bodies, o, parent, bounds.x);
      let axis = if axis == Vec2::ZERO { Vec2::X } else { axis };
      (particles.ppos[last], rotate_ref(axis, angle), 1.0)
    };

    for i in 0..entry.count as usize {
      let k = entry.first as usize + i;
      let mut p = base + direction * (rest * (i as f32 + first_step));
      p.x = wrap_x_ref(p.x, bounds.x);
      p.y = p.y.clamp(0.0, bounds.y);
      positions[2 * k] = p.x;
      positions[2 * k + 1] = p.y;
    }
  }
  positions
}

/// Plain-Rust twin of [`place_body_particles`].
pub fn place_body_particles_ref(particles: &mut SphHost, placement: &Placement, positions: &[f32]) {
  use glam::Vec2;
  for k in 0..placement.len() {
    let id = placement.ids[k] as usize;
    let p = Vec2::new(positions[2 * k], positions[2 * k + 1]);
    particles.pos[id] = p;
    particles.ppos[id] = p;
    particles.vel[id] = Vec2::ZERO;
    particles.mass[id] = 1.0;
    particles.density[id] = 0.0;
    particles.near_density[id] = 0.0;
    particles.evap_prob[id] = 0.0;
    particles.state[id] = ParticleKind::Body;
    particles.organism[id] = placement.organism[k];
    particles.limb[id] = placement.limb[k] as u8;
    particles.index_in_limb[id] = placement.index_in_limb[k] as u8;
    particles.part_type[id] = crate::genome::PartType::from_byte(placement.part_type[k] as u8);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bodies::NO_PARTICLE;
  use crate::kernels::test_support::OrganismHarness;
  use crate::soa::download_field;

  /// Freeing a slot has to leave it exactly as a never-used one, or the
  /// occupancy scan and every `Free` guard would see two kinds of empty.
  #[test]
  fn freeing_matches_reference_and_a_fresh_slot() {
    let h = OrganismHarness::new();
    let ids: Vec<u32> = (0..3).map(|i| h.bodies.particle(0, 0, 0) + i).collect();
    let where_they_were = launch_free(&h.client, &h.sph, &ids);

    let mut expected = h.particles.clone();
    let expected_positions = free_particles_ref(&mut expected, &ids);
    assert_eq!(where_they_were, expected_positions);
    assert_eq!(h.read_particles(), expected);

    // A slot the world reserved and never used, compared field by field
    // against one this just freed.
    let fresh = h.geom.num_particles - 1;
    let freed = ids[0] as usize;
    assert_eq!(expected.state[freed], expected.state[fresh]);
    assert_eq!(expected.ppos[freed], expected.ppos[fresh]);
    assert_eq!(expected.mass[freed], expected.mass[fresh]);
    assert_eq!(expected.organism[freed], expected.organism[fresh]);
    assert_eq!(expected.part_type[freed], expected.part_type[fresh]);
  }

  #[test]
  fn occupancy_matches_reference() {
    let h = OrganismHarness::new();
    let n = h.geom.num_particles;
    let flags = h.client.empty(n * size_of::<u32>());
    launch_mark_occupancy(&h.client, &h.sph, &flags, h.cfg);
    let actual: Vec<u32> = download_field(&h.client, &flags, n);
    assert_eq!(actual, mark_body_occupancy_ref(&h.particles.state));
    assert!(actual.contains(&0) && actual.contains(&1));
  }

  /// Grow two limbs off the fixture's already-laid-out root in one batch —
  /// the precondition a batch has, plus a non-zero `first` offset for the
  /// second — and compare the positions and the particle fields against the
  /// references.
  #[test]
  fn growth_matches_reference() {
    use crate::genome::PartType;

    let mut h = OrganismHarness::new();
    let cfg = h.body_cfg;
    // Record 3 is absent in the seed plant; make it a second stem off the
    // root, so both limbs in the batch have a parent that is already grown.
    let record = h.pop.limb_index(0, 3);
    h.pop.limbs.part_type[record] = PartType::Stem;
    h.pop.limbs.length[record] = 2;
    h.pop.limbs.parent[record] = 0;
    h.pop.limbs.child_slot[record] = 1;
    h.pop.limbs.grow_angle[record] = 0.5;
    h.pop.upload(&h.client);
    // Ungrow limb 1, which record 3 is now the sibling of.
    for i in cfg.limb_slice(0, 1) {
      h.bodies.limb_particles[i] = NO_PARTICLE;
    }
    h.bodies.upload(&h.client);

    let free_start = (h.geom.num_particles - 8) as u32;
    let ids: Vec<u32> = (free_start..free_start + 5).collect();
    let batch = vec![
      GrowBatchEntry {
        organism: 0,
        limb: 1,
        first: 0,
        count: 3,
      },
      GrowBatchEntry {
        organism: 0,
        limb: 3,
        first: 3,
        count: 2,
      },
    ];
    let positions = launch_layout(
      &h.client,
      &h.sph,
      &h.pop.device.limbs,
      &h.bodies.device.limb_particles,
      &h.bodies.device.anchors,
      &batch,
      ids.len(),
      &h.params_buf,
      cfg,
      h.cfg,
    );

    let expected_pos =
      layout_limbs_ref(&h.particles, &h.pop, &h.bodies, &batch, &h.params, &h.geom);
    let actual_pos: Vec<f32> = download_field(&h.client, &positions, ids.len() * 2);
    for (a, e) in actual_pos.iter().zip(&expected_pos) {
      assert!((a - e).abs() < 1e-6, "{a} against {e}");
    }
    // The first new particle landed one rest length off the root's last.
    let rest = h.params.limb_segment_length;
    let root_last = h.bodies.particle(0, 0, 1) as usize;
    let first = glam::Vec2::new(expected_pos[0], expected_pos[1]);
    assert!(((first - h.particles.ppos[root_last]).length() - rest).abs() < 1e-5);

    let placement = Placement {
      ids: ids.clone(),
      positions: expected_pos.clone(),
      organism: vec![0; 5],
      limb: vec![1, 1, 1, 3, 3],
      index_in_limb: vec![0, 1, 2, 0, 1],
      part_type: vec![PartType::Stem as u32; 5],
    };
    launch_place(&h.client, &h.sph, &placement, Some(&positions));
    let actual = h.read_particles();
    let mut expected = h.particles.clone();
    place_body_particles_ref(&mut expected, &placement, &expected_pos);
    assert_eq!(actual, expected);
  }
}
