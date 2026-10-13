//! The kernels a body spawn needs: which particle slots are claimable, where
//! a limb's particles go, and turning claimed slots into particles — body
//! particles for growth and seeds, and liquid or vapor for the water an
//! organism gives back (`docs/organism.md`, *Water*).
//!
//! Slot allocation is a prefix scan over an occupancy flag, never an atomic
//! claim (`docs/organism.md`, *Implementation layout*): the ids have to come
//! out ascending and in the same order every run, or the run stops being
//! reproducible. [`crate::genome::slots`] owns the scan; this only builds the
//! flag buffer it scans.
//!
//! Growth is two launches over a list of `(organism, limb)` requests.
//! [`layout_limbs`] needs the device's `ppos` — where the parent limb's last
//! particle actually ended up — so it runs on the device; [`place_particles`]
//! then writes the claimed slots from what it computed plus the per-particle
//! fields the host already knows. A request's parent limb must already be
//! grown, which is why [`crate::bodies::spawn`] lays a whole body out in
//! waves rather than one batch.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::constraints::{limb_axis, limb_len, particle_at, rotate, segment_fraction, wrap_x};
use super::motion::VAPOR_RISE;
use super::{
  BodyArgs, Cfg, KIND_BODY, KIND_FREE, KIND_VAPOR, P_BOUNDS_X, P_BOUNDS_Y, P_LIMB_SEGMENT_LENGTH,
  PARAM_COUNT, body_args, cube_count, whole,
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

/// Where one limb's next particles go: one unit per growth request.
///
/// A request grows a *suffix* of a limb's chain, from `first_index` on, so
/// this is where growth lands a particle as `ceil(grown)` rises as well as
/// where a whole founder limb is laid out. Each new particle sits one
/// segment along from its predecessor, the segment being
/// `limb_segment_length` times how far the limb has grown into it
/// ([`segment_fraction`]) — which is the rest length the constraint pass
/// will hold it at, so a claimed particle starts where it belongs.
///
/// What "along" means: for a limb continuing its own chain, its last
/// segment's direction; for its first particle, its parent's axis rotated by
/// the record's `grow_angle`, which is the rest shape the base-joint
/// constraint asks for. The root's first particle sits on the anchor, where
/// the pin holds it.
///
/// The map is only read — for the predecessor. The host is its master and
/// uploads it on both sides of this launch, so the claimed ids themselves
/// never have to reach this kernel.
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
  grown: &[f32],
  req_organism: &[u32],
  req_limb: &[u32],
  req_first_index: &[u32],
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
  let first_index = req_first_index[r as usize];
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
  let g = grown[(base + l) as usize];

  let mut base_x = anchors[2 * o as usize];
  let mut base_y = anchors[2 * o as usize + 1];
  let mut dir_x = f32::cos(angle);
  let mut dir_y = f32::sin(angle);
  // The root limb's particle 0 is the only one with no predecessor: it sits
  // on the anchor rather than a segment away from anything.
  let root_first = parent == l;

  if first_index > 0u32 {
    // Continuing this limb's own chain, along the direction it already
    // points in.
    let prev = particle_at(map, pbase, l, first_index - 1u32, cfg);
    base_x = ppos[2 * prev as usize];
    base_y = ppos[2 * prev as usize + 1];
    let mut axis = limb_axis(ppos, limbs, map, grown, base, pbase, l, bounds_x, cfg);
    if axis.x == 0.0f32 && axis.y == 0.0f32 {
      axis.x = 1.0f32;
      axis.y = 0.0f32;
    }
    dir_x = axis.x;
    dir_y = axis.y;
  } else if parent != l {
    let n = limb_len(map, pbase, parent, cfg);
    let last = particle_at(map, pbase, parent, n - 1u32, cfg);
    base_x = ppos[2 * last as usize];
    base_y = ppos[2 * last as usize + 1];
    let mut axis = limb_axis(ppos, limbs, map, grown, base, pbase, parent, bounds_x, cfg);
    // A parent with no direction of its own — coincident particles, or no
    // whole segment yet — grows its child along +x rather than nowhere.
    if axis.x == 0.0f32 && axis.y == 0.0f32 {
      axis.x = 1.0f32;
      axis.y = 0.0f32;
    }
    let d = rotate(axis, angle);
    dir_x = d.x;
    dir_y = d.y;
  }

  // Unwrapped as it walks the chain, wrapped on the way out: a chain that
  // crosses the seam is still one straight line.
  let mut x = base_x;
  let mut y = base_y;
  for i in 0..count {
    let k = first_index + i;
    let mut t = rest * segment_fraction(g, k);
    if root_first && k == 0u32 {
      t = 0.0f32;
    }
    x += dir_x * t;
    y += dir_y * t;
    let at = (first + i) as usize;
    positions[2 * at] = wrap_x(x, bounds_x);
    let mut out_y = y;
    if out_y < 0.0f32 {
      out_y = 0.0f32;
    } else if out_y > bounds_y {
      out_y = bounds_y;
    }
    positions[2 * at + 1] = out_y;
  }
}

/// Turn claimed slots into particles of the kind each entry asks for. One
/// unit per entry of the list.
///
/// Everything the fluid kernels read is set here, not only the position: a
/// slot that has been free since startup carries whatever `client.empty` gave
/// it in `density`, and `calculate_accel` divides by that. A slot freed by a
/// death carries the dead organism's values, which is the same problem.
///
/// Every kind starts at rest except vapor, which starts rising at the
/// [`VAPOR_RISE`] `evaporate_particles` gives a particle it turns to vapor:
/// transpired water leaves a leaf exactly as evaporated water leaves the pool.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn place_particles(
  sph: &mut BodyArgs,
  ids: &[u32],
  positions: &[f32],
  kind: &[u32],
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
  if kind[k] == KIND_VAPOR {
    sph.vel[2 * id + 1] = VAPOR_RISE;
  }
  sph.mass[id] = 1.0f32;
  sph.density[id] = 0.0f32;
  sph.near_density[id] = 0.0f32;
  sph.evap_prob[id] = 0.0f32;
  sph.state[id] = kind[k];
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
  free_slot(sph, id);
}

/// Leave slot `id` exactly as a slot that has never been used: parked, no
/// mass, no state, no owner. The one definition of what `Free` means on the
/// device — a death, a break, a discarded seed and a drink all go through it,
/// so a field added to the SoA is cleared in one place or none.
#[cube]
pub fn free_slot(sph: &mut BodyArgs, id: usize) {
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
    free_slot_ref(particles, id);
  }
  freed
}

/// Host twin of [`free_slot`].
pub fn free_slot_ref(particles: &mut SphHost, id: usize) {
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
  /// The [`ParticleKind`] each slot becomes, as its `u32` code.
  pub kind: Vec<u32>,
  /// [`crate::particles::NO_ORGANISM`] for a fluid particle.
  pub organism: Vec<u32>,
  pub limb: Vec<u32>,
  pub index_in_limb: Vec<u32>,
  pub part_type: Vec<u32>,
}

impl Placement {
  pub fn len(&self) -> usize {
    self.ids.len()
  }

  /// Append one body particle of `organism`, with no id or position yet:
  /// growth claims its slots and lays them out once the whole batch is known.
  pub fn push_body(&mut self, organism: u32, limb: u32, index_in_limb: u32, part_type: u32) {
    self.kind.push(KIND_BODY);
    self.organism.push(organism);
    self.limb.push(limb);
    self.index_in_limb.push(index_in_limb);
    self.part_type.push(part_type);
  }

  /// Append one fluid particle — liquid or vapor, owned by no organism — into
  /// slot `id` at `at`.
  pub fn push_fluid(&mut self, id: u32, kind: ParticleKind, at: glam::Vec2) {
    self.ids.push(id);
    self.positions.push(at.x);
    self.positions.push(at.y);
    self.kind.push(kind as u32);
    self.organism.push(NO_ORGANISM);
    self.limb.push(0);
    self.index_in_limb.push(0);
    self.part_type.push(0);
  }

  /// Keep the first `n` entries. `positions` is cut only where the host
  /// filled it.
  pub fn truncate(&mut self, n: usize) {
    self.ids.truncate(n);
    self.positions.truncate(2 * n);
    self.kind.truncate(n);
    self.organism.truncate(n);
    self.limb.truncate(n);
    self.index_in_limb.truncate(n);
    self.part_type.truncate(n);
  }

  pub fn is_empty(&self) -> bool {
    self.ids.is_empty()
  }
}

/// One limb's next particles, as [`layout_limbs`] takes them:
/// `first_index` is where in the limb's own chain they start, and `first`
/// indexes the claimed-id list, which is also the [`Placement`] entry order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowBatchEntry {
  pub organism: u32,
  pub limb: u32,
  pub first_index: u32,
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
  grown: &Handle,
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
  let first_index = upload(batch.iter().map(|e| e.first_index).collect());
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
    whole(grown, cfg.limb_count()),
    whole(&organism, batch.len()),
    whole(&limb, batch.len()),
    whole(&first_index, batch.len()),
    whole(&first, batch.len()),
    whole(&count, batch.len()),
    whole(&positions, particles * 2),
    whole(params, PARAM_COUNT),
    batch.len() as u32,
    cfg,
  );
  positions
}

/// Write a batch of particles. `positions` overrides `placement.positions`
/// when the device computed them.
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
  let kind = upload(&placement.kind);
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

  place_particles::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    body_args(sph),
    whole(&ids, n),
    whole(positions, n * 2),
    whole(&kind, n),
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
  use super::constraints::{limb_axis_ref, rotate_ref, segment_fraction_ref, wrap_x_ref};
  use glam::Vec2;

  let cfg = bodies.cfg;
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
    let first_index = entry.first_index as usize;
    let record = pop.limb_index(o, l);
    let angle = pop.limbs.grow_angle[record];
    let parent = pop.limbs.parent[record] as usize;
    let grown = bodies.grown[cfg.limb_index(o, l)];

    let root_first = parent == l;
    let (mut at, direction) = if first_index > 0 {
      let prev = bodies.particle(o, l, first_index - 1) as usize;
      let axis = limb_axis_ref(&particles.ppos, pop, bodies, o, l, bounds.x);
      let axis = if axis == Vec2::ZERO { Vec2::X } else { axis };
      (particles.ppos[prev], axis)
    } else if root_first {
      (bodies.anchors[o], Vec2::from_angle(angle))
    } else {
      let pcount = bodies.limb_len(o, parent);
      let last = bodies.particle(o, parent, pcount - 1) as usize;
      let axis = limb_axis_ref(&particles.ppos, pop, bodies, o, parent, bounds.x);
      let axis = if axis == Vec2::ZERO { Vec2::X } else { axis };
      (particles.ppos[last], rotate_ref(axis, angle))
    };

    for i in 0..entry.count as usize {
      let k = first_index + i;
      let step = if root_first && k == 0 {
        0.0
      } else {
        rest * segment_fraction_ref(grown, k)
      };
      at += direction * step;
      let out = entry.first as usize + i;
      positions[2 * out] = wrap_x_ref(at.x, bounds.x);
      positions[2 * out + 1] = at.y.clamp(0.0, bounds.y);
    }
  }
  positions
}

/// Plain-Rust twin of [`place_particles`].
pub fn place_particles_ref(particles: &mut SphHost, placement: &Placement, positions: &[f32]) {
  use glam::Vec2;
  for k in 0..placement.len() {
    let id = placement.ids[k] as usize;
    let p = Vec2::new(positions[2 * k], positions[2 * k + 1]);
    let kind = ParticleKind::from_byte(placement.kind[k] as u8);
    particles.pos[id] = p;
    particles.ppos[id] = p;
    particles.vel[id] = if kind == ParticleKind::Vapor {
      Vec2::new(0.0, VAPOR_RISE)
    } else {
      Vec2::ZERO
    };
    particles.mass[id] = 1.0;
    particles.density[id] = 0.0;
    particles.near_density[id] = 0.0;
    particles.evap_prob[id] = 0.0;
    particles.state[id] = kind;
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
    // Ungrow limb 1, which record 3 is now the sibling of; both grow from
    // nothing to their full length in this batch.
    for i in cfg.limb_slice(0, 1) {
      h.bodies.limb_particles[i] = NO_PARTICLE;
    }
    h.bodies.grown[cfg.limb_index(0, 1)] = 3.0;
    h.bodies.grown[cfg.limb_index(0, 3)] = 2.0;
    h.bodies.upload(&h.client);

    let free_start = (h.geom.num_particles - 8) as u32;
    let ids: Vec<u32> = (free_start..free_start + 5).collect();
    let batch = vec![
      GrowBatchEntry {
        organism: 0,
        limb: 1,
        first_index: 0,
        first: 0,
        count: 3,
      },
      GrowBatchEntry {
        organism: 0,
        limb: 3,
        first_index: 0,
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
      &h.bodies.device.grown,
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
      kind: vec![KIND_BODY; 5],
      organism: vec![0; 5],
      limb: vec![1, 1, 1, 3, 3],
      index_in_limb: vec![0, 1, 2, 0, 1],
      part_type: vec![PartType::Stem as u32; 5],
    };
    launch_place(&h.client, &h.sph, &placement, Some(&positions));
    let actual = h.read_particles();
    let mut expected = h.particles.clone();
    place_particles_ref(&mut expected, &placement, &expected_pos);
    assert_eq!(actual, expected);
  }

  /// Water an organism gives back is placed through the same kernel as a
  /// body: a liquid particle at rest and a vapor particle rising, both owned
  /// by no organism, into slots a body just gave up.
  #[test]
  fn fluid_placement_matches_reference() {
    let h = OrganismHarness::new();
    let ids: Vec<u32> = (0..2).map(|i| h.bodies.particle(0, 0, 0) + i).collect();
    launch_free(&h.client, &h.sph, &ids);
    let mut expected = h.particles.clone();
    free_particles_ref(&mut expected, &ids);

    let mut placement = Placement::default();
    placement.push_fluid(ids[0], ParticleKind::Liquid, glam::Vec2::new(1.25, 0.5));
    placement.push_fluid(ids[1], ParticleKind::Vapor, glam::Vec2::new(0.75, 1.5));
    launch_place(&h.client, &h.sph, &placement, None);
    place_particles_ref(&mut expected, &placement, &placement.positions);
    let actual = h.read_particles();
    assert_eq!(actual, expected);
    assert_eq!(actual.state[ids[0] as usize], ParticleKind::Liquid);
    assert_eq!(actual.vel[ids[1] as usize], glam::Vec2::new(0.0, 0.5));
    assert_eq!(actual.organism[ids[1] as usize], NO_ORGANISM);
  }

  /// The ordinary case once a body is standing: one more particle on the end
  /// of a limb that already has some, a partial segment along the direction
  /// the limb points in.
  #[test]
  fn a_limb_grows_its_next_particle_onto_its_own_chain() {
    let mut h = OrganismHarness::new();
    let cfg = h.body_cfg;
    // The stem holds two of its three particles and is a third of the way
    // into the third.
    let slice = cfg.limb_slice(0, 1);
    let dropped = h.bodies.limb_particles[slice.start + 2];
    h.bodies.limb_particles[slice.start + 2] = NO_PARTICLE;
    h.bodies.grown[cfg.limb_index(0, 1)] = 2.25;
    h.bodies.upload(&h.client);

    let batch = vec![GrowBatchEntry {
      organism: 0,
      limb: 1,
      first_index: 2,
      first: 0,
      count: 1,
    }];
    let positions = launch_layout(
      &h.client,
      &h.sph,
      &h.pop.device.limbs,
      &h.bodies.device.limb_particles,
      &h.bodies.device.anchors,
      &h.bodies.device.grown,
      &batch,
      1,
      &h.params_buf,
      cfg,
      h.cfg,
    );
    let expected = layout_limbs_ref(&h.particles, &h.pop, &h.bodies, &batch, &h.params, &h.geom);
    let actual: Vec<f32> = download_field(&h.client, &positions, 2);
    for (a, e) in actual.iter().zip(&expected) {
      assert!((a - e).abs() < 1e-6, "{a} against {e}");
    }

    // A quarter of a rest length off the particle before it, and nowhere
    // near where the whole third segment would have put it.
    let previous = h.particles.ppos[h.bodies.particle(0, 1, 1) as usize];
    let placed = glam::Vec2::new(expected[0], expected[1]);
    let step = (placed - previous).length();
    let rest = h.params.limb_segment_length;
    assert!((step - 0.25 * rest).abs() < 1e-5, "{step} against {rest}");
    assert_ne!(dropped, crate::bodies::NO_PARTICLE);
  }
}
