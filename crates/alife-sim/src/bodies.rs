//! Bodies: which particle holds which limb particle, and where each organism
//! is anchored.
//!
//! From `docs/organism.md`'s *Body* and *Implementation layout*. A limb is a
//! chain of particles in the fluid's particle system; the genome says how many
//! and of what type, and this is the map from `(organism, limb, index)` to the
//! particle slot that holds it. The constraint pass
//! ([`crate::kernels::constraints`]) walks that map, and the geometry kernel
//! ([`crate::kernels::limb_geometry`]) publishes what the brain will read.
//!
//! Capacity is `max_organisms x max_limbs x max_particles_per_limb`, the same
//! product the particle system reserves as body slots.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use crate::SimParams;
use crate::define_soa;
use crate::genome::slots::{SlotScan, claim_free_slots, read_free_slots};
use crate::genome::{Genome, Population};
use crate::kernels::spawn::Placement;
use crate::particles::SphDevice;
use crate::soil::SoilGrid;
use crate::world::{Cfg, WorldGeometry};

define_soa! {
    /// What the brain chunk reads about a limb's placement in the world, one
    /// entry per `(organism, limb)`. Written by
    /// [`crate::kernels::limb_geometry`] after every constraint pass.
    LimbGeometryHost / LimbGeometryDevice {
        /// The limb's first particle, relative to the root limb's first
        /// particle. The body frame is the world frame for plants.
        rel_pos: Vec2,
        /// Direction of the limb's first segment: from the parent's last
        /// particle to this limb's first, or `p0 -> p1` for the root.
        angle: f32,
        /// Hops from the root limb; the root itself is 0.
        depth: u32,
    }
}

/// An entry of [`BodyState::limb_particles`] that holds no particle: the limb
/// is absent, or has not grown that far.
pub const NO_PARTICLE: u32 = u32::MAX;

/// The shapes the organism kernels bake in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyCfg {
  pub max_organisms: u32,
  pub max_limbs: u32,
  pub max_particles_per_limb: u32,
}

impl BodyCfg {
  pub fn from_params(params: &SimParams) -> Self {
    Self {
      max_organisms: params.max_organisms.max(1) as u32,
      max_limbs: params.max_limbs.max(1) as u32,
      max_particles_per_limb: params.max_particles_per_limb.max(1) as u32,
    }
  }

  /// Entries in the `(organism, limb, index)` map.
  pub fn particle_map_len(&self) -> usize {
    (self.max_organisms * self.max_limbs * self.max_particles_per_limb) as usize
  }

  /// Entries in a per-limb buffer.
  pub fn limb_count(&self) -> usize {
    (self.max_organisms * self.max_limbs) as usize
  }

  /// Flat index of one limb's particle slice.
  pub fn limb_slice(&self, organism: usize, limb: usize) -> std::ops::Range<usize> {
    let mp = self.max_particles_per_limb as usize;
    let start = (organism * self.max_limbs as usize + limb) * mp;
    start..start + mp
  }
}

/// Device mirrors of everything the organism kernels read or write.
#[derive(Debug, Clone)]
pub struct BodyDevice {
  /// `[max_organisms x max_limbs x max_particles_per_limb]` particle ids.
  pub limb_particles: Handle,
  /// `[max_organisms]` anchor positions, interleaved x,y.
  pub anchors: Handle,
  /// Where the integrator left each particle, before the projection moved it.
  /// `[num_particles]` interleaved x,y; only body particles are written.
  pub ppos_prev: Handle,
  pub geometry: LimbGeometryDevice,
}

/// Host master copies plus their device mirrors, alongside
/// [`crate::genome::Population`].
pub struct BodyState {
  pub cfg: BodyCfg,
  pub limb_particles: Vec<u32>,
  /// Each organism's anchor: the soil cell its root germinated in.
  pub anchors: Vec<Vec2>,
  pub geometry: LimbGeometryHost,
  pub device: BodyDevice,
}

impl BodyState {
  pub fn new<R: Runtime>(
    client: &ComputeClient<R>,
    params: &SimParams,
    geom: &WorldGeometry,
  ) -> Self {
    let cfg = BodyCfg::from_params(params);
    let limb_particles = vec![NO_PARTICLE; cfg.particle_map_len()];
    let anchors = vec![Vec2::ZERO; cfg.max_organisms as usize];
    let geometry = LimbGeometryHost::new(cfg.limb_count());
    let device = BodyDevice {
      limb_particles: client.create_from_slice(bytemuck::cast_slice(&limb_particles)),
      anchors: client.create_from_slice(bytemuck::cast_slice(&flatten(&anchors))),
      ppos_prev: client.empty(geom.num_particles * 2 * size_of::<f32>()),
      geometry: LimbGeometryDevice::upload(client, &geometry),
    };
    Self {
      cfg,
      limb_particles,
      anchors,
      geometry,
      device,
    }
  }

  /// The particle holding `(organism, limb, index)`, or [`NO_PARTICLE`].
  pub fn particle(&self, organism: usize, limb: usize, index: usize) -> u32 {
    self.limb_particles[self.cfg.limb_slice(organism, limb).start + index]
  }

  /// Upload the host master copies of the map and the anchors.
  pub fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.device.limb_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.limb_particles));
    self.device.anchors = client.create_from_slice(bytemuck::cast_slice(&flatten(&self.anchors)));
  }
}

fn flatten(v: &[Vec2]) -> Vec<f32> {
  v.iter().flat_map(|p| [p.x, p.y]).collect()
}

// --- Spawning, on the host ---
//
// Births belong to the life-cycle chunk and will run on the device; until
// then a body is laid out here. The cost is honest about that: each call
// reads `ppos` back to find where the parent limb ended, and claims its slots
// with a scan and two small reads.

/// Everything a spawn touches, borrowed from one [`Sim`] at once.
pub struct SimBodies<'a, R: Runtime> {
  pub client: &'a ComputeClient<R>,
  pub params: &'a SimParams,
  pub geom: &'a WorldGeometry,
  pub cfg: Cfg,
  pub sph: &'a SphDevice,
  pub pop: &'a mut Population,
  pub bodies: &'a mut BodyState,
}

/// Why a limb could not be placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnError {
  /// The record is absent, or already holds particles.
  NothingToGrow,
  /// The parent limb has not been grown, so there is nothing to grow from.
  ParentNotGrown,
  /// Fewer free particle slots than the limb needs.
  OutOfParticles,
}

/// Install a genome in an organism slot and lay its whole body out from the
/// anchor, parents before children.
///
/// Returns the particles placed. The lineage fields (`parent_id`,
/// `birth_step`, `lineage_id`) are the life-cycle chunk's; this sets only
/// what a body needs: the genome, the anchor and `alive`.
pub fn spawn<R: Runtime>(
  sim: &mut crate::sim::Sim<R>,
  organism: usize,
  genome: &Genome,
  anchor: Vec2,
) -> usize {
  {
    let access = sim.body_access();
    access.pop.write_genome(organism, genome);
    access.pop.organisms.alive[organism] = 1;
    access.pop.upload(access.client);
    access.bodies.anchors[organism] = anchor;
    let cfg = access.bodies.cfg;
    let per_organism = (cfg.max_limbs * cfg.max_particles_per_limb) as usize;
    let start = organism * per_organism;
    access.bodies.limb_particles[start..start + per_organism].fill(NO_PARTICLE);
  }

  // Parents before children: a limb record's parent may sit at a higher
  // index than the limb itself, because a structural add takes the first
  // absent record rather than the next one.
  let max_limbs = sim.population().max_limbs;
  let mut placed = 0;
  let mut done = vec![false; max_limbs];
  #[allow(clippy::needless_range_loop)]
  for _ in 0..max_limbs {
    let mut progress = false;
    for limb in 0..max_limbs {
      if done[limb] {
        continue;
      }
      // `grow_limb` needs `sim` mutably, so this cannot hold a borrow of
      // `done` across the call; the index loop is the point.
      match grow_limb(sim, organism, limb) {
        Ok(n) => {
          placed += n;
          done[limb] = true;
          progress = true;
        }
        Err(SpawnError::ParentNotGrown) => {}
        Err(_) => done[limb] = true,
      }
    }
    if !progress {
      break;
    }
  }
  placed
}

/// Grow one limb: claim its particle slots and lay them out from its parent's
/// last particle along its rest direction, at rest spacing.
///
/// This is what the brain's sprout head will call once it exists.
pub fn grow_limb<R: Runtime>(
  sim: &mut crate::sim::Sim<R>,
  organism: usize,
  limb: usize,
) -> Result<usize, SpawnError> {
  let access = sim.body_access();
  let cfg = access.bodies.cfg;
  let record = access.pop.limb_index(organism, limb);
  let part_type = access.pop.limbs.part_type[record];
  let count = (access.pop.limbs.length[record] as usize).min(cfg.max_particles_per_limb as usize);
  if !part_type.is_present() || count == 0 {
    return Err(SpawnError::NothingToGrow);
  }
  let slice = cfg.limb_slice(organism, limb);
  if access.bodies.limb_particles[slice.clone()]
    .iter()
    .any(|id| *id != NO_PARTICLE)
  {
    return Err(SpawnError::NothingToGrow);
  }

  let ppos: Vec<Vec2> =
    crate::soa::download_field(access.client, &access.sph.ppos, access.geom.num_particles);
  let bounds = access.geom.bounds;
  let rest = access.params.limb_segment_length;
  let grow_angle = access.pop.limbs.grow_angle[record];
  let parent = access.pop.limbs.parent[record] as usize;

  // Where the chain starts and which way it goes. The root starts on the
  // anchor, where the pin holds it; a child starts one rest length off its
  // parent's last particle, along its parent's axis rotated by its grow
  // angle — the rest shape the base-joint constraint asks for.
  let (base, direction) = if parent == limb {
    (
      access.bodies.anchors[organism],
      Vec2::from_angle(grow_angle),
    )
  } else {
    let precord = access.pop.limb_index(organism, parent);
    let pcount =
      (access.pop.limbs.length[precord] as usize).min(cfg.max_particles_per_limb as usize);
    if !access.pop.limbs.part_type[precord].is_present() || pcount == 0 {
      return Err(SpawnError::ParentNotGrown);
    }
    let last = access.bodies.particle(organism, parent, pcount - 1);
    if last == NO_PARTICLE {
      return Err(SpawnError::ParentNotGrown);
    }
    let axis = crate::kernels::constraints::limb_axis_ref(
      &ppos,
      access.pop,
      access.bodies,
      organism,
      parent,
      bounds.x,
    );
    let axis = if axis == Vec2::ZERO { Vec2::X } else { axis };
    (ppos[last as usize], rotate_ref(axis, grow_angle))
  };
  let first_step = if parent == limb { 0.0 } else { 1.0 };

  let ids = claim_particles(&access, count)?;
  let mut placement = Placement {
    ids: ids.clone(),
    positions: Vec::with_capacity(count * 2),
    limb: vec![limb as u32; count],
    index_in_limb: (0..count as u32).collect(),
    part_type: vec![part_type as u32; count],
  };
  for i in 0..count {
    let mut p = base + direction * (rest * (i as f32 + first_step));
    p.x = crate::kernels::constraints::wrap_x_ref(p.x, bounds.x);
    p.y = p.y.clamp(0.0, bounds.y);
    placement.positions.push(p.x);
    placement.positions.push(p.y);
  }

  crate::kernels::spawn::launch_place(access.client, access.sph, organism as u32, &placement);
  for (i, id) in ids.iter().enumerate() {
    access.bodies.limb_particles[slice.start + i] = *id;
  }
  access.bodies.upload(access.client);
  Ok(count)
}

/// The first `n` claimable particle slots, ascending.
fn claim_particles<R: Runtime>(
  access: &SimBodies<'_, R>,
  n: usize,
) -> Result<Vec<u32>, SpawnError> {
  let total = access.geom.num_particles;
  let occupancy = access.client.empty(total * size_of::<u32>());
  crate::kernels::spawn::launch_mark_occupancy(access.client, access.sph, &occupancy, access.cfg);
  let scan = SlotScan::alloc(access.client, total);
  claim_free_slots(access.client, &occupancy, &scan);
  let free = read_free_slots(access.client, &scan);
  if free.len() < n {
    return Err(SpawnError::OutOfParticles);
  }
  Ok(free[..n].to_vec())
}

fn rotate_ref(d: Vec2, angle: f32) -> Vec2 {
  let (s, c) = (angle.sin(), angle.cos());
  Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c)
}

/// Seed `count` founders: one [`Genome::seed_plant`] each, at evenly spaced
/// x, anchored on the soil surface of its own column.
///
/// Returns how many were actually seeded, which is capped by the organism
/// slots. `--founders 0` does nothing at all, which is the default.
pub fn spawn_founders<R: Runtime>(sim: &mut crate::sim::Sim<R>, count: usize) -> usize {
  let count = count.min(sim.population().max_organisms);
  if count == 0 {
    return 0;
  }
  let seed = sim.seed();
  let shape = sim.population().shape;
  let max_limbs = sim.population().max_limbs;
  let bounds = sim.geometry().bounds;

  let anchors: Vec<Vec2> = (0..count)
    .map(|i| {
      let x = (i as f32 + 0.5) * bounds.x / count as f32;
      Vec2::new(x, soil_surface(sim.soil(), x))
    })
    .collect();

  for (slot, anchor) in anchors.into_iter().enumerate() {
    let genome = Genome::seed_plant(&shape, max_limbs, slot as u32, seed);
    spawn(sim, slot, &genome, anchor);
  }
  count
}

/// The y of the first soil cell from the top of `x`'s column that holds any
/// soil at all. A column with no soil anchors on the world floor.
pub fn soil_surface(soil: &SoilGrid, x: f32) -> f32 {
  let column = ((x / soil.cell_size) as usize).min(soil.width.saturating_sub(1));
  for row in (0..soil.height).rev() {
    let i = row * soil.width + column;
    let total =
      soil.cells.sand_density[i] + soil.cells.silt_density[i] + soil.cells.clay_density[i];
    if total > 0.0 {
      return (row as f32 + 0.5) * soil.cell_size;
    }
  }
  soil.cell_size * 0.5
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::genome::PartType;
  use crate::particles::ParticleKind;
  use crate::sim::Sim;
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  fn small_world() -> Sim<CpuRuntime> {
    let params = SimParams {
      world_width: 4.0,
      world_height: 3.0,
      smoothing_radius: 0.5,
      soil_cell_size: 0.25,
      particles_per_cell: 1,
      // The noise terrain, not the capillary test: mode 1 fills every column
      // to the ceiling, so a plant anchored on its surface has no headroom.
      terrain_mode: 0,
      max_organisms: 4,
      max_limbs: 4,
      max_particles_per_limb: 3,
      ..SimParams::default()
    };
    Sim::new(CpuRuntime::client(&CpuDevice), params, 42, None)
  }

  #[test]
  fn a_founder_lands_on_the_soil_surface_with_its_limbs_laid_out() {
    let mut sim = small_world();
    assert_eq!(spawn_founders(&mut sim, 2), 2);
    assert_eq!(sim.organism_count(), 2);

    let particles = sim.read_particles();
    let rest = sim.params().limb_segment_length;
    let cfg = sim.body_cfg();

    for o in 0..2usize {
      let anchor = sim.bodies().anchors[o];
      // Evenly spaced across the world, on the surface of that column.
      let expected_x = (o as f32 + 0.5) * sim.geometry().bounds.x / 2.0;
      assert!((anchor.x - expected_x).abs() < 1e-6);
      assert_eq!(anchor.y, soil_surface(sim.soil(), anchor.x));

      let root = sim.bodies().particle(o, 0, 0) as usize;
      assert_eq!(
        particles.ppos[root], anchor,
        "the root starts on the anchor"
      );

      for limb in 0..cfg.max_limbs as usize {
        let record = sim.population().limb_index(o, limb);
        let part_type = sim.population().limbs.part_type[record];
        let n =
          (sim.population().limbs.length[record] as usize).min(cfg.max_particles_per_limb as usize);
        if !part_type.is_present() {
          for i in 0..cfg.max_particles_per_limb as usize {
            assert_eq!(sim.bodies().particle(o, limb, i), NO_PARTICLE);
          }
          continue;
        }
        for i in 0..n {
          let id = sim.bodies().particle(o, limb, i) as usize;
          assert_eq!(particles.state[id], ParticleKind::Body);
          assert_eq!(particles.organism[id], o as u32);
          assert_eq!(particles.limb[id], limb as u8);
          assert_eq!(particles.index_in_limb[id], i as u8);
          assert_eq!(particles.part_type[id], part_type);
          assert_eq!(particles.mass[id], 1.0);
          if i > 0 {
            let prev = sim.bodies().particle(o, limb, i - 1) as usize;
            let d = (particles.ppos[id] - particles.ppos[prev]).length();
            assert!(
              (d - rest).abs() < 1e-5,
              "limb {limb} of organism {o}: spacing {d} against {rest}"
            );
          }
        }
      }
    }

    // The seed plant is root, stem, leaf, and the stem hangs off the root's
    // last particle at one rest length.
    let root_last = sim.bodies().particle(0, 0, 1) as usize;
    let stem_first = sim.bodies().particle(0, 1, 0) as usize;
    assert_eq!(particles.part_type[stem_first], PartType::Stem);
    let d = (particles.ppos[stem_first] - particles.ppos[root_last]).length();
    assert!((d - rest).abs() < 1e-5, "base joint spacing {d}");
  }

  #[test]
  fn slots_are_claimed_ascending_from_the_body_capacity() {
    let mut sim = small_world();
    spawn_founders(&mut sim, 2);
    let fluid = sim.geometry().fluid_particles as u32;
    let mut claimed: Vec<u32> = sim
      .bodies()
      .limb_particles
      .iter()
      .copied()
      .filter(|id| *id != NO_PARTICLE)
      .collect();
    claimed.sort_unstable();
    assert!(
      claimed.iter().all(|id| *id >= fluid),
      "a body particle took a fluid slot"
    );
    claimed.dedup();
    let particles = sim.read_particles();
    assert_eq!(
      claimed.len(),
      particles.count_of(ParticleKind::Body),
      "a slot was claimed twice"
    );
    // The first organism's particles come before the second's, and both are
    // contiguous from the start of the capacity: the scan hands them out in
    // order.
    assert_eq!(claimed[0], fluid);
    assert!(claimed.windows(2).all(|w| w[1] == w[0] + 1));
  }

  #[test]
  fn founders_cannot_outnumber_the_organism_slots() {
    let mut sim = small_world();
    assert_eq!(spawn_founders(&mut sim, 99), 4);
    assert_eq!(sim.organism_count(), 4);
  }

  #[test]
  fn nothing_happens_without_founders() {
    let mut sim = small_world();
    let before = sim.read_particles();
    assert_eq!(spawn_founders(&mut sim, 0), 0);
    assert_eq!(sim.organism_count(), 0);
    assert_eq!(sim.read_particles(), before);
    assert!(
      sim
        .bodies()
        .limb_particles
        .iter()
        .all(|id| *id == NO_PARTICLE)
    );
  }
}
