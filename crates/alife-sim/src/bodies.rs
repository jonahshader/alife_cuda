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
use crate::genome::population::NO_PARENT;
use crate::genome::slots::{SlotScan, claim_free_slots, read_free_slots};
use crate::genome::{Genome, Population};
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
  /// `[max_organisms]` particle ids: the one particle a seed organism is,
  /// or [`NO_PARTICLE`] once it has germinated (or never was a seed).
  pub seed_particles: Handle,
  /// Where the integrator left each particle, before the projection moved it.
  /// `[num_particles]` interleaved x,y; only body particles are written.
  pub ppos_prev: Handle,
  pub geometry: LimbGeometryDevice,
}

/// Host master copies plus their device mirrors, alongside
/// [`crate::genome::Population`].
pub struct BodyState {
  pub cfg: BodyCfg,
  /// One past the highest particle slot ever claimed, or 0 if none has been.
  /// Slots are claimed ascending, so everything above this is still `Free`
  /// and the per-particle kernels do not have to be launched over it
  /// ([`crate::kernels::LiveParticles`]). It only ever grows: a slot freed by
  /// a death leaves the mark where it was.
  pub high_water: usize,
  pub limb_particles: Vec<u32>,
  /// Each organism's anchor: the soil cell its root germinated in.
  pub anchors: Vec<Vec2>,
  /// The single particle a seed organism is, or [`NO_PARTICLE`]. A seed is
  /// not in [`Self::limb_particles`], which is why the constraint pass leaves
  /// it alone: it has no limbs to hold it, so it falls and drifts.
  pub seed_particles: Vec<u32>,
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
    let seed_particles = vec![NO_PARTICLE; cfg.max_organisms as usize];
    let geometry = LimbGeometryHost::new(cfg.limb_count());
    let device = BodyDevice {
      limb_particles: client.create_from_slice(bytemuck::cast_slice(&limb_particles)),
      anchors: client.create_from_slice(bytemuck::cast_slice(&flatten(&anchors))),
      seed_particles: client.create_from_slice(bytemuck::cast_slice(&seed_particles)),
      ppos_prev: client.empty(geom.num_particles * 2 * size_of::<f32>()),
      geometry: LimbGeometryDevice::upload(client, &geometry),
    };
    Self {
      cfg,
      high_water: 0,
      limb_particles,
      anchors,
      seed_particles,
      geometry,
      device,
    }
  }

  /// Every particle slot this organism holds: its limbs' and, if it is still
  /// a seed, the seed particle.
  pub fn organism_particles(&self, organism: usize) -> Vec<u32> {
    let cfg = self.cfg;
    let per_organism = (cfg.max_limbs * cfg.max_particles_per_limb) as usize;
    let start = organism * per_organism;
    let mut out: Vec<u32> = self.limb_particles[start..start + per_organism]
      .iter()
      .copied()
      .filter(|id| *id != NO_PARTICLE)
      .collect();
    if self.seed_particles[organism] != NO_PARTICLE {
      out.push(self.seed_particles[organism]);
    }
    out
  }

  /// Forget every particle `organism` held, and the seed slot with it.
  pub fn release(&mut self, organism: usize) {
    let cfg = self.cfg;
    let per_organism = (cfg.max_limbs * cfg.max_particles_per_limb) as usize;
    let start = organism * per_organism;
    self.limb_particles[start..start + per_organism].fill(NO_PARTICLE);
    self.seed_particles[organism] = NO_PARTICLE;
  }

  /// One past the highest claimed particle slot, recomputed from the map.
  ///
  /// [`Self::high_water`] only grows as slots are claimed; after deaths the
  /// top of the range can be empty again, and the per-particle kernels should
  /// not be launched over it (`TODO.md`, left open by the bodies chunk).
  pub fn recompute_high_water(&mut self) {
    let highest = self
      .limb_particles
      .iter()
      .chain(self.seed_particles.iter())
      .filter(|id| **id != NO_PARTICLE)
      .max();
    self.high_water = highest.map_or(0, |id| *id as usize + 1);
  }

  /// The particle holding `(organism, limb, index)`, or [`NO_PARTICLE`].
  pub fn particle(&self, organism: usize, limb: usize, index: usize) -> u32 {
    self.limb_particles[self.cfg.limb_slice(organism, limb).start + index]
  }

  /// Upload the host master copies of the map, the anchors and the seeds.
  pub fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.device.limb_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.limb_particles));
    self.device.anchors = client.create_from_slice(bytemuck::cast_slice(&flatten(&self.anchors)));
    self.device.seed_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.seed_particles));
  }
}

fn flatten(v: &[Vec2]) -> Vec<f32> {
  v.iter().flat_map(|p| [p.x, p.y]).collect()
}

// --- Spawning ---
//
// The host decides *which* limbs to grow and claims their particle slots; the
// device lays them out, because where a limb starts depends on where the
// constraint pass actually left its parent's last particle. `docs/organism.md`
// asks for growth to be a kernel over a list of requests, and
// `kernels::spawn` is that kernel pair.

/// Everything a spawn touches, borrowed from one [`Sim`] at once.
pub struct SimBodies<'a, R: Runtime> {
  pub client: &'a ComputeClient<R>,
  pub params: &'a SimParams,
  pub geom: &'a WorldGeometry,
  pub cfg: Cfg,
  pub sph: &'a SphDevice,
  /// The kernels' runtime parameter buffer, which the layout kernel reads.
  pub params_buf: &'a Handle,
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
}

/// One limb to grow: the sprout head's unit of work, and the unit a whole
/// body is laid out in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowRequest {
  pub organism: usize,
  pub limb: usize,
}

/// Install a genome in an organism slot and lay its whole body out from the
/// anchor.
///
/// Returns the particles placed. The lineage fields (`parent_id`,
/// `birth_step`, `lineage_id`) are the life cycle's; this sets only what a
/// body needs: the genome, the anchor and `alive`.
pub fn spawn<R: Runtime>(
  sim: &mut crate::sim::Sim<R>,
  organism: usize,
  genome: &Genome,
  anchor: Vec2,
) -> usize {
  install(sim, organism, genome, anchor);
  {
    let access = sim.body_access();
    access.pop.upload(access.client);
  }
  grow_bodies(sim, &[organism])
}

/// Write one organism's genome, anchor and `alive` flag into the host
/// mirrors. The caller uploads, so a batch of founders pays for one upload.
fn install<R: Runtime>(
  sim: &mut crate::sim::Sim<R>,
  organism: usize,
  genome: &Genome,
  anchor: Vec2,
) {
  let access = sim.body_access();
  access.pop.write_genome(organism, genome);
  access.pop.organisms.alive[organism] = 1;
  // A founder starts germinated: `--founders` is the run's initial
  // condition, not something the life cycle produced.
  access.pop.organisms.stage[organism] = crate::genome::population::STAGE_PLANT;
  access.bodies.anchors[organism] = anchor;
  access.bodies.release(organism);
}

/// Grow every ungrown limb of these organisms, parents before children.
///
/// A limb record's parent may sit at a *higher* index than the limb itself,
/// because a structural add takes the first absent record rather than the
/// next one, so one pass is not enough. Each pass is one batch — one layout
/// launch and one placement launch — and the loop stops as soon as a pass
/// places nothing.
pub fn grow_bodies<R: Runtime>(sim: &mut crate::sim::Sim<R>, organisms: &[usize]) -> usize {
  let max_limbs = sim.population().max_limbs;
  let requests: Vec<GrowRequest> = organisms
    .iter()
    .flat_map(|o| (0..max_limbs).map(move |limb| GrowRequest { organism: *o, limb }))
    .collect();
  let mut placed = 0;
  for _ in 0..max_limbs {
    let n = grow_limbs(sim, &requests);
    if n == 0 {
      break;
    }
    placed += n;
  }
  placed
}

/// Grow one batch of limbs: claim their particle slots, lay them out on the
/// device, and write the particles.
///
/// Requests that cannot be grown *yet* — an ungrown parent — are skipped
/// silently, which is what makes [`grow_bodies`]'s repeated passes work.
/// Returns the particles placed.
pub fn grow_limbs<R: Runtime>(sim: &mut crate::sim::Sim<R>, requests: &[GrowRequest]) -> usize {
  use crate::kernels::spawn::{GrowBatchEntry, Placement, launch_layout, launch_place};

  if requests.is_empty() {
    return 0;
  }
  let access = sim.body_access();
  let cfg = access.bodies.cfg;

  let mut batch: Vec<GrowBatchEntry> = Vec::new();
  let mut placement = Placement::default();
  let mut total = 0usize;
  for request in requests {
    let Ok(count) = growable(&access, request.organism, request.limb) else {
      continue;
    };
    let record = access.pop.limb_index(request.organism, request.limb);
    let part_type = access.pop.limbs.part_type[record] as u32;
    batch.push(GrowBatchEntry {
      organism: request.organism as u32,
      limb: request.limb as u32,
      first: total as u32,
      count: count as u32,
    });
    for i in 0..count {
      placement.organism.push(request.organism as u32);
      placement.limb.push(request.limb as u32);
      placement.index_in_limb.push(i as u32);
      placement.part_type.push(part_type);
    }
    total += count;
  }
  if total == 0 {
    return 0;
  }

  // Fewer free slots than the batch wants: drop whole limbs off the end
  // rather than growing a half limb. Deterministic, because the request
  // order is.
  let mut ids = claim_particles(&access, total);
  if ids.len() < total {
    while batch
      .last()
      .is_some_and(|e| (e.first + e.count) as usize > ids.len())
    {
      batch.pop();
    }
    total = batch.last().map_or(0, |e| (e.first + e.count) as usize);
    if total == 0 {
      return 0;
    }
    ids.truncate(total);
    placement.organism.truncate(total);
    placement.limb.truncate(total);
    placement.index_in_limb.truncate(total);
    placement.part_type.truncate(total);
  }
  placement.ids = ids.clone();

  // The device map and anchors are what the layout kernel reads, and the host
  // is the master of both, so they go up before the launch.
  access.bodies.upload(access.client);
  let positions = launch_layout(
    access.client,
    access.sph,
    &access.pop.device.limbs,
    &access.bodies.device.limb_particles,
    &access.bodies.device.anchors,
    &batch,
    total,
    access.params_buf,
    cfg,
    access.cfg,
  );
  launch_place(access.client, access.sph, &placement, Some(&positions));

  // The host map is the master; the kernel only read it. Record the claimed
  // slots and send the map back up, so the constraint pass sees the new limb
  // on the very next step.
  for entry in &batch {
    let slice = cfg.limb_slice(entry.organism as usize, entry.limb as usize);
    for i in 0..entry.count as usize {
      let id = ids[entry.first as usize + i];
      access.bodies.limb_particles[slice.start + i] = id;
      access.bodies.high_water = access.bodies.high_water.max(id as usize + 1);
    }
  }
  access.bodies.upload(access.client);
  total
}

/// Particles a limb would take, or why it cannot be grown now.
fn growable<R: Runtime>(
  access: &SimBodies<'_, R>,
  organism: usize,
  limb: usize,
) -> Result<usize, SpawnError> {
  let cfg = access.bodies.cfg;
  let mp = cfg.max_particles_per_limb as usize;
  let record = access.pop.limb_index(organism, limb);
  let count = (access.pop.limbs.length[record] as usize).min(mp);
  if !access.pop.limbs.part_type[record].is_present() || count == 0 {
    return Err(SpawnError::NothingToGrow);
  }
  let slice = cfg.limb_slice(organism, limb);
  if access.bodies.limb_particles[slice]
    .iter()
    .any(|id| *id != NO_PARTICLE)
  {
    return Err(SpawnError::NothingToGrow);
  }
  let parent = access.pop.limbs.parent[record] as usize;
  if parent == limb {
    return Ok(count);
  }
  let precord = access.pop.limb_index(organism, parent);
  let pcount = (access.pop.limbs.length[precord] as usize).min(mp);
  if !access.pop.limbs.part_type[precord].is_present() || pcount == 0 {
    return Err(SpawnError::ParentNotGrown);
  }
  if access.bodies.particle(organism, parent, pcount - 1) == NO_PARTICLE {
    return Err(SpawnError::ParentNotGrown);
  }
  Ok(count)
}

/// Whether a limb record could be grown right now — what the sprout head
/// asks before it spends anything.
pub fn can_grow<R: Runtime>(sim: &mut crate::sim::Sim<R>, organism: usize, limb: usize) -> bool {
  growable(&sim.body_access(), organism, limb).is_ok()
}

/// The first `n` claimable particle slots, ascending, or as many as there are.
///
/// The scan is the spec's allocator: an atomic claim would hand ids out in
/// arrival order and the run would stop being reproducible.
pub fn claim_particles<R: Runtime>(access: &SimBodies<'_, R>, n: usize) -> Vec<u32> {
  let total = access.geom.num_particles;
  let occupancy = access.client.empty(total * size_of::<u32>());
  crate::kernels::spawn::launch_mark_occupancy(access.client, access.sph, &occupancy, access.cfg);
  let scan = SlotScan::alloc(access.client, total);
  claim_free_slots(access.client, &occupancy, &scan);
  let mut free = read_free_slots(access.client, &scan);
  free.truncate(n);
  free
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
    install(sim, slot, &genome, anchor);
  }

  // A founder is the root ancestor of its own lineage and generation 0, which
  // is what the evolutionary metrics count from. It starts with a seed's worth
  // of energy so that its first interval is not spent dying; an offspring's
  // lineage fields are the life cycle's.
  let step = sim.step_count();
  let seed_energy = sim.params().seed_energy;
  {
    let access = sim.body_access();
    for slot in 0..count {
      access.pop.organisms.lineage_id[slot] = slot as u32;
      access.pop.organisms.parent_id[slot] = NO_PARENT;
      access.pop.organisms.birth_step[slot] = step;
      access.pop.organisms.generation[slot] = 0;
      access.pop.organisms.energy[slot] = seed_energy;
    }
    access.pop.upload(access.client);
  }

  // One batch per wave over every founder at once, rather than a whole
  // layout per organism: the founders' roots go down together, then their
  // stems, then their leaves.
  let slots: Vec<usize> = (0..count).collect();
  grow_bodies(sim, &slots);
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
