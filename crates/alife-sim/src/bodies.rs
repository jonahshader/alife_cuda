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

/// A limb's live thickness: its `target_thickness` scaled by the square root
/// of its progress, so it thickens as it lengthens (`docs/organism.md`,
/// *Growth*). `target` is the record's length capped at
/// `max_particles_per_limb`, and a limb with nowhere to grow has no
/// thickness.
pub fn limb_thickness(target_thickness: f32, target: f32, grown: f32) -> f32 {
  if target <= 0.0 {
    return 0.0;
  }
  target_thickness * (grown / target).clamp(0.0, 1.0).sqrt()
}

/// A limb's volume: `grown * limb_segment_length * thickness^2`.
///
/// With the thickness tied to `sqrt(grown / target)` this is
/// `segment_length * target_thickness^2 * grown^2 / target`, quadratic in
/// `grown` — which is what lets [`grown_from_volume`] invert it in closed
/// form. It is what upkeep is charged on and what growth buys
/// (`docs/organism.md`, *Growth*).
pub fn limb_volume(segment_length: f32, target_thickness: f32, target: f32, grown: f32) -> f32 {
  let thickness = limb_thickness(target_thickness, target, grown);
  grown * segment_length * thickness * thickness
}

/// The volume of one `(organism, limb)` as it stands: [`limb_volume`] at the
/// limb's `grown`, zero for an absent record.
pub fn limb_volume_of(
  pop: &Population,
  bodies: &BodyState,
  segment_length: f32,
  organism: usize,
  limb: usize,
) -> f32 {
  let record = pop.limb_index(organism, limb);
  if !pop.limbs.part_type[record].is_present() {
    return 0.0;
  }
  let mp = bodies.cfg.max_particles_per_limb as usize;
  let target = (pop.limbs.length[record] as usize).min(mp) as f32;
  limb_volume(
    segment_length,
    pop.limbs.target_thickness[record],
    target,
    bodies.grown[bodies.cfg.limb_index(organism, limb)],
  )
}

/// The volume of every limb an organism has grown: what upkeep is charged
/// on, what its water pool holds `water_capacity_per_volume` particles per
/// unit of, and what holds `water_per_volume` particles per unit as tissue
/// (`docs/organism.md`, *Water*).
pub fn organism_volume(
  pop: &Population,
  bodies: &BodyState,
  segment_length: f32,
  organism: usize,
) -> f32 {
  (0..pop.max_limbs)
    .map(|limb| limb_volume_of(pop, bodies, segment_length, organism, limb))
    .sum()
}

/// The length a volume buys: [`limb_volume`] inverted.
///
/// Zero when the limb has no volume to buy with — no length to grow into, or
/// no thickness to grow at, which the thickness floor and the params
/// validator between them keep out of a live genome.
pub fn grown_from_volume(
  segment_length: f32,
  target_thickness: f32,
  target: f32,
  volume: f32,
) -> f32 {
  if target <= 0.0 {
    return 0.0;
  }
  let quadratic = segment_length * target_thickness * target_thickness / target;
  if quadratic <= 0.0 {
    return 0.0;
  }
  (volume / quadratic).max(0.0).sqrt()
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

  /// Flat index of one `(organism, limb)` entry.
  pub fn limb_index(&self, organism: usize, limb: usize) -> usize {
    organism * self.max_limbs as usize + limb
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
  /// `[max_organisms x max_limbs]` lengths in segments; see
  /// [`BodyState::grown`].
  pub grown: Handle,
  /// `[max_organisms x max_limbs]` radians: the worst the base joint of each
  /// limb has been bent away from its rest angle since the last life tick.
  ///
  /// Device-mastered, and the only body buffer that is — the constraint pass
  /// accumulates it every sweep and [`crate::life::pack`] reads and zeroes it
  /// once a tick, so there is no host copy for [`BodyState::upload`] to
  /// clobber it with. `docs/organism.md`, *Growth*: a limb breaks on it.
  pub bend_max: Handle,
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
  /// One past the highest slot above the fluid range that may hold a fluid
  /// particle the life tick placed — water an organism gave back as liquid
  /// or transpired as vapor (`docs/organism.md`, *Water*) — or 0.
  ///
  /// Such a particle is in no organism's map, so [`Self::recompute_high_water`]
  /// cannot see it and the live prefix would stop covering it; this is the
  /// mark that keeps it covered. It grows as fluid is placed and is tightened
  /// by every particle claim, which reads the whole free list anyway and so
  /// knows where the highest occupied slot is. A root can drink such a
  /// particle without the host hearing of it, so the mark may sit above the
  /// last fluid particle — an over-estimate only launches a few `Free` slots.
  pub fluid_high_water: usize,
  pub limb_particles: Vec<u32>,
  /// Each organism's anchor: the soil cell its root germinated in.
  pub anchors: Vec<Vec2>,
  /// The single particle a seed organism is, or [`NO_PARTICLE`]. A seed is
  /// not in [`Self::limb_particles`], which is why the constraint pass leaves
  /// it alone: it has no limbs to hold it, so it falls and drifts.
  pub seed_particles: Vec<u32>,
  /// How long each `(organism, limb)` has grown, in segments, `0 ..= length`
  /// (`docs/organism.md`, *Growth*). Particle `k` of a limb exists while
  /// `grown > k`, so the map holds `ceil(grown)` of them and the last
  /// segment's rest length is the fractional remainder — a limb lengthens
  /// between particle claims rather than jumping a whole segment.
  ///
  /// State, not genome: the life tick is its master and a kernel only reads
  /// it. The particle map is the other half of the same fact, and the two
  /// are kept consistent by [`grow_limbs`], which is the only thing that
  /// claims a particle.
  pub grown: Vec<f32>,
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
    let grown = vec![0.0f32; cfg.limb_count()];
    let geometry = LimbGeometryHost::new(cfg.limb_count());
    let device = BodyDevice {
      limb_particles: client.create_from_slice(bytemuck::cast_slice(&limb_particles)),
      anchors: client.create_from_slice(bytemuck::cast_slice(&flatten(&anchors))),
      seed_particles: client.create_from_slice(bytemuck::cast_slice(&seed_particles)),
      grown: client.create_from_slice(bytemuck::cast_slice(&grown)),
      bend_max: client.create_from_slice(bytemuck::cast_slice(&vec![0.0f32; cfg.limb_count()])),
      ppos_prev: client.empty(geom.num_particles * 2 * size_of::<f32>()),
      geometry: LimbGeometryDevice::upload(client, &geometry),
    };
    Self {
      cfg,
      high_water: 0,
      fluid_high_water: 0,
      limb_particles,
      anchors,
      seed_particles,
      grown,
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

  /// Every particle `(organism, limb)` holds, in index order.
  pub fn limb_particles_of(&self, organism: usize, limb: usize) -> Vec<u32> {
    let slice = self.cfg.limb_slice(organism, limb);
    self.limb_particles[slice]
      .iter()
      .copied()
      .take_while(|id| *id != NO_PARTICLE)
      .collect()
  }

  /// Forget one limb's particles and its length: the limb is gone from the
  /// body but its record stays in the genome, so it grows again from nothing
  /// through the ordinary allocation (`docs/organism.md`, *Growth*, breaking).
  ///
  /// The caller frees the particle slots themselves — a break leaves them in
  /// the soil as organic matter, exactly as a death does.
  pub fn release_limb(&mut self, organism: usize, limb: usize) {
    let slice = self.cfg.limb_slice(organism, limb);
    self.limb_particles[slice].fill(NO_PARTICLE);
    self.grown[self.cfg.limb_index(organism, limb)] = 0.0;
  }

  /// Forget every particle `organism` held, and the seed slot with it. Its
  /// limbs are unlengthened too: the next organism in this slot starts from
  /// nothing.
  pub fn release(&mut self, organism: usize) {
    let cfg = self.cfg;
    let per_organism = (cfg.max_limbs * cfg.max_particles_per_limb) as usize;
    let start = organism * per_organism;
    self.limb_particles[start..start + per_organism].fill(NO_PARTICLE);
    self.seed_particles[organism] = NO_PARTICLE;
    let limbs = cfg.max_limbs as usize;
    self.grown[organism * limbs..(organism + 1) * limbs].fill(0.0);
  }

  /// Particles `(organism, limb)` actually holds: the leading run of the
  /// map's entries, which is `ceil(grown)` — particles are claimed in index
  /// order and a limb gives them all back at once, so the map's prefix is
  /// the whole of what a limb has grown.
  pub fn limb_len(&self, organism: usize, limb: usize) -> usize {
    let slice = self.cfg.limb_slice(organism, limb);
    self.limb_particles[slice]
      .iter()
      .take_while(|id| **id != NO_PARTICLE)
      .count()
  }

  /// One past the highest claimed particle slot, recomputed from the map
  /// and [`Self::fluid_high_water`].
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
    self.high_water = highest
      .map_or(0, |id| *id as usize + 1)
      .max(self.fluid_high_water);
  }

  /// The particle holding `(organism, limb, index)`, or [`NO_PARTICLE`].
  pub fn particle(&self, organism: usize, limb: usize, index: usize) -> u32 {
    self.limb_particles[self.cfg.limb_slice(organism, limb).start + index]
  }

  /// Upload the host master copies of the map, the anchors and the seeds.
  ///
  /// Not `bend_max`: the device is its master, and replacing its handle here
  /// would throw away whatever the constraint pass has accumulated since the
  /// last life tick — this runs mid-interval every time a limb grows.
  pub fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.device.limb_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.limb_particles));
    self.device.anchors = client.create_from_slice(bytemuck::cast_slice(&flatten(&self.anchors)));
    self.device.seed_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.seed_particles));
    self.device.grown = client.create_from_slice(bytemuck::cast_slice(&self.grown));
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

/// One limb's next particles: grow `(organism, limb)` until it holds `target`
/// of them.
///
/// A limb grows a particle at a time as `ceil(grown)` rises, so a request is a
/// *suffix* of a limb's chain rather than the whole of it; the first index it
/// covers is whatever the map already holds. A request whose target the limb
/// has already reached places nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrowRequest {
  pub organism: usize,
  pub limb: usize,
  pub target: usize,
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
  let mp = sim.body_cfg().max_particles_per_limb as usize;
  // A whole body: every present record at its full target length, which is
  // what a founder starts at and what a loaded population comes back as.
  let requests: Vec<GrowRequest> = organisms
    .iter()
    .flat_map(|o| {
      (0..max_limbs).map(move |limb| GrowRequest {
        organism: *o,
        limb,
        target: mp,
      })
    })
    .collect();
  {
    // A whole body is grown, so every present limb is at its full length.
    // `grow_limbs` only ever clamps this down, to what the particle capacity
    // actually had room for.
    let access = sim.body_access();
    let cfg = access.bodies.cfg;
    for o in organisms {
      for limb in 0..max_limbs {
        let record = access.pop.limb_index(*o, limb);
        if !access.pop.limbs.part_type[record].is_present() {
          continue;
        }
        let n = (access.pop.limbs.length[record] as usize).min(mp);
        access.bodies.grown[cfg.limb_index(*o, limb)] = n as f32;
      }
    }
  }

  let mut placed = 0;
  for _ in 0..max_limbs {
    let n = grow_limbs(sim, &requests);
    if n == 0 {
      break;
    }
    placed += n;
  }
  // `grow_limbs` reconciles only the limbs a pass admitted. A limb whose
  // parent never got its first particle — the pool ran dry — was admitted by
  // no pass, and its `grown` would otherwise stay at the full length over an
  // empty slice.
  {
    let access = sim.body_access();
    let cfg = access.bodies.cfg;
    for o in organisms {
      for limb in 0..max_limbs {
        let held = access.bodies.limb_len(*o, limb) as f32;
        let grown = &mut access.bodies.grown[cfg.limb_index(*o, limb)];
        *grown = grown.min(held);
      }
    }
    access.bodies.upload(access.client);
  }
  placed
}

/// Grow one batch of limbs: claim the particle slots their next segments
/// need, lay them out on the device, and write the particles.
///
/// The caller has already moved `grown` — it is the length the limb is
/// growing *to*, and the request's target is `ceil` of it. Requests that
/// cannot be grown *yet* — an ungrown parent — are skipped silently, which is
/// what makes [`grow_bodies`]'s repeated passes work, and a limb that gets
/// fewer particles than it asked for has its `grown` clamped back to what it
/// actually holds. Returns the particles placed.
pub fn grow_limbs<R: Runtime>(sim: &mut crate::sim::Sim<R>, requests: &[GrowRequest]) -> usize {
  use crate::kernels::spawn::{GrowBatchEntry, Placement, launch_layout, launch_place};

  if requests.is_empty() {
    return 0;
  }
  let mut access = sim.body_access();
  let cfg = access.bodies.cfg;

  let mut batch: Vec<GrowBatchEntry> = Vec::new();
  let mut placement = Placement::default();
  // Every limb this batch admitted, whether or not its particles survived the
  // capacity truncation below: each one's `grown` is reconciled with the map
  // once the placement has happened.
  let mut touched: Vec<(usize, usize)> = Vec::new();
  let mut total = 0usize;
  for request in requests {
    let Ok((first_index, count)) = growable(&access, request) else {
      continue;
    };
    let record = access.pop.limb_index(request.organism, request.limb);
    let part_type = access.pop.limbs.part_type[record] as u32;
    batch.push(GrowBatchEntry {
      organism: request.organism as u32,
      limb: request.limb as u32,
      first_index: first_index as u32,
      first: total as u32,
      count: count as u32,
    });
    for i in 0..count {
      placement.push_body(
        request.organism as u32,
        request.limb as u32,
        (first_index + i) as u32,
        part_type,
      );
    }
    touched.push((request.organism, request.limb));
    total += count;
  }
  if total == 0 {
    return 0;
  }

  // Fewer free slots than the batch wants: drop whole limbs off the end
  // rather than growing a half limb. Deterministic, because the request
  // order is.
  let mut ids = claim_particles(&mut access, total);
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
    placement.truncate(total);
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
    &access.bodies.device.grown,
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
    let (o, l) = (entry.organism as usize, entry.limb as usize);
    let slice = cfg.limb_slice(o, l);
    let first_index = entry.first_index as usize;
    for i in 0..entry.count as usize {
      let id = ids[entry.first as usize + i];
      access.bodies.limb_particles[slice.start + first_index + i] = id;
      access.bodies.high_water = access.bodies.high_water.max(id as usize + 1);
    }
  }
  // What a limb holds is what it has grown. The two only disagree when a
  // request was dropped for want of particle slots, and then the map is the
  // truth and `grown` comes back down to it.
  for (o, l) in touched {
    let held = access.bodies.limb_len(o, l) as f32;
    let grown = &mut access.bodies.grown[cfg.limb_index(o, l)];
    *grown = grown.min(held);
  }
  access.bodies.upload(access.client);
  total
}

/// `(first index, how many)` this request would place, or why it cannot be
/// grown now.
///
/// The first index is what the limb already holds: growth appends to a chain
/// rather than laying a whole one out, and a request that asks for no more
/// than is already there places nothing.
fn growable<R: Runtime>(
  access: &SimBodies<'_, R>,
  request: &GrowRequest,
) -> Result<(usize, usize), SpawnError> {
  let (organism, limb) = (request.organism, request.limb);
  let mp = access.bodies.cfg.max_particles_per_limb as usize;
  let record = access.pop.limb_index(organism, limb);
  let full = (access.pop.limbs.length[record] as usize).min(mp);
  if !access.pop.limbs.part_type[record].is_present() || full == 0 {
    return Err(SpawnError::NothingToGrow);
  }
  let target = request.target.min(full);
  let have = access.bodies.limb_len(organism, limb);
  if target <= have {
    return Err(SpawnError::NothingToGrow);
  }
  // Only the first particle hangs off the parent; the rest hang off this
  // limb's own chain, which is already standing.
  if have == 0 {
    let parent = access.pop.limbs.parent[record] as usize;
    if parent != limb && access.bodies.limb_len(organism, parent) == 0 {
      return Err(SpawnError::ParentNotGrown);
    }
  }
  Ok((have, target - have))
}

/// The first `n` claimable particle slots, ascending, or as many as there are.
///
/// The scan is the spec's allocator: an atomic claim would hand ids out in
/// arrival order and the run would stop being reproducible. It is over every
/// slot, the fluid range included: a liquid particle a root drank is a `Free`
/// slot like any other (`docs/organism.md`, *Water*), and the lowest free
/// slot is the first one handed out.
///
/// The free list comes back whole, so this is also where
/// [`BodyState::fluid_high_water`] learns how high the occupied slots
/// actually reach.
pub fn claim_particles<R: Runtime>(access: &mut SimBodies<'_, R>, n: usize) -> Vec<u32> {
  let total = access.geom.num_particles;
  let occupancy = access.client.empty(total * size_of::<u32>());
  crate::kernels::spawn::launch_mark_occupancy(access.client, access.sph, &occupancy, access.cfg);
  let scan = SlotScan::alloc(access.client, total);
  claim_free_slots(access.client, &occupancy, &scan);
  let mut free = read_free_slots(access.client, &scan);
  // The free list is ascending, so its trailing run of consecutive ids ending
  // at the last slot is exactly the empty top of the buffer.
  let empty_top = free
    .iter()
    .rev()
    .zip((0..total).rev())
    .take_while(|(id, slot)| **id as usize == *slot)
    .count();
  access.bodies.fluid_high_water = access.bodies.fluid_high_water.min(total - empty_top);
  free.truncate(n);
  free
}

/// Seed `count` founders: one [`Genome::seed_plant`] each, at evenly spaced
/// x, anchored on the soil surface under it.
///
/// "Evenly spaced" means over the soil, not over the world: the founders are
/// laid out along the union of [`crate::soil::SoilGrid::columns`], so a
/// capillary terrain's gaps get none and each column gets founders in
/// proportion to its width. A terrain whose only column spans the world — the
/// noise one — is the same even spacing across the world it always was. The
/// gaps matter because they have no soil surface at all: a founder in one
/// would anchor on the world floor, at the bottom of the pool, which is not a
/// habitat the experiment is asking about.
///
/// Returns how many were actually seeded, which is capped by the organism
/// slots. `--founders 0` does nothing at all.
pub fn spawn_founders<R: Runtime>(sim: &mut crate::sim::Sim<R>, count: usize) -> usize {
  let count = count.min(sim.population().max_organisms);
  if count == 0 {
    return 0;
  }
  let seed = sim.seed();
  let shape = sim.population().shape;
  let max_limbs = sim.population().max_limbs;
  let params = *sim.params();

  let anchors: Vec<Vec2> = (0..count)
    .map(|i| {
      let x = founder_x(sim.soil(), i, count);
      Vec2::new(x, soil_surface(sim.soil(), x))
    })
    .collect();

  for (slot, anchor) in anchors.into_iter().enumerate() {
    let genome = Genome::seed_plant(&shape, max_limbs, slot as u32, seed, &params);
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
  // A founder's body is grown whole rather than out of a pool, and it starts
  // with that pool full (`docs/organism.md`, *Water*).
  sim.endow_water(&slots, true);
  count
}

/// Move every organism standing in one soil column to another.
///
/// The transplant test of `docs/organism.md`'s *The first experiment*: take a
/// lineage from one column, put it in another, and watch what its energy does
/// against the residents'. An organism keeps its *relative* x within the
/// column, so a plant a third of the way across the sand column lands a third
/// of the way across the clay one however differently wide the two are, and it
/// re-anchors on the destination's own soil surface.
///
/// Every move's membership is read from the anchors as they stand *before* any
/// of them is applied, so `sand:clay` and `clay:sand` given together swap the
/// two columns rather than piling both into one. Returns the slots each move
/// took, in the order the moves were given.
///
/// Seeds in flight are not moved: they have no anchor to move (`crate::life`).
pub fn transplant_anchors(
  soil: &SoilGrid,
  organisms: &crate::genome::OrganismHost,
  anchors: &mut [Vec2],
  moves: &[(&str, &str)],
) -> Result<Vec<Vec<usize>>, String> {
  use crate::genome::population::STAGE_PLANT;

  let columns = soil.columns();
  let find = |label: &str| -> Result<crate::soil::ColumnExtent, String> {
    columns
      .iter()
      .find(|c| c.label == label)
      .copied()
      .ok_or_else(|| {
        let known: Vec<&str> = columns.iter().map(|c| c.label).collect();
        format!("no soil column is labelled `{label}`; this terrain has {known:?}")
      })
  };

  let mut plan = Vec::with_capacity(moves.len());
  for (from, to) in moves {
    let (source, destination) = (find(from)?, find(to)?);
    let slots: Vec<usize> = (0..anchors.len())
      .filter(|o| {
        organisms.alive[*o] == 1
          && organisms.stage[*o] == STAGE_PLANT
          && source.contains(soil.cell_column(anchors[*o].x))
      })
      .collect();
    plan.push((source, destination, slots));
  }

  let cell = soil.cell_size;
  // Every move reads the anchors as they were before any move: two moves
  // that share a source column would otherwise compute the second one's
  // relative position from anchors the first had already rewritten. When
  // they overlap, the later move wins.
  let before: Vec<Vec2> = anchors.to_vec();
  let mut moved = Vec::with_capacity(plan.len());
  for (source, destination, slots) in plan {
    for o in &slots {
      let span = (source.width() as f32 * cell).max(f32::MIN_POSITIVE);
      let relative = ((before[*o].x - source.x0 as f32 * cell) / span).clamp(0.0, 1.0);
      let x0 = destination.x0 as f32 * cell;
      // Half a cell short of the right edge, so a plant at the far end of a
      // wider source column still lands inside the destination rather than in
      // the gap past it.
      let x = (x0 + relative * destination.width() as f32 * cell)
        .min(destination.x1 as f32 * cell - cell * 0.5)
        .max(x0);
      anchors[*o] = Vec2::new(x, soil_surface(soil, x));
    }
    moved.push(slots);
  }
  Ok(moved)
}

/// Where founder `i` of `count` stands: the point `(i + 0.5) / count` of the
/// way along the soil, measured across the columns in order.
fn founder_x(soil: &SoilGrid, i: usize, count: usize) -> f32 {
  let cell = soil.cell_size;
  let columns = soil.columns();
  let total: f32 = columns.iter().map(|c| c.width() as f32 * cell).sum();
  let target = (i as f32 + 0.5) * total / count as f32;

  let mut passed = 0.0f32;
  for column in &columns {
    let span = column.width() as f32 * cell;
    if target < passed + span {
      return column.x0 as f32 * cell + (target - passed);
    }
    passed += span;
  }
  // Past the end of the last column, which only rounding can reach.
  columns.last().map_or(0.0, |c| (c.x1 as f32 - 0.5) * cell)
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
      // Evenly spaced across the world, on the surface of that column. Exact,
      // not approximate: the noise terrain's single column spans the world, so
      // spacing the founders along the soil is the same arithmetic as spacing
      // them along the world and has to stay bit-identical to it.
      let expected_x = (o as f32 + 0.5) * sim.geometry().bounds.x / 2.0;
      assert_eq!(anchor.x, expected_x);
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

  /// A mode-2 world narrow enough that the `width / 40` gap is zero, so its
  /// six columns tile the soil and six founders land one per column.
  fn six_columns() -> Sim<CpuRuntime> {
    let params = SimParams {
      world_width: 3.0,
      world_height: 2.0,
      smoothing_radius: 0.5,
      soil_cell_size: 0.1,
      particles_per_cell: 1,
      terrain_mode: 2,
      max_organisms: 8,
      max_limbs: 4,
      max_particles_per_limb: 3,
      life_interval: 100_000,
      ..SimParams::default()
    };
    Sim::new(CpuRuntime::client(&CpuDevice), params, 42, None)
  }

  /// The gaps between the capillary columns get no founders: every one stands
  /// on soil, and the columns share them out by width.
  #[test]
  fn founders_go_on_the_soil_and_not_in_the_gaps() {
    let params = SimParams {
      terrain_mode: 2,
      max_organisms: 60,
      ..SimParams::default()
    };
    let mut sim = Sim::new(CpuRuntime::client(&CpuDevice), params, 42, None);
    assert_eq!(spawn_founders(&mut sim, 60), 60);

    let columns = sim.soil().columns();
    // At the default width the columns really do leave gaps, so this is not a
    // vacuous check.
    assert!(columns.windows(2).any(|p| p[0].x1 < p[1].x0));
    let mut per_column = vec![0usize; columns.len()];
    for o in 0..60usize {
      let anchor = sim.bodies().anchors[o];
      let cell = sim.soil().cell_column(anchor.x);
      let c = columns
        .iter()
        .position(|column| column.contains(cell))
        .unwrap_or_else(|| panic!("founder {o} at x={} is in a gap", anchor.x));
      per_column[c] += 1;
      assert_eq!(anchor.y, soil_surface(sim.soil(), anchor.x));
      // On a column, "the soil surface" is the top of the column rather than
      // the floor of the pool.
      assert!(
        anchor.y > sim.geometry().bounds.y * 0.2,
        "founder {o} is in the pool"
      );
    }
    // Six columns of nearly equal width, so ten founders each, give or take
    // the rounding of the gaps.
    assert_eq!(per_column.iter().sum::<usize>(), 60);
    assert!(
      per_column.iter().all(|n| (9..=11).contains(n)),
      "{per_column:?}"
    );
  }

  #[test]
  fn a_transplant_moves_exactly_the_organisms_of_one_column() {
    let mut sim = six_columns();
    assert_eq!(spawn_founders(&mut sim, 6), 6);
    let columns = sim.soil().columns();
    let before = sim.bodies().anchors.clone();
    // One founder per column, in order, which is what makes the assertions
    // below about *which* slots moved mean anything.
    for (o, column) in columns.iter().enumerate() {
      assert!(column.contains(sim.soil().cell_column(before[o].x)), "{o}");
    }

    let mut anchors = before.clone();
    let moved = transplant_anchors(
      sim.soil(),
      &sim.population().organisms,
      &mut anchors,
      &[("sand", "clay"), ("clay", "sand")],
    )
    .unwrap();

    // Exactly the one organism of each named column, and nobody else.
    assert_eq!(moved, vec![vec![0usize], vec![2usize]]);
    for o in 0..8usize {
      if o == 0 || o == 2 {
        continue;
      }
      assert_eq!(anchors[o], before[o], "slot {o} moved and should not have");
    }

    // Each lands at the same relative x in the other column, on its surface.
    let sand = columns[0];
    let clay = columns[2];
    let cell = sim.soil().cell_size;
    let relative =
      |x: f32, c: crate::soil::ColumnExtent| (x - c.x0 as f32 * cell) / (c.width() as f32 * cell);
    assert!((relative(anchors[0].x, clay) - relative(before[0].x, sand)).abs() < 1e-5);
    assert!((relative(anchors[2].x, sand) - relative(before[2].x, clay)).abs() < 1e-5);
    assert!(clay.contains(sim.soil().cell_column(anchors[0].x)));
    assert!(sand.contains(sim.soil().cell_column(anchors[2].x)));
    for o in [0usize, 2] {
      assert_eq!(anchors[o].y, soil_surface(sim.soil(), anchors[o].x));
    }
  }

  #[test]
  fn a_transplant_of_an_unknown_column_is_an_error() {
    let mut sim = six_columns();
    spawn_founders(&mut sim, 6);
    let mut anchors = sim.bodies().anchors.clone();
    let err = transplant_anchors(
      sim.soil(),
      &sim.population().organisms,
      &mut anchors,
      &[("sand", "loam")],
    )
    .unwrap_err();
    assert!(err.contains("loam"), "{err}");
    assert_eq!(
      anchors,
      sim.bodies().anchors,
      "a failed move touched nothing"
    );
  }

  /// The three volume rules in one: a limb at full length is at its target
  /// thickness, volume is quadratic in the length, and the length solves back
  /// out of a volume exactly.
  #[test]
  fn volume_follows_the_length_squared_and_inverts() {
    let (seg, target_thickness, target) = (0.15f32, 0.04f32, 4.0f32);

    assert_eq!(limb_thickness(target_thickness, target, target), 0.04);
    assert_eq!(limb_thickness(target_thickness, target, 0.0), 0.0);
    // A quarter of the way along is half the thickness: `sqrt(f)`.
    assert!((limb_thickness(target_thickness, target, 1.0) - 0.02).abs() < 1e-7);

    let full = limb_volume(seg, target_thickness, target, target);
    assert!((full - target * seg * target_thickness * target_thickness).abs() < 1e-9);
    // Half the length is a quarter of the volume.
    let half = limb_volume(seg, target_thickness, target, 0.5 * target);
    assert!((half - 0.25 * full).abs() < 1e-9, "{half} against {full}");

    for grown in [0.0f32, 0.3, 1.0, 2.5, 4.0] {
      let volume = limb_volume(seg, target_thickness, target, grown);
      let back = grown_from_volume(seg, target_thickness, target, volume);
      assert!((back - grown).abs() < 1e-5, "{grown} came back as {back}");
    }

    // Nothing to grow into, or nothing to grow at: no volume either way, and
    // no division by it.
    assert_eq!(limb_volume(seg, target_thickness, 0.0, 2.0), 0.0);
    assert_eq!(grown_from_volume(seg, target_thickness, 0.0, 1.0), 0.0);
    assert_eq!(grown_from_volume(seg, 0.0, target, 1.0), 0.0);
  }

  /// Shedding one limb takes its particles and its length and nothing else:
  /// the record stays in the genome, so the limb can grow back.
  #[test]
  fn a_released_limb_keeps_its_record_and_loses_its_body() {
    let mut sim = small_world();
    assert_eq!(spawn_founders(&mut sim, 1), 1);
    let cfg = sim.body_cfg();
    let stem = sim.bodies().limb_particles_of(0, 1);
    let root = sim.bodies().limb_particles_of(0, 0);
    assert!(!stem.is_empty() && !root.is_empty());

    let access = sim.body_access();
    access.bodies.release_limb(0, 1);

    assert!(sim.bodies().limb_particles_of(0, 1).is_empty());
    assert_eq!(sim.bodies().limb_len(0, 1), 0);
    assert_eq!(sim.bodies().grown[cfg.limb_index(0, 1)], 0.0);
    // The genome still says what to grow back, and the root is untouched.
    let record = sim.population().limb_index(0, 1);
    assert_eq!(sim.population().limbs.part_type[record], PartType::Stem);
    assert_eq!(sim.bodies().limb_particles_of(0, 0), root);
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
