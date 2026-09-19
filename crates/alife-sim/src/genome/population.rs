//! The population tensors: every organism slot's genome and runtime state.
//!
//! Capacity is fixed at startup, as the particle system's is. A slot is either
//! alive or free; births claim free slots in order (`super::slots`), so
//! nothing here ever grows.
//!
//! Layout, from `docs/organism.md`'s *Genome buffers*:
//!
//! - the discrete section is `[max_organisms x max_limbs]` limb records,
//!   indexed `organism * max_limbs + limb`;
//! - the continuous section is `[max_organisms x param_count()]` fp32, with
//!   the host copy as the master (the brain chunk adds an fp16 device shadow);
//! - the latent state is `[max_organisms x n_latents x d_latent]` fp32, seeded
//!   from each organism's `latent_init` slice. It is state, not genome.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::shape::{BrainShape, IDENTITY_DIM};
use crate::SimParams;
use crate::define_soa;
use crate::soa::SoaField;

/// What a limb record is. `Absent` is the empty record, which is why it is
/// zero: a zeroed discrete section is an organism with no body.
///
/// The codes are also the row index into the brain's part-type embedding
/// table, so they are bounded by [`super::shape::N_TYPES`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PartType {
  #[default]
  Absent = 0,
  /// Pinned in soil, draws water from the cell's saturation.
  Root = 1,
  /// Structure.
  Stem = 2,
  /// Gains energy from light.
  Leaf = 3,
  /// A reproduction particle.
  Seed = 4,
  /// Milestone 2 and 3 types — actuated limb, digger, mouth — keep their
  /// codes reserved so the embedding table's height never changes.
  Reserved5 = 5,
  Reserved6 = 6,
  Reserved7 = 7,
}

/// The part types a structural mutation may introduce, in code order.
pub const MUTABLE_TYPES: [PartType; 4] = [
  PartType::Root,
  PartType::Stem,
  PartType::Leaf,
  PartType::Seed,
];

impl PartType {
  /// Only 0..=7 are ever stored — the validator caps a type code at the
  /// embedding table's height — so anything else is a corrupt record and
  /// reads back as the empty one.
  pub fn from_byte(b: u8) -> Self {
    match b {
      1 => PartType::Root,
      2 => PartType::Stem,
      3 => PartType::Leaf,
      4 => PartType::Seed,
      5 => PartType::Reserved5,
      6 => PartType::Reserved6,
      7 => PartType::Reserved7,
      _ => PartType::Absent,
    }
  }

  pub fn is_present(self) -> bool {
    self != PartType::Absent
  }
}

impl SoaField for PartType {
  type Device = u32;
  type Raw = u8;
  const COMPONENTS: usize = 1;

  fn write_device(self, out: &mut Vec<u32>) {
    out.push(self as u32);
  }

  fn read_device(src: &[u32]) -> Self {
    Self::from_byte(src[0] as u8)
  }

  fn to_raw(self) -> u8 {
    self as u8
  }

  fn from_raw(raw: u8) -> Self {
    Self::from_byte(raw)
  }
}

define_soa! {
    /// The discrete section: one record per `(organism, limb)` pair.
    LimbHost / LimbDevice {
        part_type: PartType,
        /// Particles in the limb's chain, capped by `max_particles_per_limb`.
        length: u8,
        /// Index of the parent limb within the same organism. The root limb
        /// is its own parent, which is what terminates a walk to the root.
        parent: u8,
        /// Which of the parent's child slots this limb occupies.
        child_slot: u8,
        /// Rest angle at the base joint, against the parent segment's
        /// direction. The root limb, being its own parent, measures against
        /// the world frame instead: 0 is +x, angles increase counter-clockwise.
        grow_angle: f32,
        /// Per-limb identity, so the brain can tell identical limbs apart.
        identity: [f32; IDENTITY_DIM],
    }
}

define_soa! {
    /// Per-organism lineage and runtime state. One entry per organism slot.
    OrganismHost / OrganismDevice {
        /// 1 while the slot holds an organism, 0 while it is free. A u32
        /// because the slot allocator scans it as a kernel flag buffer.
        alive: u32,
        /// Organism slot this one was born from, or [`NO_PARENT`].
        parent_id: u32,
        birth_step: u32,
        /// Root ancestor's slot: the lineage this organism belongs to.
        lineage_id: u32,
        energy: f32,
    }
}

/// `parent_id` / `lineage_id` of an organism with no parent — a seeded
/// founder rather than an offspring.
pub const NO_PARENT: u32 = u32::MAX;

/// The discrete section as kernel arguments. Field order matches [`LimbHost`].
///
/// The byte-wide fields are `u32` on the device, as [`crate::soa::SoaField`]
/// spells out; `identity` is [`IDENTITY_DIM`] contiguous floats per record, so
/// record `i`'s component `k` is at `i * IDENTITY_DIM + k`.
#[derive(CubeLaunch, CubeType)]
pub struct LimbArgs {
  pub part_type: Box<[u32]>,
  pub length: Box<[u32]>,
  pub parent: Box<[u32]>,
  pub child_slot: Box<[u32]>,
  pub grow_angle: Box<[f32]>,
  pub identity: Box<[f32]>,
}

/// Launch arguments are consumed by a launch, so every launch site builds its
/// own from the long-lived handles.
pub fn limb_args<R: Runtime>(limbs: &LimbDevice) -> LimbArgsLaunch<R> {
  let n = limbs.len();
  let whole = crate::kernels::whole::<R>;
  LimbArgsLaunch::new(
    whole(&limbs.part_type, n),
    whole(&limbs.length, n),
    whole(&limbs.parent, n),
    whole(&limbs.child_slot, n),
    whole(&limbs.grow_angle, n),
    whole(&limbs.identity, n * IDENTITY_DIM),
  )
}

/// One limb record, outside the SoA.
///
/// The population buffers are the working representation; this is the one an
/// organism is built or read back as, where a whole record is in hand at once
/// (`super::seed`, `super::species`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct LimbRecord {
  pub part_type: PartType,
  pub length: u8,
  pub parent: u8,
  pub child_slot: u8,
  pub grow_angle: f32,
  pub identity: [f32; IDENTITY_DIM],
}

/// One organism's genome: `max_limbs` records and one brain row.
#[derive(Debug, Clone, PartialEq)]
pub struct Genome {
  pub limbs: Vec<LimbRecord>,
  pub brain: Vec<f32>,
}

impl Genome {
  /// An empty genome of the right shape: no limbs, zero brain.
  pub fn empty(shape: &BrainShape, max_limbs: usize) -> Self {
    Self {
      limbs: vec![LimbRecord::default(); max_limbs],
      brain: vec![0.0; shape.param_count()],
    }
  }
}

/// The device mirror of every population buffer.
#[derive(Debug, Clone)]
pub struct PopulationDevice {
  pub limbs: LimbDevice,
  pub organisms: OrganismDevice,
  /// `[max_organisms x param_count()]` fp32, the brain tensor.
  pub brain: Handle,
  /// `[max_organisms x n_latents x d_latent]` fp32, the persistent latents.
  pub latents: Handle,
}

/// Host master copies plus their device mirrors.
pub struct Population {
  pub shape: BrainShape,
  pub max_organisms: usize,
  pub max_limbs: usize,
  /// Cap on a limb record's `length`, from `--max-particles-per-limb`.
  pub max_particles_per_limb: u8,
  pub limbs: LimbHost,
  pub organisms: OrganismHost,
  /// `[max_organisms x param_count()]`, row-major. The host copy is the
  /// master: the fp16 device shadow the brain chunk adds is derived from it.
  pub brain: Vec<f32>,
  /// `[max_organisms x n_latents x d_latent]`, row-major.
  pub latents: Vec<f32>,
  pub device: PopulationDevice,
}

impl Population {
  /// An empty population: every slot free, every record absent, every brain
  /// parameter zero. Callers seed the slots they want (`super::seed`).
  pub fn new<R: Runtime>(client: &ComputeClient<R>, params: &SimParams, shape: BrainShape) -> Self {
    let max_organisms = params.max_organisms.max(1) as usize;
    let max_limbs = params.max_limbs.max(1) as usize;
    let limbs = LimbHost::new(max_organisms * max_limbs);
    let mut organisms = OrganismHost::new(max_organisms);
    organisms.parent_id.fill(NO_PARENT);
    organisms.lineage_id.fill(NO_PARENT);
    let brain = vec![0.0f32; max_organisms * shape.param_count()];
    let latents = vec![0.0f32; max_organisms * shape.latent_state_len()];

    let device = PopulationDevice {
      limbs: LimbDevice::upload(client, &limbs),
      organisms: OrganismDevice::upload(client, &organisms),
      brain: client.create_from_slice(bytemuck::cast_slice(&brain)),
      latents: client.create_from_slice(bytemuck::cast_slice(&latents)),
    };

    Self {
      shape,
      max_organisms,
      max_limbs,
      max_particles_per_limb: params.max_particles_per_limb.clamp(1, 255) as u8,
      limbs,
      organisms,
      brain,
      latents,
      device,
    }
  }

  /// Flat index of one limb record.
  pub fn limb_index(&self, organism: usize, limb: usize) -> usize {
    debug_assert!(limb < self.max_limbs);
    organism * self.max_limbs + limb
  }

  /// One organism's limb records.
  pub fn limb_range(&self, organism: usize) -> std::ops::Range<usize> {
    let start = organism * self.max_limbs;
    start..start + self.max_limbs
  }

  pub fn brain_row(&self, organism: usize) -> &[f32] {
    let n = self.shape.param_count();
    &self.brain[organism * n..(organism + 1) * n]
  }

  pub fn brain_row_mut(&mut self, organism: usize) -> &mut [f32] {
    let n = self.shape.param_count();
    &mut self.brain[organism * n..(organism + 1) * n]
  }

  pub fn latent_row(&self, organism: usize) -> &[f32] {
    let n = self.shape.latent_state_len();
    &self.latents[organism * n..(organism + 1) * n]
  }

  /// Copy every organism's `latent_init` slice into its latent state, which
  /// is what birth does. The two are the same length by construction.
  pub fn reset_latents_from_brain(&mut self) {
    let init = self.shape.latent_init();
    let params = self.shape.param_count();
    let state = self.shape.latent_state_len();
    for organism in 0..self.max_organisms {
      let src = organism * params + init.start;
      let dst = organism * state;
      for i in 0..state {
        self.latents[dst + i] = self.brain[src + i];
      }
    }
  }

  /// Install one organism's genome into a slot, including its latent state.
  pub fn write_genome(&mut self, organism: usize, genome: &Genome) {
    assert_eq!(genome.limbs.len(), self.max_limbs);
    assert_eq!(genome.brain.len(), self.shape.param_count());
    for (limb, record) in genome.limbs.iter().enumerate() {
      let i = self.limb_index(organism, limb);
      self.limbs.part_type[i] = record.part_type;
      self.limbs.length[i] = record.length;
      self.limbs.parent[i] = record.parent;
      self.limbs.child_slot[i] = record.child_slot;
      self.limbs.grow_angle[i] = record.grow_angle;
      self.limbs.identity[i] = record.identity;
    }
    self.brain_row_mut(organism).copy_from_slice(&genome.brain);
    let init = self.shape.latent_init();
    let state = self.shape.latent_state_len();
    let dst = organism * state;
    self.latents[dst..dst + state].copy_from_slice(&genome.brain[init.clone()]);
  }

  /// Host to device, every buffer. Handles are replaced, as
  /// [`crate::particles::SphDevice::upload`] does.
  pub fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.device = PopulationDevice {
      limbs: LimbDevice::upload(client, &self.limbs),
      organisms: OrganismDevice::upload(client, &self.organisms),
      brain: client.create_from_slice(bytemuck::cast_slice(&self.brain)),
      latents: client.create_from_slice(bytemuck::cast_slice(&self.latents)),
    };
  }

  /// Device to host, every buffer.
  pub fn download<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.limbs = self.device.limbs.download(client);
    self.organisms = self.device.organisms.download(client);
    self.brain = read_f32(client, &self.device.brain, self.brain.len());
    self.latents = read_f32(client, &self.device.latents, self.latents.len());
  }
}

/// Read `n` floats back from a device buffer.
pub fn read_f32<R: Runtime>(client: &ComputeClient<R>, handle: &Handle, n: usize) -> Vec<f32> {
  let bytes = client.read_one_unchecked(handle.clone());
  bytemuck::cast_slice::<u8, f32>(&bytes)[..n].to_vec()
}

#[cfg(test)]
mod tests {
  use super::*;
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  fn small_params() -> SimParams {
    SimParams {
      max_organisms: 4,
      max_limbs: 5,
      brain_d_token: 4,
      brain_d_latent: 6,
      brain_n_latents: 2,
      brain_trunk_hidden: 3,
      ..SimParams::default()
    }
  }

  #[test]
  fn part_type_round_trips_every_code() {
    for code in 0u8..8 {
      assert_eq!(PartType::from_byte(code) as u8, code);
    }
    assert_eq!(PartType::from_byte(8), PartType::Absent);
    assert!(!PartType::Absent.is_present());
    assert!(PartType::Root.is_present());
  }

  #[test]
  fn a_new_population_is_empty_and_every_slot_is_free() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = small_params();
    let shape = BrainShape::from_params(&params);
    let pop = Population::new(&client, &params, shape);

    assert_eq!(pop.limbs.len(), 4 * 5);
    assert_eq!(pop.organisms.len(), 4);
    assert_eq!(pop.brain.len(), 4 * shape.param_count());
    assert_eq!(pop.latents.len(), 4 * shape.latent_state_len());
    assert!(pop.organisms.alive.iter().all(|a| *a == 0));
    assert!(pop.organisms.parent_id.iter().all(|p| *p == NO_PARENT));
    assert!(pop.limbs.part_type.iter().all(|t| !t.is_present()));
  }

  #[test]
  fn upload_then_download_round_trips_every_buffer() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = small_params();
    let shape = BrainShape::from_params(&params);
    let mut pop = Population::new(&client, &params, shape);

    // Something distinguishable in every field, including the reserved
    // part-type codes and the four identity components.
    for i in 0..pop.limbs.len() {
      pop.limbs.part_type[i] = PartType::from_byte((i % 8) as u8);
      pop.limbs.length[i] = (i % 7) as u8;
      pop.limbs.parent[i] = (i % 5) as u8;
      pop.limbs.child_slot[i] = (i % 3) as u8;
      pop.limbs.grow_angle[i] = i as f32 * 0.25;
      pop.limbs.identity[i] = [i as f32, -1.0, 0.5, i as f32 * 2.0];
    }
    for i in 0..pop.organisms.len() {
      pop.organisms.alive[i] = (i % 2) as u32;
      pop.organisms.parent_id[i] = i as u32;
      pop.organisms.birth_step[i] = 100 + i as u32;
      pop.organisms.lineage_id[i] = 7;
      pop.organisms.energy[i] = i as f32 * -3.5;
    }
    for (i, v) in pop.brain.iter_mut().enumerate() {
      *v = i as f32 * 1e-3;
    }
    pop.reset_latents_from_brain();

    let expected = (
      pop.limbs.clone(),
      pop.organisms.clone(),
      pop.brain.clone(),
      pop.latents.clone(),
    );
    pop.upload(&client);
    pop.limbs = LimbHost::new(0);
    pop.download(&client);

    assert_eq!(pop.limbs, expected.0);
    assert_eq!(pop.organisms, expected.1);
    assert_eq!(pop.brain, expected.2);
    assert_eq!(pop.latents, expected.3);
  }

  #[test]
  fn latents_are_seeded_from_the_latent_init_slice() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = small_params();
    let shape = BrainShape::from_params(&params);
    let mut pop = Population::new(&client, &params, shape);
    for (i, v) in pop.brain.iter_mut().enumerate() {
      *v = i as f32;
    }
    pop.reset_latents_from_brain();

    let init = shape.latent_init();
    for organism in 0..pop.max_organisms {
      assert_eq!(
        pop.latent_row(organism),
        &pop.brain_row(organism)[init.clone()]
      );
    }
  }

  #[test]
  fn limb_index_is_organism_major() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = small_params();
    let pop = Population::new(&client, &params, BrainShape::from_params(&params));
    assert_eq!(pop.limb_index(0, 0), 0);
    assert_eq!(pop.limb_index(2, 3), 2 * 5 + 3);
    assert_eq!(pop.limb_range(2), 10..15);
  }
}
