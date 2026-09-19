//! The `--save-pop` / `--load-pop` binary format: a population, without the
//! world it was living in.
//!
//! `--dump` carries the particles and no organisms; this carries the organisms
//! and no particles. The two are independent because a body is not state worth
//! storing — it is a function of the genome and the anchor, and re-growing it
//! costs one batch of growth launches — while a genome is the only thing in
//! the run that cannot be recomputed.
//!
//! Layout (little-endian):
//!
//! ```text
//!   char[8]  magic "ALIFEPOP"
//!   uint32   version (3)
//!   uint32   organism slots (max_organisms)
//!   uint32   limb records per organism (max_limbs)
//!   uint32   brain parameters per organism
//!   uint32   latent state floats per organism
//!   uint32   step count the run had reached
//!   uint64   resolved seed of the run that wrote it
//!   then one contiguous array per field, in SoA declaration order:
//!   the organism SoA over `max_organisms` entries — alive, parent_id,
//!     birth_step, lineage_id (uint32), energy (float), generation, stage
//!     (uint32);
//!   the limb SoA over `max_organisms * max_limbs` records — part_type,
//!     length, parent, child_slot (uint8), grow_angle (float), identity
//!     (IDENTITY_DIM floats), growth_bias, growth_rate, target_thickness
//!     (float);
//!   brain[max_organisms * param_count] (float);
//!   latents[max_organisms * latent_len] (float);
//!   anchors[max_organisms] (float2).
//! ```
//!
//! As with `dump.rs`, the field order is the SoA declaration order, so adding
//! a field there extends this format. The older versions are the same file
//! without the limb SoA's trailing float arrays, and they still load, because
//! the arrays before them are a prefix of the current ones: version 1 stops
//! before `growth_bias` and comes back with a founder's growth genes (bias 0,
//! rate `--growth-rate`), version 2 stops before `target_thickness` and comes
//! back with the `--thickness-*` param of each record's part type. The four
//! shape words in the header are checked against the run's own before a byte
//! of it is used.
//!
//! **What a load restores, and what it does not.** The genomes, the lineage
//! fields, the energies, the latent state and the anchors come back exactly as
//! they were written; the bodies are re-grown from the genomes at those
//! anchors through the ordinary growth path, so a limb sits where growth would
//! put it rather than where the fluid had pushed it. Seeds in flight are
//! dropped: a seed is a particle in mid-air, which is world state, and the
//! world it was falling through is gone. The step count and the seed in the
//! header are provenance — they are printed, not applied, because a load
//! starts a fresh world at step 0.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use cubecl::prelude::Runtime;
use glam::Vec2;

use crate::bodies::{self, BodyState};
use crate::genome::population::{STAGE_PLANT, STAGE_SEED};
use crate::genome::{LimbHost, OrganismHost, Population};
use crate::sim::Sim;

pub const MAGIC: &[u8; 8] = b"ALIFEPOP";
pub const VERSION: u32 = 3;

/// Limb SoA fields a file of version `v` carries: version 1 stopped before
/// `growth_bias`, version 2 before `target_thickness`.
fn limb_fields(version: u32) -> usize {
  match version {
    0 | 1 => LimbHost::FIELD_COUNT - 3,
    2 => LimbHost::FIELD_COUNT - 1,
    _ => LimbHost::FIELD_COUNT,
  }
}

/// Bytes before the first field array.
const HEADER_BYTES: u64 = 8 + 4 * 6 + 8;

#[derive(Debug, thiserror::Error)]
pub enum PopDumpError {
  #[error("io error: {0}")]
  Io(#[from] std::io::Error),
  #[error("not an alife population file (bad magic)")]
  BadMagic,
  #[error("unsupported population file version {0} (this build writes {VERSION})")]
  BadVersion(u32),
  #[error("population file is {actual} bytes but its header implies {expected} (truncated)")]
  Truncated { expected: u64, actual: u64 },
  #[error(
    "population file holds {found} {what} but this run has {expected}; \
     it was written by a run with different parameters"
  )]
  ShapeMismatch {
    what: &'static str,
    found: usize,
    expected: usize,
  },
  #[error("--load-pop needs an empty population, but {alive} organism slots are already taken")]
  NotEmpty { alive: usize },
}

/// A population as it was written, before anything is done with it.
#[derive(Debug, Clone, PartialEq)]
pub struct PopSnapshot {
  /// Format version the file was written at. A file older than the current
  /// version carries only some of the growth genes, and [`restore`] fills the
  /// rest with the founder values.
  pub version: u32,
  pub max_organisms: usize,
  pub max_limbs: usize,
  pub param_count: usize,
  pub latent_len: usize,
  /// What the writing run's step counter stood at. Provenance only.
  pub step_count: u32,
  /// The writing run's resolved seed. Provenance only.
  pub seed: u64,
  pub organisms: OrganismHost,
  pub limbs: LimbHost,
  pub brain: Vec<f32>,
  pub latents: Vec<f32>,
  pub anchors: Vec<Vec2>,
}

impl PopSnapshot {
  /// Organism slots holding a germinated plant — what a load brings back.
  pub fn plants(&self) -> Vec<usize> {
    (0..self.max_organisms)
      .filter(|o| self.organisms.alive[*o] == 1 && self.organisms.stage[*o] == STAGE_PLANT)
      .collect()
  }
}

pub fn write<P: AsRef<Path>>(
  path: P,
  pop: &Population,
  bodies: &BodyState,
  step_count: u32,
  seed: u64,
) -> Result<(), PopDumpError> {
  let mut out = BufWriter::new(File::create(path)?);
  out.write_all(MAGIC)?;
  out.write_all(&VERSION.to_le_bytes())?;
  out.write_all(&(pop.max_organisms as u32).to_le_bytes())?;
  out.write_all(&(pop.max_limbs as u32).to_le_bytes())?;
  out.write_all(&(pop.shape.param_count() as u32).to_le_bytes())?;
  out.write_all(&(pop.shape.latent_state_len() as u32).to_le_bytes())?;
  out.write_all(&step_count.to_le_bytes())?;
  out.write_all(&seed.to_le_bytes())?;

  pop.organisms.write_fields(&mut out)?;
  pop.limbs.write_fields(&mut out)?;
  out.write_all(bytemuck::cast_slice(&pop.brain))?;
  out.write_all(bytemuck::cast_slice(&pop.latents))?;
  let anchors: Vec<f32> = bodies.anchors.iter().flat_map(|p| [p.x, p.y]).collect();
  out.write_all(bytemuck::cast_slice(&anchors))?;
  out.flush()?;
  Ok(())
}

pub fn read<P: AsRef<Path>>(path: P) -> Result<PopSnapshot, PopDumpError> {
  let mut input = BufReader::new(File::open(path)?);

  let mut magic = [0u8; 8];
  input.read_exact(&mut magic)?;
  if &magic != MAGIC {
    return Err(PopDumpError::BadMagic);
  }
  let version = read_u32(&mut input)?;
  if version == 0 || version > VERSION {
    return Err(PopDumpError::BadVersion(version));
  }

  let max_organisms = read_u32(&mut input)? as usize;
  let max_limbs = read_u32(&mut input)? as usize;
  let param_count = read_u32(&mut input)? as usize;
  let latent_len = read_u32(&mut input)? as usize;
  let step_count = read_u32(&mut input)?;
  let mut seed_bytes = [0u8; 8];
  input.read_exact(&mut seed_bytes)?;

  // Length first, as `dump.rs` does: a corrupt count would otherwise ask for
  // gigabytes and abort instead of failing cleanly.
  // In checked u64 arithmetic: the header words are untrusted, and a
  // corrupt pair near u32::MAX would otherwise wrap the product into a
  // small number that passes the check.
  let actual = input.get_ref().metadata()?.len();
  let o = max_organisms as u64;
  let expected = (|| {
    let records = o.checked_mul(max_limbs as u64)?;
    HEADER_BYTES
      .checked_add(o.checked_mul(OrganismHost::RAW_BYTES as u64)?)?
      .checked_add(records.checked_mul(LimbHost::raw_bytes_prefix(limb_fields(version)) as u64)?)?
      .checked_add(o.checked_mul(param_count as u64)?.checked_mul(4)?)?
      .checked_add(o.checked_mul(latent_len as u64)?.checked_mul(4)?)?
      .checked_add(o.checked_mul(8)?)
  })();
  match expected {
    Some(expected) if actual >= expected => {}
    Some(expected) => return Err(PopDumpError::Truncated { expected, actual }),
    None => {
      return Err(PopDumpError::Truncated {
        expected: u64::MAX,
        actual,
      });
    }
  }
  let records = max_organisms * max_limbs;

  let organisms = OrganismHost::read_fields(&mut input, max_organisms)?;
  let limbs = LimbHost::read_fields_prefix(&mut input, records, limb_fields(version))?;
  let brain = read_f32(&mut input, max_organisms * param_count)?;
  let latents = read_f32(&mut input, max_organisms * latent_len)?;
  let flat = read_f32(&mut input, max_organisms * 2)?;
  let anchors = flat
    .as_chunks::<2>()
    .0
    .iter()
    .map(|p| Vec2::from(*p))
    .collect();

  Ok(PopSnapshot {
    version,
    max_organisms,
    max_limbs,
    param_count,
    latent_len,
    step_count,
    seed: u64::from_le_bytes(seed_bytes),
    organisms,
    limbs,
    brain,
    latents,
    anchors,
  })
}

/// Install a snapshot into a fresh world and re-grow its bodies.
///
/// Returns how many plants came back. The population's shape has to match the
/// run's own — a brain row is meaningless read at another width — and the
/// world has to be empty, because a body already standing in it holds particle
/// slots this would never give back.
pub fn restore<R: Runtime>(
  sim: &mut Sim<R>,
  snapshot: &PopSnapshot,
) -> Result<usize, PopDumpError> {
  {
    let pop = sim.population();
    for (what, found, expected) in [
      ("organism slots", snapshot.max_organisms, pop.max_organisms),
      (
        "limb records per organism",
        snapshot.max_limbs,
        pop.max_limbs,
      ),
      (
        "brain parameters",
        snapshot.param_count,
        pop.shape.param_count(),
      ),
      (
        "latent state floats",
        snapshot.latent_len,
        pop.shape.latent_state_len(),
      ),
    ] {
      if found != expected {
        return Err(PopDumpError::ShapeMismatch {
          what,
          found,
          expected,
        });
      }
    }
  }
  let alive = sim.organism_count();
  if alive > 0 {
    return Err(PopDumpError::NotEmpty { alive });
  }

  let plants = snapshot.plants();
  let params = *sim.params();
  {
    let access = sim.body_access();
    access.pop.organisms = snapshot.organisms.clone();
    access.pop.limbs = snapshot.limbs.clone();
    // A file older than a growth gene predates it, and its records came back
    // at the SoA's zeroes. A zero rate is a limb that can never grow and a
    // zero thickness is one with no volume at all, so the founder values are
    // what an old population comes back on.
    if snapshot.version < 2 {
      access.pop.limbs.growth_bias.fill(0.0);
      access.pop.limbs.growth_rate.fill(params.growth_rate);
    }
    if snapshot.version < 3 {
      // Present records only, as mutation perturbs present records only: an
      // absent record's thickness is the genome of nothing, and a structural
      // add writes its own.
      for i in 0..access.pop.limbs.len() {
        let part = access.pop.limbs.part_type[i];
        if part.is_present() {
          access.pop.limbs.target_thickness[i] =
            crate::genome::seed::founder_thickness(&params, part);
        }
      }
    }
    access.pop.brain = snapshot.brain.clone();
    access.pop.latents = snapshot.latents.clone();
    access.bodies.anchors = snapshot.anchors.clone();

    // A seed is world state, not population state: it was a particle falling
    // through a world this run does not have. Its slot goes back free.
    for o in 0..snapshot.max_organisms {
      if access.pop.organisms.alive[o] == 1 && access.pop.organisms.stage[o] == STAGE_PLANT {
        continue;
      }
      access.pop.organisms.alive[o] = 0;
      access.pop.organisms.energy[o] = 0.0;
      access.pop.organisms.stage[o] = STAGE_SEED;
      access.bodies.anchors[o] = Vec2::ZERO;
      access.bodies.release(o);
    }
    access.bodies.recompute_high_water();
    access.pop.upload(access.client);
    access.bodies.upload(access.client);
  }

  bodies::grow_bodies(sim, &plants);
  Ok(plants.len())
}

fn read_u32<R: Read>(input: &mut R) -> std::io::Result<u32> {
  let mut buf = [0u8; 4];
  input.read_exact(&mut buf)?;
  Ok(u32::from_le_bytes(buf))
}

fn read_f32<R: Read>(input: &mut R, n: usize) -> std::io::Result<Vec<f32>> {
  let mut out = vec![0.0f32; n];
  input.read_exact(bytemuck::cast_slice_mut(&mut out))?;
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SimParams;
  use crate::bodies::{NO_PARTICLE, spawn_founders};
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  fn params() -> SimParams {
    SimParams {
      world_width: 4.0,
      world_height: 3.0,
      smoothing_radius: 0.5,
      soil_cell_size: 0.25,
      particles_per_cell: 1,
      terrain_mode: 0,
      max_organisms: 4,
      max_limbs: 4,
      max_particles_per_limb: 3,
      life_interval: 100_000,
      ..SimParams::default()
    }
  }

  fn world() -> Sim<CpuRuntime> {
    Sim::new(CpuRuntime::client(&CpuDevice), params(), 42, None)
  }

  fn temp_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
      "alife-pop-{name}-{}-{:?}.bin",
      std::process::id(),
      std::thread::current().id()
    ));
    path
  }

  /// The whole point of the format: a population written out and read back
  /// into a fresh world is the same population, genome for genome.
  #[test]
  fn a_three_founder_population_round_trips() {
    let mut source = world();
    assert_eq!(spawn_founders(&mut source, 3), 3);
    // Something in every runtime field, so a round trip that silently drops
    // one is visible.
    {
      let access = source.body_access();
      for (slot, energy) in [(0usize, 1.25f32), (1, -0.5), (2, 3.0)] {
        access.pop.organisms.energy[slot] = energy;
        access.pop.organisms.birth_step[slot] = 10 * slot as u32;
      }
      access.pop.organisms.lineage_id[2] = 99;
      access.pop.organisms.generation[2] = 7;
      access.pop.upload_organisms(access.client);
    }

    let path = temp_path("round-trip");
    write(
      &path,
      source.population(),
      source.bodies(),
      source.step_count(),
      source.seed(),
    )
    .unwrap();
    let snapshot = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(snapshot.max_organisms, 4);
    assert_eq!(snapshot.max_limbs, 4);
    assert_eq!(snapshot.seed, source.seed());
    assert_eq!(snapshot.plants(), [0, 1, 2]);

    let mut target = world();
    assert_eq!(restore(&mut target, &snapshot).unwrap(), 3);

    // Genomes, lineage and energy, bit for bit.
    assert_eq!(target.population().limbs, source.population().limbs);
    assert_eq!(target.population().organisms, source.population().organisms);
    assert_eq!(target.population().brain, source.population().brain);
    assert_eq!(target.population().latents, source.population().latents);
    assert_eq!(target.bodies().anchors, source.bodies().anchors);
    assert_eq!(target.organism_count(), 3);

    // And the bodies are back: every limb the source had grown, the target
    // grew too, particle for particle in the same slots.
    assert_eq!(
      target.bodies().limb_particles,
      source.bodies().limb_particles
    );
    let (before, after) = (source.read_particles(), target.read_particles());
    for o in 0..3usize {
      for limb in 0..4usize {
        for i in 0..3usize {
          let id = target.bodies().particle(o, limb, i);
          if id == NO_PARTICLE {
            continue;
          }
          assert_eq!(after.ppos[id as usize], before.ppos[id as usize]);
          assert_eq!(after.part_type[id as usize], before.part_type[id as usize]);
        }
      }
    }
  }

  /// Seeds in flight are world state and do not come back; their slots do.
  #[test]
  fn a_load_drops_the_seeds_in_flight() {
    let mut source = world();
    assert_eq!(spawn_founders(&mut source, 3), 3);
    {
      let access = source.body_access();
      access.pop.organisms.alive[3] = 1;
      access.pop.organisms.stage[3] = STAGE_SEED;
      access.pop.organisms.lineage_id[3] = 0;
      access.pop.upload_organisms(access.client);
    }

    let path = temp_path("seeds");
    write(
      &path,
      source.population(),
      source.bodies(),
      source.step_count(),
      source.seed(),
    )
    .unwrap();
    let snapshot = read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(snapshot.organisms.alive[3], 1, "the file keeps the seed");

    let mut target = world();
    assert_eq!(restore(&mut target, &snapshot).unwrap(), 3);
    assert_eq!(target.population().organisms.alive[3], 0);
    assert_eq!(target.organism_count(), 3);
  }

  /// Write a population and age the file down to `version`: rewrite the
  /// version word and drop the growth-gene arrays that version did not have,
  /// which are the tail of the limb block.
  fn aged_file(source: &Sim<CpuRuntime>, version: u32, tag: &str) -> PopSnapshot {
    let path = temp_path(tag);
    write(
      &path,
      source.population(),
      source.bodies(),
      source.step_count(),
      source.seed(),
    )
    .unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&version.to_le_bytes());
    let records = source.population().max_organisms * source.population().max_limbs;
    let organisms = source.population().max_organisms * OrganismHost::RAW_BYTES;
    // Four one-byte fields, `grow_angle` and `identity`, then one float
    // array per growth gene — the ones the older version lacks are the tail
    // of the limb block, not all of it.
    const HEAD_FIELDS: usize = 6;
    let head = HEADER_BYTES as usize
      + organisms
      + records * (4 + 4 + 4 * crate::genome::shape::IDENTITY_DIM);
    let kept = limb_fields(version) - HEAD_FIELDS;
    let dropped = LimbHost::FIELD_COUNT - limb_fields(version);
    let start = head + kept * records * 4;
    bytes.drain(start..start + dropped * records * 4);
    std::fs::write(&path, &bytes).unwrap();

    let snapshot = read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    snapshot
  }

  /// A file written before the growth genes existed still loads, and its
  /// organisms come back able to grow rather than frozen at a zero rate.
  #[test]
  fn a_version_one_file_loads_with_the_founder_growth_genes() {
    let mut source = world();
    assert_eq!(spawn_founders(&mut source, 2), 2);
    let snapshot = aged_file(&source, 1, "v1");
    assert_eq!(snapshot.version, 1);
    assert_eq!(snapshot.limbs.identity, source.population().limbs.identity);
    assert_eq!(
      snapshot.limbs.grow_angle,
      source.population().limbs.grow_angle
    );
    assert!(snapshot.limbs.growth_rate.iter().all(|r| *r == 0.0));
    assert!(snapshot.limbs.target_thickness.iter().all(|t| *t == 0.0));

    let mut target = world();
    assert_eq!(restore(&mut target, &snapshot).unwrap(), 2);
    let rate = target.params().growth_rate;
    assert!(
      target
        .population()
        .limbs
        .growth_rate
        .iter()
        .all(|r| *r == rate)
    );
    assert!(
      target
        .population()
        .limbs
        .growth_bias
        .iter()
        .all(|b| *b == 0.0)
    );
    // And the field version 2 added is filled the same way as from a
    // version-2 file: the founder thickness of each present record's type.
    let limbs = &target.population().limbs;
    let params = target.params();
    for i in 0..limbs.part_type.len() {
      if limbs.part_type[i].is_present() {
        assert_eq!(
          limbs.target_thickness[i],
          crate::genome::seed::founder_thickness(params, limbs.part_type[i]),
        );
      }
    }
  }

  /// A version-2 file predates `target_thickness` alone: its growth bias and
  /// rate come back as written, and every record takes its part type's
  /// founder thickness.
  #[test]
  fn a_version_two_file_loads_with_the_founder_thicknesses() {
    let mut source = world();
    assert_eq!(spawn_founders(&mut source, 2), 2);
    let snapshot = aged_file(&source, 2, "v2");
    assert_eq!(snapshot.version, 2);
    assert_eq!(
      snapshot.limbs.growth_rate,
      source.population().limbs.growth_rate
    );
    assert!(snapshot.limbs.target_thickness.iter().all(|t| *t == 0.0));

    let mut target = world();
    assert_eq!(restore(&mut target, &snapshot).unwrap(), 2);
    let params = *target.params();
    let limbs = &target.population().limbs;
    for i in 0..limbs.len() {
      if limbs.part_type[i].is_present() {
        assert_eq!(
          limbs.target_thickness[i],
          crate::genome::seed::founder_thickness(&params, limbs.part_type[i]),
          "record {i}"
        );
      }
    }
    // And the founders it came from were written with exactly those values,
    // so an aged file is indistinguishable from a current one here.
    assert_eq!(
      limbs.target_thickness,
      source.population().limbs.target_thickness
    );
  }

  #[test]
  fn a_population_of_another_shape_is_refused() {
    let mut source = world();
    spawn_founders(&mut source, 1);
    let path = temp_path("shape");
    write(&path, source.population(), source.bodies(), 0, 1).unwrap();
    let snapshot = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let wider = SimParams {
      max_organisms: 8,
      ..params()
    };
    let mut target = Sim::new(CpuRuntime::client(&CpuDevice), wider, 42, None);
    match restore(&mut target, &snapshot) {
      Err(PopDumpError::ShapeMismatch { what, .. }) => assert_eq!(what, "organism slots"),
      other => panic!("expected a shape mismatch, got {other:?}"),
    }
  }

  #[test]
  fn a_world_that_already_has_organisms_is_refused() {
    let mut source = world();
    spawn_founders(&mut source, 1);
    let path = temp_path("occupied");
    write(&path, source.population(), source.bodies(), 0, 1).unwrap();
    let snapshot = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    let mut target = world();
    spawn_founders(&mut target, 2);
    match restore(&mut target, &snapshot) {
      Err(PopDumpError::NotEmpty { alive }) => assert_eq!(alive, 2),
      other => panic!("expected NotEmpty, got {other:?}"),
    }
  }
}
