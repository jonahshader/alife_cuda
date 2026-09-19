//! The `--dump` / `--load` binary format, shared with the C++ binary.
//!
//! Layout (little-endian), identical to the writer documented above
//! `write_particle_dump` in the C++ tree's `src/main.cu`:
//!
//! ```text
//!   char[8]  magic "ALIFEDMP"
//!   uint32   version (2)
//!   uint32   particle count N
//!   uint32   step count
//!   uint64   resolved seed
//!   then one contiguous array per field, in `SphHost` declaration order:
//!   pos[N], ppos[N], vel[N], acc[N] (float2 = two floats each),
//!   mass[N], density[N], near_density[N] (float),
//!   sym_break[N], state[N] (uint8), evap_prob[N] (float),
//!   organism[N] (uint32), limb[N], index_in_limb[N], part_type[N] (uint8)
//! ```
//!
//! Field order and sizes follow the SoA declaration, so adding a field there
//! extends the dump automatically. Version 1 is the C++ layout — the ten
//! arrays up to `evap_prob`, [`LEGACY_RAW_BYTES`] per particle — and still
//! loads: the organism fields keep their declared defaults.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::particles::SphHost;

pub const MAGIC: &[u8; 8] = b"ALIFEDMP";
pub const VERSION: u32 = 2;

/// The version the C++ binary writes, and that `resources/parity/*.bin` are.
pub const LEGACY_VERSION: u32 = 1;

/// Arrays a version-1 dump holds: the C++ SoA, `pos` through `evap_prob`.
pub const LEGACY_FIELDS: usize = 10;

/// Bytes per particle in a version-1 dump: 4 float2 + 4 float + 2 uint8.
pub const LEGACY_RAW_BYTES: usize = SphHost::raw_bytes_prefix(LEGACY_FIELDS);

#[derive(Debug, thiserror::Error)]
pub enum DumpError {
  #[error("io error: {0}")]
  Io(#[from] std::io::Error),
  #[error("not an alife dump (bad magic)")]
  BadMagic,
  #[error("unsupported dump version {0} (this build writes {VERSION} and reads {LEGACY_VERSION})")]
  BadVersion(u32),
  #[error("dump is {actual} bytes but its header implies {expected} (truncated or corrupt)")]
  Truncated { expected: u64, actual: u64 },
}

/// Bytes before the first field array.
const HEADER_BYTES: u64 = 8 + 4 + 4 + 4 + 8;

/// Everything a dump carries besides the particle arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DumpHeader {
  pub count: u32,
  pub step_count: u32,
  pub seed: u64,
}

pub fn write<P: AsRef<Path>>(
  path: P,
  particles: &SphHost,
  step_count: u32,
  seed: u64,
) -> Result<(), DumpError> {
  let mut out = BufWriter::new(File::create(path)?);
  out.write_all(MAGIC)?;
  out.write_all(&VERSION.to_le_bytes())?;
  out.write_all(&(particles.len() as u32).to_le_bytes())?;
  out.write_all(&step_count.to_le_bytes())?;
  out.write_all(&seed.to_le_bytes())?;
  particles.write_fields(&mut out)?;
  out.flush()?;
  Ok(())
}

pub fn read<P: AsRef<Path>>(path: P) -> Result<(DumpHeader, SphHost), DumpError> {
  let mut input = BufReader::new(File::open(path)?);

  let mut magic = [0u8; 8];
  input.read_exact(&mut magic)?;
  if &magic != MAGIC {
    return Err(DumpError::BadMagic);
  }

  let version = read_u32(&mut input)?;
  let fields = match version {
    VERSION => SphHost::FIELD_COUNT,
    LEGACY_VERSION => LEGACY_FIELDS,
    other => return Err(DumpError::BadVersion(other)),
  };

  let count = read_u32(&mut input)?;
  let step_count = read_u32(&mut input)?;
  let mut seed_bytes = [0u8; 8];
  input.read_exact(&mut seed_bytes)?;

  // Check the file can hold what the header promises before `read_fields`
  // allocates `count` elements per field: a corrupt count would otherwise
  // request gigabytes and abort instead of failing cleanly.
  let expected = HEADER_BYTES + count as u64 * SphHost::raw_bytes_prefix(fields) as u64;
  let actual = input.get_ref().metadata()?.len();
  if actual < expected {
    return Err(DumpError::Truncated { expected, actual });
  }

  let particles = SphHost::read_fields_prefix(&mut input, count as usize, fields)?;
  Ok((
    DumpHeader {
      count,
      step_count,
      seed: u64::from_le_bytes(seed_bytes),
    },
    particles,
  ))
}

fn read_u32<R: Read>(input: &mut R) -> std::io::Result<u32> {
  let mut buf = [0u8; 4];
  input.read_exact(&mut buf)?;
  Ok(u32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::particles::ParticleKind;
  use glam::Vec2;

  #[test]
  fn round_trip() {
    let mut particles = SphHost::new(3);
    particles.pos[1] = Vec2::new(1.5, -2.5);
    particles.state[2] = ParticleKind::Vapor;
    particles.sym_break[0] = 200;
    particles.evap_prob[2] = 0.25;
    particles.state[0] = ParticleKind::Body;
    particles.organism[0] = 5;
    particles.limb[0] = 2;
    particles.index_in_limb[0] = 3;
    particles.part_type[0] = crate::genome::PartType::Leaf;

    let path = std::env::temp_dir().join("alife_dump_round_trip.bin");
    write(&path, &particles, 7, 42).unwrap();
    let (header, loaded) = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(header.count, 3);
    assert_eq!(header.step_count, 7);
    assert_eq!(header.seed, 42);
    assert_eq!(loaded, particles);
  }

  #[test]
  fn raw_bytes_matches_the_cpp_layout() {
    // 4 float2 + 3 float + 2 uint8 + 1 float, as `src/main.cu` documents
    assert_eq!(LEGACY_RAW_BYTES, 50);
    // plus the version-2 tail: one uint32 and three uint8
    assert_eq!(SphHost::RAW_BYTES, 50 + 4 + 3);
    assert_eq!(SphHost::FIELD_COUNT, LEGACY_FIELDS + 4);
  }

  /// A version-1 dump — every `resources/parity/*.bin`, and anything the C++
  /// binary writes — still loads, with the organism fields at their defaults.
  #[test]
  fn a_version_1_dump_loads_with_the_new_fields_defaulted() {
    let mut particles = SphHost::new(2);
    particles.pos[1] = Vec2::new(3.0, 4.0);
    particles.evap_prob[1] = 0.5;

    let path = std::env::temp_dir().join("alife_dump_v1.bin");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&LEGACY_VERSION.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&42u64.to_le_bytes());
    let mut fields = Vec::new();
    particles.write_fields(&mut fields).unwrap();
    bytes.extend_from_slice(&fields[..2 * LEGACY_RAW_BYTES]);
    std::fs::write(&path, &bytes).unwrap();

    let (header, loaded) = read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(header.count, 2);
    assert_eq!(header.step_count, 9);
    assert_eq!(loaded, particles);
  }

  #[test]
  fn a_truncated_dump_fails_before_allocating() {
    let particles = SphHost::new(4);
    let path = std::env::temp_dir().join("alife_dump_truncated.bin");
    write(&path, &particles, 0, 1).unwrap();
    // Rewrite the count as a huge number while leaving the file short.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let result = read(&path);
    std::fs::remove_file(&path).ok();
    match result {
      Err(DumpError::Truncated { .. }) => {}
      Err(other) => panic!("expected Truncated, got {other}"),
      Ok(_) => panic!("truncated dump must fail"),
    }
  }
}
