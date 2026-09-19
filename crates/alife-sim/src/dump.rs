//! The `--dump` / `--load` binary format, shared with the C++ binary.
//!
//! Layout (little-endian), identical to the writer documented above
//! `write_particle_dump` in the C++ tree's `src/main.cu`:
//!
//! ```text
//!   char[8]  magic "ALIFEDMP"
//!   uint32   version (1)
//!   uint32   particle count N
//!   uint32   step count
//!   uint64   resolved seed
//!   then one contiguous array per field, in `SphHost` declaration order:
//!   pos[N], ppos[N], vel[N], acc[N] (float2 = two floats each),
//!   mass[N], density[N], near_density[N] (float),
//!   sym_break[N], state[N] (uint8), evap_prob[N] (float)
//! ```
//!
//! Field order and sizes follow the SoA declaration, so adding a field there
//! extends the dump automatically — in both trees, as long as both are edited.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::particles::SphHost;

pub const MAGIC: &[u8; 8] = b"ALIFEDMP";
pub const VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum DumpError {
  #[error("io error: {0}")]
  Io(#[from] std::io::Error),
  #[error("not an alife dump (bad magic)")]
  BadMagic,
  #[error("unsupported dump version {0} (this build writes {VERSION})")]
  BadVersion(u32),
}

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
  if version != VERSION {
    return Err(DumpError::BadVersion(version));
  }

  let count = read_u32(&mut input)?;
  let step_count = read_u32(&mut input)?;
  let mut seed_bytes = [0u8; 8];
  input.read_exact(&mut seed_bytes)?;

  let particles = SphHost::read_fields(&mut input, count as usize)?;
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

    let path = std::env::temp_dir().join("alife_dump_round_trip.bin");
    write(&path, &particles, 7, 42).unwrap();
    let (header, loaded) = read(&path).unwrap();
    std::fs::remove_file(&path).ok();

    assert_eq!(header.count, 3);
    assert_eq!(header.step_count, 7);
    assert_eq!(header.seed, 42);
    assert_eq!(loaded, particles);
  }
}
