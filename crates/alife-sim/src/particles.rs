//! The SPH particle state.
//!
//! The field list mirrors `FOR_SPH` in the C++ tree's
//! `src/systems/particle_fluid2.cuh`, in the same order, so a dump written by
//! either binary loads into the other.

use glam::Vec2;

use crate::define_soa;
use crate::soa::SoaField;

/// What a particle currently is. The C++ carries this as a raw `uint8_t`
/// `state` field tested with `!= 0` / `== 1`; the dump keeps the byte.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ParticleKind {
  #[default]
  Liquid = 0,
  Vapor = 1,
}

impl ParticleKind {
  pub fn from_byte(b: u8) -> Self {
    // The C++ only ever stores 0 or 1; anything else is "not liquid", and
    // every kernel that acts on vapor tests `state == 1` specifically.
    match b {
      0 => ParticleKind::Liquid,
      _ => ParticleKind::Vapor,
    }
  }
}

impl SoaField for ParticleKind {
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
    /// Particle state, one array per field.
    SphHost / SphDevice {
        /// Predicted position, one `dt_predict` ahead — what every neighbour
        /// kernel reads.
        pos: Vec2,
        /// Position at the end of the last step; `move_particles` integrates
        /// from this, not from `pos`.
        ppos: Vec2,
        vel: Vec2,
        /// Carried for dump compatibility with the C++ SoA; no kernel reads it.
        acc: Vec2,
        mass: f32,
        density: f32,
        near_density: f32,
        /// Per-particle random direction seed. Unused by the fluid kernels but
        /// part of the SoA and the dump.
        sym_break: u8,
        state: ParticleKind,
        evap_prob: f32,
    }
}

impl SphHost {
  /// Number of particles that are currently liquid.
  pub fn liquid_count(&self) -> usize {
    self
      .state
      .iter()
      .filter(|k| **k == ParticleKind::Liquid)
      .count()
  }

  /// Number of particles that are currently vapor.
  pub fn vapor_count(&self) -> usize {
    self.len() - self.liquid_count()
  }
}
