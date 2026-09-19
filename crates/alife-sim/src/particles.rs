//! The SPH particle state.
//!
//! The field list mirrors `FOR_SPH` in the C++ tree's
//! `src/systems/particle_fluid2.cuh`, in the same order, so a dump written by
//! either binary loads into the other.

use glam::Vec2;

use crate::define_soa;
use crate::genome::PartType;
use crate::soa::SoaField;

/// What a particle currently is. The C++ carries this as a raw `uint8_t`
/// `state` field tested with `!= 0` / `== 1`; the dump keeps the byte.
///
/// 0 and 1 are the C++'s own codes and do not move. The organism codes are
/// appended: `Body` is a particle belonging to a limb, `Free` an unallocated
/// slot that every kernel skips.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ParticleKind {
  #[default]
  Liquid = 0,
  Vapor = 1,
  /// An SPH boundary particle owned by a limb: in the grid, carries mass,
  /// pushed by the same forces as liquid, never evaporates.
  Body = 2,
  /// An unallocated slot, parked at [`PARKED_POS`] outside the world. Not in
  /// the grid, not integrated, not drawn.
  Free = 3,
}

impl ParticleKind {
  pub fn from_byte(b: u8) -> Self {
    // The C++ only ever stores 0 or 1; anything else it could produce is
    // "not liquid", and every kernel that acts on vapor tests `state == 1`
    // specifically. 2 and 3 are this tree's own codes.
    match b {
      0 => ParticleKind::Liquid,
      2 => ParticleKind::Body,
      3 => ParticleKind::Free,
      _ => ParticleKind::Vapor,
    }
  }

  /// Whether the fluid kernels see this particle at all: it is in the
  /// neighbour grid, it has a density, and the integrator moves it.
  pub fn in_fluid(self) -> bool {
    matches!(self, ParticleKind::Liquid | ParticleKind::Body)
  }
}

/// Where a [`ParticleKind::Free`] slot is parked: outside the world, so that
/// even if a kernel ever did read one its position is obviously not a place.
pub const PARKED_POS: Vec2 = Vec2::new(-1.0, -1.0);

/// `organism` of a particle that belongs to no organism.
pub const NO_ORGANISM: u32 = u32::MAX;

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
        /// Organism slot this particle belongs to, or [`NO_ORGANISM`]. The
        /// four fields below this line are the dump's version-2 tail; a
        /// version-1 dump has no bytes for them and they keep these defaults.
        organism: u32 = NO_ORGANISM,
        /// Limb record within the organism.
        limb: u8,
        /// Position along the limb's particle chain.
        index_in_limb: u8,
        /// The owning limb's part type, copied here so the renderer and the
        /// energy pass act per particle without a genome lookup.
        part_type: PartType,
    }
}

impl SphHost {
  /// How many particles are of one kind.
  pub fn count_of(&self, kind: ParticleKind) -> usize {
    self.state.iter().filter(|k| **k == kind).count()
  }

  /// Number of particles that are currently liquid.
  pub fn liquid_count(&self) -> usize {
    self.count_of(ParticleKind::Liquid)
  }

  /// Number of particles that are currently vapor.
  pub fn vapor_count(&self) -> usize {
    self.count_of(ParticleKind::Vapor)
  }

  /// Particles that are not organism body slots: the fluid, whatever state it
  /// is in. This is the count a dump's fluid capacity is taken from.
  pub fn fluid_count(&self) -> usize {
    self
      .state
      .iter()
      .filter(|k| !matches!(k, ParticleKind::Body | ParticleKind::Free))
      .count()
  }

  /// Append `n` unallocated body slots, parked outside the world.
  pub fn push_free_slots(&mut self, n: usize) {
    let mut free = Self::new(n);
    free.state.fill(ParticleKind::Free);
    free.pos.fill(PARKED_POS);
    free.ppos.fill(PARKED_POS);
    self.append(&mut free);
  }

  /// Everything that is not a body slot, in order — what `--load` keeps of a
  /// dump before the body capacity is appended on top. The organisms
  /// themselves are not in a dump, so their particles cannot be resumed.
  pub fn fluid_only(&self) -> Self {
    let mut out = Self::new(0);
    for i in 0..self.len() {
      if !matches!(self.state[i], ParticleKind::Body | ParticleKind::Free) {
        out.push_from(self, i);
      }
    }
    out
  }
}
