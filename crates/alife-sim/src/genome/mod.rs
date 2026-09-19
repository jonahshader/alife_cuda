//! The genome: an organism's discrete body program and its continuous brain.
//!
//! Built from the *Genome*, *Evolution operators* and *Implementation layout*
//! sections of `docs/organism.md`. The two sections live in different places
//! for the same reason they are described separately there: the discrete
//! section is a short list of byte-wide limb records, and the continuous
//! section is one row of a population-wide tensor whose shape never changes.
//!
//! Nothing here is wired into [`crate::sim::Sim::step`] yet; the life-cycle
//! chunk is what calls the mutation kernel and the slot allocator.

pub mod mutate;
pub mod population;
pub mod seed;
pub mod shape;
pub mod slots;
pub mod species;

pub use population::{
  Genome, LimbDevice, LimbHost, LimbRecord, OrganismDevice, OrganismHost, PartType, Population,
  PopulationDevice, Weights,
};
pub use shape::{BrainShape, FEATURE_DIM, IDENTITY_DIM, InitKind, N_TYPES, SENSOR_DIM, SliceSpec};
pub use species::species_distance;
