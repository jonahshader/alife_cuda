//! The simulation: state, CubeCL kernels and their plain-Rust references, and
//! the step function. No windowing, no egui — see the `alife` binary for those.
//!
//! Ported from the C++/CUDA tree's `src/systems/particle_fluid2.cu` (the
//! soil-coupled path) and `src/systems/soil.cu`, which stay as the reference
//! until this port reaches parity.

pub mod dump;
pub mod genome;
pub mod kernels;
pub mod params;
pub mod particles;
pub mod rng;
pub mod runtime;
pub mod sim;
pub mod soa;
pub mod soil;
pub mod timing;
pub mod wgpu_backend;
pub mod world;

pub use genome::BrainShape;
pub use params::{SimParams, SimParamsCli};
pub use particles::{ParticleKind, SphDevice, SphHost};
