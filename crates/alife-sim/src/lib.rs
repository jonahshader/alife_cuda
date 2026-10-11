//! The simulation: state, CubeCL kernels and their plain-Rust references, and
//! the step function. No windowing, no egui — see the `alife` binary for those.
//!
//! The fluid and soil are a port of the C++/CUDA tree's
//! `src/systems/particle_fluid2.cu` (the soil-coupled path) and
//! `src/systems/soil.cu`, checked against that binary until the tree was
//! deleted; it survives at the git tag `cpp-reference`.

pub mod bodies;
pub mod brain;
pub mod dump;
pub mod genome;
pub mod kernels;
pub mod life;
pub mod metrics;
pub mod params;
pub mod particles;
pub mod popdump;
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
