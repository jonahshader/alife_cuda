//! The CubeCL kernels and, next to each one, its plain-Rust reference.
//!
//! The kernel bodies follow the C++ in `src/systems/particle_fluid2.cu`
//! line for line — same constants, same neighbour-loop structure, same order
//! within a step. The host side around them does not.
//!
//! The references exist because CubeCL kernels are not debuggable: the CPU
//! runtime's LLVM JIT emits no symbols or line tables, so `gdb` never sees the
//! kernel (`docs/organism.md`, the spike outcome). Every kernel therefore has a
//! sequential Rust twin plus a test that runs both over one fixed input.

use cubecl::prelude::*;

pub mod accel;
pub mod constraints;
pub mod density;
pub mod evap;
pub mod grid;
pub mod limb_geometry;
pub mod motion;
pub mod scan;
pub mod soil_sample;
pub mod spawn;

#[cfg(test)]
pub mod test_support;

/// Units per cube. 256 matches the C++ launch configuration.
pub const CUBE_DIM: u32 = 256;

/// Threads used by the grid-build prefix scan; see `kernels::grid`.
pub const SCAN_THREADS: u32 = 256;

pub fn cube_count(work: usize) -> CubeCount {
  CubeCount::Static(work.div_ceil(CUBE_DIM as usize).max(1) as u32, 1, 1)
}

/// How many particle slots a per-particle kernel is actually launched over.
///
/// Not `cfg.num_particles`: body slots are claimed ascending from the start
/// of the capacity, so everything above the high-water mark is `Free` and
/// every kernel would terminate on it immediately. Launching over the live
/// prefix instead costs the three neighbour kernels 6-9% less at
/// `--founders 0` — the whole body capacity, 64% more units, would otherwise
/// be launched and thrown away every step.
///
/// The comptime bound inside each kernel stays `cfg.num_particles`, so this
/// changes no generated code: it only launches fewer cubes of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveParticles(pub usize);

impl LiveParticles {
  pub fn get(self) -> usize {
    self.0
  }
}

/// The comptime half of the kernel configuration; it belongs to the world's
/// geometry, which is what decides it.
pub use crate::world::Cfg;

/// One runtime float per entry of the params buffer.
///
/// A single declaration yields the slot index used inside the kernels and the
/// packing expression used on the host, so the two cannot drift.
macro_rules! kernel_params {
    ($( $doc:literal $name:ident = |$p:ident, $g:ident| $expr:expr; )*) => {
        kernel_params!(@consts 0u32; $( $name = |$p, $g| $expr; )*);

        /// Pack the runtime parameters into the buffer the kernels index.
        pub fn pack_params(p: &crate::SimParams, g: &crate::world::WorldGeometry) -> Vec<f32> {
            let mut out = Vec::new();
            $( { let $p = p; let $g = g; out.push($expr); } )*
            out
        }
    };
    (@consts $idx:expr; $name:ident = |$p:ident, $g:ident| $expr:expr; $($rest:tt)*) => {
        pub const $name: u32 = $idx;
        kernel_params!(@consts $idx + 1; $($rest)*);
    };
    (@consts $idx:expr;) => {
        /// Number of slots in the params buffer.
        pub const PARAM_COUNT: usize = ($idx) as usize;
    };
}

kernel_params! {
    "" P_DT = |p, _g| p.dt;
    "" P_DT_PREDICT = |p, _g| p.dt_predict;
    "" P_GRAVITY = |p, _g| p.gravity;
    "" P_COLLISION_DAMPING = |p, _g| p.collision_damping;
    "" P_SMOOTHING_RADIUS = |p, _g| p.smoothing_radius;
    "" P_TARGET_DENSITY = |p, _g| p.target_density;
    "" P_PRESSURE_MULT = |p, _g| p.pressure_mult;
    "" P_NEAR_PRESSURE_MULT = |p, _g| p.near_pressure_mult;
    "" P_VISCOSITY_STRENGTH = |p, _g| p.viscosity_strength;
    "" P_CAPILLARY_MULT = |p, _g| p.capillary_mult;
    "" P_EVAP_RATE = |p, _g| p.evap_rate;
    "" P_CONDENSE_RATE = |p, _g| p.condense_rate;
    "" P_VAPOR_BUOYANCY = |p, _g| p.vapor_buoyancy;
    "" P_VAPOR_DRIFT = |p, _g| p.vapor_drift;
    "" P_CONDENSE_ALT_POWER = |p, _g| p.condense_altitude_power;
    "" P_BOUNDS_X = |_p, g| g.bounds.x;
    "" P_BOUNDS_Y = |_p, g| g.bounds.y;
    "" P_CELL_SIZE = |_p, g| g.cell_size;
    "" P_SOIL_SIZE = |_p, g| g.soil_cell_size;
    "" P_LIMB_SEGMENT_LENGTH = |p, _g| p.limb_segment_length;
    "" P_JOINT_STIFFNESS = |p, _g| p.joint_stiffness;
    "" P_BEND_STIFFNESS = |p, _g| p.bend_stiffness;
    "" P_CONSTRAINT_ITERS = |p, _g| p.constraint_iterations as f32;
}

/// The particle SoA as kernel arguments. Field order matches `SphHost`.
///
/// `pos`, `ppos`, `vel` and `acc` are `float2` in the C++ SoA and stay
/// interleaved here, so element `i` lives at `2 * i` and `2 * i + 1`.
#[derive(CubeLaunch, CubeType)]
pub struct SphArgs {
  pub pos: Box<[f32]>,
  pub ppos: Box<[f32]>,
  pub vel: Box<[f32]>,
  pub mass: Box<[f32]>,
  pub density: Box<[f32]>,
  pub near_density: Box<[f32]>,
  pub state: Box<[u32]>,
  pub evap_prob: Box<[f32]>,
}

/// The particle SoA as the organism kernels see it: the fluid's fields plus
/// the four that say which limb a particle belongs to.
///
/// Deliberately a second struct rather than more fields on [`SphArgs`]. The
/// fluid kernels' generated code is keyed on their argument list, and the
/// fluid has to stay byte-for-byte what it was before organisms existed.
#[derive(CubeLaunch, CubeType)]
pub struct BodyArgs {
  pub pos: Box<[f32]>,
  pub ppos: Box<[f32]>,
  pub vel: Box<[f32]>,
  pub mass: Box<[f32]>,
  pub density: Box<[f32]>,
  pub near_density: Box<[f32]>,
  pub state: Box<[u32]>,
  pub evap_prob: Box<[f32]>,
  pub organism: Box<[u32]>,
  pub limb: Box<[u32]>,
  pub index_in_limb: Box<[u32]>,
  pub part_type: Box<[u32]>,
}

pub fn body_args<R: Runtime>(sph: &crate::particles::SphDevice) -> BodyArgsLaunch<R> {
  let n = sph.len();
  BodyArgsLaunch::new(
    whole(&sph.pos, n * 2),
    whole(&sph.ppos, n * 2),
    whole(&sph.vel, n * 2),
    whole(&sph.mass, n),
    whole(&sph.density, n),
    whole(&sph.near_density, n),
    whole(&sph.state, n),
    whole(&sph.evap_prob, n),
    whole(&sph.organism, n),
    whole(&sph.limb, n),
    whole(&sph.index_in_limb, n),
    whole(&sph.part_type, n),
  )
}

/// The neighbour grid as kernel arguments; see `kernels::grid` for the layout.
#[derive(CubeLaunch, CubeType)]
pub struct GridArgs {
  pub cell_counts: Box<[u32]>,
  pub cell_start: Box<[u32]>,
  pub sorted_ids: Box<[u32]>,
}

/// The soil grid as kernel arguments. Only the three mineral fractions are
/// read by the fluid kernels.
#[derive(CubeLaunch, CubeType)]
pub struct SoilArgs {
  pub sand_density: Box<[f32]>,
  pub silt_density: Box<[f32]>,
  pub clay_density: Box<[f32]>,
}

/// Bind a whole handle as a kernel buffer argument.
///
/// A CubeCL handle is a slice of a pooled buffer, and `from_raw_parts` is how
/// the offset travels with it.
pub fn whole<R: Runtime>(handle: &cubecl_runtime::server::Handle, len: usize) -> BufferArg<R> {
  unsafe { BufferArg::from_raw_parts(handle.clone(), len) }
}

/// Launch arguments are consumed by a launch, so every launch site builds its
/// own from the long-lived handles.
pub fn sph_args<R: Runtime>(sph: &crate::particles::SphDevice) -> SphArgsLaunch<R> {
  let n = sph.len();
  SphArgsLaunch::new(
    whole(&sph.pos, n * 2),
    whole(&sph.ppos, n * 2),
    whole(&sph.vel, n * 2),
    whole(&sph.mass, n),
    whole(&sph.density, n),
    whole(&sph.near_density, n),
    whole(&sph.state, n),
    whole(&sph.evap_prob, n),
  )
}

pub fn grid_args<R: Runtime>(grid: &grid::GridDevice, cfg: &Cfg) -> GridArgsLaunch<R> {
  let cells = cfg.num_cells as usize;
  let particles = cfg.num_particles as usize;
  GridArgsLaunch::new(
    whole(&grid.cell_counts, cells),
    whole(&grid.cell_start, cells),
    whole(&grid.sorted_ids, particles),
  )
}

pub fn soil_args<R: Runtime>(soil: &crate::soil::SoilDevice, cfg: &Cfg) -> SoilArgsLaunch<R> {
  let cells = (cfg.soil_w * cfg.soil_h) as usize;
  SoilArgsLaunch::new(
    whole(&soil.sand_density, cells),
    whole(&soil.silt_density, cells),
    whole(&soil.clay_density, cells),
  )
}

// --- Particle kinds as the kernels see them ---

/// [`crate::ParticleKind`] codes. A kernel reads `state` as a `u32`, and the
/// C++ tests it with bare literals (`state != 0`, `state == 1`); these name
/// the same numbers so the organism codes do not arrive as more literals.
pub const KIND_LIQUID: u32 = crate::ParticleKind::Liquid as u32;
pub const KIND_VAPOR: u32 = crate::ParticleKind::Vapor as u32;
pub const KIND_BODY: u32 = crate::ParticleKind::Body as u32;
pub const KIND_FREE: u32 = crate::ParticleKind::Free as u32;

/// Whether the fluid kernels see this particle at all: it goes in the
/// neighbour grid, it gets a density, and the integrator moves it. Liquid and
/// body particles do; vapor has its own integrator and a free slot has none.
#[cube]
pub fn in_fluid(state: u32) -> bool {
  state == KIND_LIQUID || state == KIND_BODY
}

// --- Shared device helpers, ported one-to-one from particle_fluid2.cu ---

#[cube]
pub fn clamp_i32(v: i32, lo: i32, hi: i32) -> i32 {
  let mut out = v;
  if out < lo {
    out = lo;
  }
  if out > hi {
    out = hi;
  }
  out
}

/// Given a particle's position, return the cell index it belongs to.
///
/// The columns tile the width exactly, so `pos.x == bounds.x` lands one column
/// past the last one; clamp rather than index out of the grid. Unlike the C++
/// `particle_to_cid`, this clamps y too: the C++ is not given the grid height,
/// so any `pos.y >= bounds.y` indexes one row past the grid.
#[cube]
pub fn particle_to_cid(pos_x: f32, pos_y: f32, cell_size: f32, #[comptime] cfg: Cfg) -> u32 {
  let grid_x = clamp_i32((pos_x / cell_size) as i32, 0, cfg.grid_w - 1);
  let grid_y = clamp_i32((pos_y / cell_size) as i32, 0, cfg.grid_h - 1);
  (grid_y * cfg.grid_w + grid_x) as u32
}

/// Host twin of [`particle_to_cid`].
pub fn particle_to_cid_ref(pos: glam::Vec2, cell_size: f32, cfg: &Cfg) -> u32 {
  let grid_x = ((pos.x / cell_size) as i32).clamp(0, cfg.grid_w - 1);
  let grid_y = ((pos.y / cell_size) as i32).clamp(0, cfg.grid_h - 1);
  (grid_y * cfg.grid_w + grid_x) as u32
}

#[cube]
pub fn density_kernel(radius: f32, dst: f32) -> f32 {
  let mut out = 0.0f32;
  if dst < radius {
    let normalization_factor_2d =
      6.0f32 / (core::f32::consts::PI * radius * radius * radius * radius);
    let value = radius - dst;
    out = normalization_factor_2d * value * value;
  }
  out
}

#[cube]
pub fn near_density_kernel(radius: f32, dst: f32) -> f32 {
  let mut out = 0.0f32;
  if dst < radius {
    let normalization_factor_2d =
      20.0f32 / (core::f32::consts::PI * radius * radius * radius * radius * radius);
    let value = radius - dst;
    out = normalization_factor_2d * value * value * value;
  }
  out
}

/// One component of the density kernel's gradient.
///
/// The C++ `density_kernel_gradient` builds the whole `float2` as
/// `-2 * norm * diff * value / dst`; this evaluates one component in that same
/// order, because the evaporation kernel sums components that nearly cancel
/// and a different association shows up there. The `dst > 1e-5` guard keeps a
/// coincident pair from producing a NaN direction.
#[cube]
pub fn density_kernel_gradient_component(radius: f32, diff: f32, dst: f32) -> f32 {
  let mut grad = 0.0f32;
  if dst < radius && dst > 1e-5f32 {
    let r2 = radius * radius;
    let normalization_factor_2d = 6.0f32 / (core::f32::consts::PI * r2 * r2);
    let value = radius - dst;
    grad = -2.0f32 * normalization_factor_2d * diff * value / dst;
  }
  grad
}

/// poly6: `(r^2 - d^2)^3 * 4/(pi r^8)`, so it only ever needs the squared
/// distance — the C++ takes `dst2` for the same reason.
#[cube]
pub fn viscosity_kernel(radius: f32, dst2: f32) -> f32 {
  let radius2 = radius * radius;
  let mut out = 0.0f32;
  if dst2 <= radius2 {
    let r4 = radius2 * radius2;
    let normalization_factor_2d = 4.0f32 / (core::f32::consts::PI * r4 * r4);
    let value = radius2 - dst2;
    out = normalization_factor_2d * value * value * value;
  }
  out
}

// --- Host twins of the smoothing kernels ---

pub fn density_kernel_ref(radius: f32, dst: f32) -> f32 {
  if dst >= radius {
    return 0.0;
  }
  let normalization_factor_2d = 6.0f32 / (std::f32::consts::PI * radius.powi(4));
  let value = radius - dst;
  normalization_factor_2d * value * value
}

pub fn near_density_kernel_ref(radius: f32, dst: f32) -> f32 {
  if dst >= radius {
    return 0.0;
  }
  let normalization_factor_2d = 20.0f32 / (std::f32::consts::PI * radius.powi(5));
  let value = radius - dst;
  normalization_factor_2d * value * value * value
}

pub fn density_kernel_gradient_component_ref(radius: f32, diff: f32, dst: f32) -> f32 {
  if dst < radius && dst > 1e-5 {
    let normalization_factor_2d = 6.0 / (std::f32::consts::PI * radius.powi(4));
    let value = radius - dst;
    -2.0 * normalization_factor_2d * diff * value / dst
  } else {
    0.0
  }
}

pub fn viscosity_kernel_ref(radius: f32, dst2: f32) -> f32 {
  let radius2 = radius * radius;
  if dst2 > radius2 {
    return 0.0;
  }
  let normalization_factor_2d = 4.0 / (std::f32::consts::PI * radius.powi(8));
  let value = radius2 - dst2;
  normalization_factor_2d * value * value * value
}
