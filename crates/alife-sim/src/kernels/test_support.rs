//! One small fixed world, shared by every kernel-versus-reference test.
//!
//! The tests run on the CPU runtime because it needs no GPU and the kernel
//! source is the same on all of them; what they check is the kernel against
//! its plain-Rust twin, not one backend against another. Backend agreement is
//! the parity test's job (`tests/parity.rs`).

use cubecl::prelude::*;
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_runtime::server::Handle;
use glam::Vec2;

use super::grid::{GridDevice, GridRef};
use super::{Cfg, PARAM_COUNT, pack_params};
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState};
use crate::genome::Population;
use crate::particles::{ParticleKind, SphDevice, SphHost};
use crate::rng::{threefry4x32_20_ref, u01_ref};
use crate::soil::{SoilDevice, SoilGrid, TerrainMode};
use crate::world::WorldGeometry;

/// A world small enough to debug by hand but wide enough that the wrap seam,
/// the floor, the ceiling, crowded cells and vapor particles are all live.
pub struct Harness {
  pub client: ComputeClient<CpuRuntime>,
  pub params: SimParams,
  pub geom: WorldGeometry,
  pub cfg: Cfg,
  pub soil: SoilGrid,
  pub particles: SphHost,
  pub sph: SphDevice,
  pub soil_dev: SoilDevice,
  pub grid: GridDevice,
  pub params_buf: Handle,
  pub vel_next: Handle,
}

impl Default for Harness {
  fn default() -> Self {
    Self::new()
  }
}

impl Harness {
  pub fn new() -> Self {
    let params = SimParams {
      world_width: 2.0,
      world_height: 1.5,
      smoothing_radius: 0.25,
      soil_cell_size: 0.1,
      particles_per_cell: 3,
      terrain_mode: 1,
      // Big enough that nothing evaporates by accident, small enough
      // that the probability path still runs.
      evap_rate: 0.01,
      // A body capacity in proportion to this world: the default 256 x 16 x 8
      // would bury 144 fluid particles under 32,768 free slots.
      max_organisms: 4,
      max_limbs: 5,
      max_particles_per_limb: 3,
      ..SimParams::default()
    };
    let geom = WorldGeometry::from_params(&params);
    let cfg = geom.cfg();
    let soil = SoilGrid::new(
      geom.soil_width,
      geom.soil_height,
      geom.soil_cell_size,
      TerrainMode::from_flag(params.terrain_mode),
      7,
    );
    let particles = fixture_particles(&geom);

    let client = CpuRuntime::client(&CpuDevice);
    let sph = SphDevice::upload(&client, &particles);
    let soil_dev = SoilDevice::upload(&client, &soil.cells);
    let grid = GridDevice::alloc(&client, &cfg);
    let params_buf = client.create_from_slice(bytemuck::cast_slice(&pack_params(&params, &geom)));
    let vel_next = client.empty(particles.len() * 2 * size_of::<f32>());

    Self {
      client,
      params,
      geom,
      cfg,
      soil,
      particles,
      sph,
      soil_dev,
      grid,
      params_buf,
      vel_next,
    }
  }

  /// Launch over every slot: the fixture's body particles sit among the
  /// fluid ones rather than above it, so there is no live prefix to take.
  pub fn live(&self) -> super::LiveParticles {
    super::LiveParticles(self.geom.num_particles)
  }

  pub fn build_grid(&self) {
    super::grid::build(
      &self.client,
      &self.sph,
      &self.grid,
      &self.params_buf,
      self.cfg,
      self.live(),
      &mut |_, f| f(),
    );
  }

  pub fn grid_ref(&self) -> GridRef {
    super::grid::build_ref(&self.particles, self.geom.cell_size, &self.cfg)
  }

  pub fn read_grid(&self) -> GridRef {
    let read = |handle: &Handle, n: usize| -> Vec<u32> {
      let bytes = self.client.read_one_unchecked(handle.clone());
      bytemuck::cast_slice::<u8, u32>(&bytes)[..n].to_vec()
    };
    let cells = self.cfg.num_cells as usize;
    let cell_counts = read(&self.grid.cell_counts, cells);
    // Only the placed particles occupy slots; the tail of the buffer is
    // never written and holds whatever the allocator handed back.
    let placed = cell_counts.iter().sum::<u32>() as usize;
    GridRef {
      cell_start: read(&self.grid.cell_start, cells),
      sorted_ids: read(&self.grid.sorted_ids, placed),
      cell_counts,
    }
  }

  pub fn read_particles(&self) -> SphHost {
    self.sph.download(&self.client)
  }

  pub fn param_count(&self) -> usize {
    PARAM_COUNT
  }
}

/// One organism in a small world, for the constraint and geometry passes.
///
/// The body is [`crate::genome::Genome::seed_plant`] — root, stem, leaf — laid
/// into body slots by hand and then knocked out of shape, so a projection has
/// something to do. Laying it out properly is `bodies::spawn`'s job; this
/// fixture deliberately does not use it, so a broken spawn cannot make a
/// constraint test pass.
pub struct OrganismHarness {
  pub client: ComputeClient<CpuRuntime>,
  pub params: SimParams,
  pub geom: WorldGeometry,
  pub cfg: Cfg,
  pub body_cfg: BodyCfg,
  pub particles: SphHost,
  pub sph: SphDevice,
  pub pop: Population,
  pub bodies: BodyState,
  pub params_buf: Handle,
}

impl Default for OrganismHarness {
  fn default() -> Self {
    Self::new()
  }
}

impl OrganismHarness {
  pub fn new() -> Self {
    let params = SimParams {
      world_width: 4.0,
      world_height: 3.0,
      smoothing_radius: 0.5,
      soil_cell_size: 0.25,
      particles_per_cell: 1,
      terrain_mode: 1,
      max_organisms: 2,
      max_limbs: 4,
      max_particles_per_limb: 3,
      limb_segment_length: 0.15,
      ..SimParams::default()
    };
    let geom = WorldGeometry::from_params(&params);
    let cfg = geom.cfg();
    let client = CpuRuntime::client(&CpuDevice);

    let shape = crate::genome::BrainShape::from_params(&params);
    let mut pop = Population::new(&client, &params, shape);
    let genome = crate::genome::Genome::seed_plant(&shape, pop.max_limbs, 0, 5);
    pop.write_genome(0, &genome);
    pop.organisms.alive[0] = 1;
    pop.upload(&client);

    let mut particles = fixture_particles(&geom);
    let mut bodies = BodyState::new(&client, &params, &geom);
    let body_cfg = bodies.cfg;

    // Organism 0 takes the first body slots, limb by limb, and each particle
    // starts somewhere plausible but wrong.
    let mut next = geom.fluid_particles as u32;
    let anchor = Vec2::new(1.0, 1.0);
    bodies.anchors[0] = anchor;
    for limb in 0..pop.max_limbs {
      let record = pop.limb_index(0, limb);
      if !pop.limbs.part_type[record].is_present() {
        continue;
      }
      let count = (pop.limbs.length[record] as usize).min(body_cfg.max_particles_per_limb as usize);
      for i in 0..count {
        let id = next as usize;
        next += 1;
        bodies.limb_particles[body_cfg.limb_slice(0, limb).start + i] = id as u32;
        let offset = Vec2::new(0.02 * (limb + i) as f32, 0.05 * (1 + limb + i) as f32);
        particles.pos[id] = anchor + offset;
        particles.ppos[id] = anchor + offset;
        particles.vel[id] = Vec2::new(0.1, -0.2);
        particles.mass[id] = 1.0;
        particles.state[id] = ParticleKind::Body;
        particles.organism[id] = 0;
        particles.limb[id] = limb as u8;
        particles.index_in_limb[id] = i as u8;
        particles.part_type[id] = pop.limbs.part_type[record];
      }
    }
    bodies.upload(&client);

    let sph = SphDevice::upload(&client, &particles);
    let params_buf = client.create_from_slice(bytemuck::cast_slice(&pack_params(&params, &geom)));

    Self {
      client,
      params,
      geom,
      cfg,
      body_cfg,
      particles,
      sph,
      pop,
      bodies,
      params_buf,
    }
  }

  pub fn read_particles(&self) -> SphHost {
    self.sph.download(&self.client)
  }
}

/// Deterministic particles: a spread over the lower half of the world, plus
/// deliberate placements at the wrap seam, the floor and the ceiling.
///
/// Every particle kind is present: a fixture without body and free slots
/// would leave the guards that tell them apart untested.
fn fixture_particles(geom: &WorldGeometry) -> SphHost {
  let n = geom.fluid_particles;
  let mut particles = SphHost::new(n);
  let ctr = [0xFACE, 0, 0, 0];

  for i in 0..n {
    let r = threefry4x32_20_ref(ctr, [i as u32, 0, 0, 0]);
    let pos = Vec2::new(
      u01_ref(r[0]) * geom.bounds.x,
      u01_ref(r[1]) * geom.bounds.y * 0.6,
    );
    particles.pos[i] = pos;
    particles.ppos[i] = pos;
    particles.vel[i] = Vec2::new((u01_ref(r[2]) - 0.5) * 0.4, (u01_ref(r[3]) - 0.5) * 0.4);
    particles.mass[i] = 0.9 + 0.2 * u01_ref(r[2]);
    particles.sym_break[i] = (r[3] % 256) as u8;
    // Every seventh particle is vapor and every eleventh a body particle, so
    // every `state` guard is live.
    particles.state[i] = if i % 7 == 0 {
      ParticleKind::Vapor
    } else if i % 11 == 0 {
      ParticleKind::Body
    } else {
      ParticleKind::Liquid
    };
    if particles.state[i] == ParticleKind::Body {
      particles.organism[i] = (i % 3) as u32;
      particles.limb[i] = (i % 4) as u8;
      particles.index_in_limb[i] = (i % 5) as u8;
      particles.part_type[i] = crate::genome::PartType::from_byte(1 + (i % 4) as u8);
    }
  }

  // Corner cases, in cells the neighbour loops have to wrap or clamp into.
  let edge = [
    Vec2::new(0.0, 0.0),
    Vec2::new(geom.bounds.x - 1e-4, 0.001),
    Vec2::new(1e-4, geom.bounds.y - 1e-4),
    Vec2::new(geom.bounds.x - 1e-4, geom.bounds.y - 1e-4),
    // Two coincident particles, so the gradient's `dst > 1e-5` guard runs.
    Vec2::new(1.0, 0.5),
    Vec2::new(1.0, 0.5),
  ];
  for (k, pos) in edge.iter().enumerate() {
    particles.pos[k] = *pos;
    particles.ppos[k] = *pos;
    particles.state[k] = ParticleKind::Liquid;
  }

  // The body-slot capacity, exactly as a real world reserves it: parked
  // outside the grid and skipped by every kernel.
  particles.push_free_slots(geom.body_slots);
  particles
}

/// Largest absolute difference between two float fields, and the index where
/// it happens.
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
  assert_eq!(a.len(), b.len());
  let mut worst = (0.0f32, 0usize);
  for (i, (x, y)) in a.iter().zip(b).enumerate() {
    let d = (x - y).abs();
    if d > worst.0 {
      worst = (d, i);
    }
  }
  worst
}

pub fn max_abs_diff_vec2(a: &[Vec2], b: &[Vec2]) -> (f32, usize) {
  assert_eq!(a.len(), b.len());
  let mut worst = (0.0f32, 0usize);
  for (i, (x, y)) in a.iter().zip(b).enumerate() {
    let d = (*x - *y).abs().max_element();
    if d > worst.0 {
      worst = (d, i);
    }
  }
  worst
}

/// Relative tolerance for a kernel-versus-reference comparison.
///
/// The two sum in the same order — the sorted grid makes that true — so the
/// only source of difference is the JIT contracting a multiply-add where the
/// host does not.
pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
  let (diff, index) = max_abs_diff(actual, expected);
  let scale = expected.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
  assert!(
    diff <= tolerance * scale,
    "{what}: max |diff| {diff} at {index} (device {}, reference {}) exceeds {tolerance} * {scale}",
    actual[index],
    expected[index],
  );
}

pub fn assert_close_vec2(actual: &[Vec2], expected: &[Vec2], tolerance: f32, what: &str) {
  let (diff, index) = max_abs_diff_vec2(actual, expected);
  let scale = expected
    .iter()
    .fold(1.0f32, |acc, v| acc.max(v.abs().max_element()));
  assert!(
    diff <= tolerance * scale,
    "{what}: max |diff| {diff} at {index} (device {:?}, reference {:?}) exceeds {tolerance} * {scale}",
    actual[index],
    expected[index],
  );
}
