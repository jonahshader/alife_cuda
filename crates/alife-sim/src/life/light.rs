//! Light: a per-soil-cell occlusion grid, rebuilt every `life_interval` steps.
//!
//! `docs/organism.md`, *World coupling and life cycle*: "Leaf particles gain
//! energy from light, computed by a per-column occlusion scan from the top of
//! the world so plants shade each other."
//!
//! Two launches. The first counts the stem and leaf particles in each soil
//! cell; the second walks each column from the top down, dimming the light by
//! `light_attenuation` once per particle it has passed.
//!
//! **The count is an integer atomic on purpose.** Adding 1 to a `u32` is
//! associative and exact, so the order the units arrive in cannot change the
//! result — which is what `docs/organism.md`'s "no atomics that decide order"
//! rules out, and this is not that. A float accumulation here, or an atomic
//! that handed out slots, would make the run non-reproducible.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::genome::PartType;
use crate::kernels::{
  CUBE_DIM, Cfg, KIND_BODY, LiveParticles, P_LIGHT_ATTENUATION, P_LIGHT_TOP, P_SOIL_SIZE,
  PARAM_COUNT, clamp_i32, cube_count, whole,
};
use crate::particles::{SphDevice, SphHost};
use crate::soil::SoilGrid;

/// Part-type codes that block light. Stem and leaf stand above ground; a root
/// is in the soil, where there is nothing to shade, and a seed is in flight.
const TYPE_STEM: u32 = PartType::Stem as u32;
const TYPE_LEAF: u32 = PartType::Leaf as u32;

/// The soil cell a world position falls in. x wraps, y clamps — the same
/// asymmetry as the particle grid and [`crate::kernels::soil_sample`].
#[cube]
pub fn soil_cell_index(pos_x: f32, pos_y: f32, soil_size: f32, #[comptime] cfg: Cfg) -> u32 {
  let x0 = f32::floor(pos_x / soil_size) as i32;
  let x = ((x0 % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let y = clamp_i32(f32::floor(pos_y / soil_size) as i32, 0, cfg.soil_h - 1);
  (y * cfg.soil_w + x) as u32
}

/// Host twin of [`soil_cell_index`].
pub fn soil_cell_index_ref(pos: glam::Vec2, soil_size: f32, cfg: &Cfg) -> usize {
  let x0 = (pos.x / soil_size).floor() as i32;
  let x = ((x0 % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let y = ((pos.y / soil_size).floor() as i32).clamp(0, cfg.soil_h - 1);
  (y * cfg.soil_w + x) as usize
}

/// One unit per particle slot: count the stem and leaf particles per soil cell.
#[cube(launch)]
pub fn count_occluders(
  ppos: &[f32],
  state: &[u32],
  part_type: &[u32],
  counts: &[Atomic<u32>],
  params: &[f32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if state[i] != KIND_BODY {
    terminate!();
  }
  let t = part_type[i];
  if t != TYPE_STEM && t != TYPE_LEAF {
    terminate!();
  }
  let cell = soil_cell_index(
    ppos[2 * i],
    ppos[2 * i + 1],
    params[P_SOIL_SIZE as usize],
    cfg,
  );
  counts[cell as usize].fetch_add(1u32);
}

/// One unit per soil column: light arriving at each cell, top row first.
///
/// The counts are consumed as they are read — the next rebuild wants a clean
/// grid, and this pass is the only reader, so zeroing here saves a launch.
#[cube(launch)]
pub fn column_light(counts: &mut [u32], light: &mut [f32], params: &[f32], #[comptime] cfg: Cfg) {
  let x = ABSOLUTE_POS;
  if x >= cfg.soil_w as usize {
    terminate!();
  }
  let attenuation = params[P_LIGHT_ATTENUATION as usize];
  let mut l = params[P_LIGHT_TOP as usize];
  for row in 0..cfg.soil_h as usize {
    // Top row first: row `soil_h - 1` is the top of the world.
    let cell = (cfg.soil_h as usize - 1 - row) * cfg.soil_w as usize + x;
    light[cell] = l;
    let n = counts[cell];
    counts[cell] = 0u32;
    // Repeated multiplication rather than `powf`: exact, and identical on
    // every backend, where a transcendental is not (`docs/organism.md`,
    // decisions, 2026-09-19). `n` is the occluders in one 0.1 m cell.
    for _ in 0..n {
      l *= attenuation;
    }
  }
}

/// The occlusion grid: one `u32` count and one `f32` light value per soil cell.
#[derive(Debug, Clone)]
pub struct LightGrid {
  pub counts: Handle,
  pub light: Handle,
  pub cells: usize,
}

impl LightGrid {
  /// Allocated with full light everywhere, so the sensors read something
  /// sensible before the first rebuild.
  pub fn new<R: Runtime>(client: &ComputeClient<R>, soil: &SoilGrid, light_top: f32) -> Self {
    let cells = soil.len();
    Self {
      counts: client.create_from_slice(bytemuck::cast_slice(&vec![0u32; cells])),
      light: client.create_from_slice(bytemuck::cast_slice(&vec![light_top; cells])),
      cells,
    }
  }

  pub fn read<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<f32> {
    crate::genome::population::read_f32(client, &self.light, self.cells)
  }
}

/// Rebuild the grid: count, then scan each column.
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  grid: &LightGrid,
  params: &Handle,
  cfg: Cfg,
  live: LiveParticles,
) {
  let n = cfg.num_particles as usize;
  count_occluders::launch::<R>(
    client,
    cube_count(live.get()),
    CubeDim::new_1d(CUBE_DIM),
    whole(&sph.ppos, n * 2),
    whole(&sph.state, n),
    whole(&sph.part_type, n),
    whole(&grid.counts, grid.cells),
    whole(params, PARAM_COUNT),
    cfg,
  );
  column_light::launch::<R>(
    client,
    cube_count(cfg.soil_w as usize),
    CubeDim::new_1d(CUBE_DIM),
    whole(&grid.counts, grid.cells),
    whole(&grid.light, grid.cells),
    whole(params, PARAM_COUNT),
    cfg,
  );
}

/// Plain-Rust twin of [`count_occluders`] followed by [`column_light`].
///
/// Returns the occluder counts as the first pass leaves them — the kernel
/// zeroes them in the second pass, so a test compares the light against this
/// and the counts against a run of the first pass alone.
pub fn light_grid_ref(
  particles: &SphHost,
  soil_size: f32,
  light_top: f32,
  attenuation: f32,
  cfg: &Cfg,
) -> (Vec<u32>, Vec<f32>) {
  let cells = (cfg.soil_w * cfg.soil_h) as usize;
  let mut counts = vec![0u32; cells];
  for i in 0..particles.len() {
    if particles.state[i] != crate::ParticleKind::Body {
      continue;
    }
    if !matches!(particles.part_type[i], PartType::Stem | PartType::Leaf) {
      continue;
    }
    counts[soil_cell_index_ref(particles.ppos[i], soil_size, cfg)] += 1;
  }

  let mut light = vec![0.0f32; cells];
  for x in 0..cfg.soil_w as usize {
    let mut l = light_top;
    for row in 0..cfg.soil_h as usize {
      let cell = (cfg.soil_h as usize - 1 - row) * cfg.soil_w as usize + x;
      light[cell] = l;
      for _ in 0..counts[cell] {
        l *= attenuation;
      }
    }
  }
  (counts, light)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  /// A world with a plant in it, so the counting pass has occluders and the
  /// column scan has a column that is darker at the bottom than at the top.
  #[test]
  fn matches_reference() {
    let h = OrganismHarness::new();
    let grid = LightGrid::new(&h.client, &h.soil, h.params.light_top);
    launch(
      &h.client,
      &h.sph,
      &grid,
      &h.params_buf,
      h.cfg,
      crate::kernels::LiveParticles(h.geom.num_particles),
    );
    let actual = grid.read(&h.client);
    let (counts, expected) = light_grid_ref(
      &h.particles,
      h.geom.soil_cell_size,
      h.params.light_top,
      h.params.light_attenuation,
      &h.cfg,
    );
    assert_eq!(actual.len(), expected.len());
    assert_close(&actual, &expected, 1e-6, "light");
    assert!(
      counts.iter().sum::<u32>() > 0,
      "the fixture has stem and leaf particles to count"
    );
    assert!(
      expected.iter().any(|l| *l < h.params.light_top),
      "something is shaded"
    );
  }

  /// The counts are consumed by the column scan, so a second rebuild over the
  /// same world produces the same light rather than a darker one.
  #[test]
  fn a_rebuild_does_not_accumulate() {
    let h = OrganismHarness::new();
    let grid = LightGrid::new(&h.client, &h.soil, h.params.light_top);
    let live = crate::kernels::LiveParticles(h.geom.num_particles);
    launch(&h.client, &h.sph, &grid, &h.params_buf, h.cfg, live);
    let first = grid.read(&h.client);
    launch(&h.client, &h.sph, &grid, &h.params_buf, h.cfg, live);
    assert_eq!(grid.read(&h.client), first);
  }

  /// Light falls off by one factor of the attenuation per occluding particle
  /// above, and never rises going down a column.
  #[test]
  fn light_attenuates_monotonically_down_a_column() {
    let h = OrganismHarness::new();
    let (counts, light) = light_grid_ref(
      &h.particles,
      h.geom.soil_cell_size,
      h.params.light_top,
      h.params.light_attenuation,
      &h.cfg,
    );
    let (w, hgt) = (h.cfg.soil_w as usize, h.cfg.soil_h as usize);
    for x in 0..w {
      assert_eq!(light[(hgt - 1) * w + x], h.params.light_top);
      for row in (0..hgt - 1).rev() {
        let above = light[(row + 1) * w + x];
        let here = light[row * w + x];
        let expected = above
          * h
            .params
            .light_attenuation
            .powi(counts[(row + 1) * w + x] as i32);
        assert!((here - expected).abs() <= 1e-6, "column {x} row {row}");
        assert!(here <= above);
      }
    }
  }
}
