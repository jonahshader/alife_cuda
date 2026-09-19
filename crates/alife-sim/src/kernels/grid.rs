//! Deterministic neighbour-grid build: a counting sort by cell.
//!
//! The C++ `populate_grid_indices` puts each particle in its cell with one
//! `atomicAdd` into a fixed `max_particles_per_cell` stride, so a cell's
//! particle order — and with it the float summation order of every neighbour
//! kernel — depends on which atomics won the race. Two runs of the same binary
//! at the same seed already diverge for that reason (`docs/perf.md`).
//!
//! This port sorts instead, which is a deliberate difference: same-seed runs
//! are bit-reproducible on a given runtime. The result is the stable order a
//! sequential build would produce — ascending particle id within a cell.
//!
//! Layout: cell `c` owns `sorted_ids[cell_start[c] .. cell_start[c] +
//! cell_counts[c]]`. Unlike the C++ fixed stride, nothing is dropped when a
//! cell is crowded; the neighbour kernels still read at most
//! `max_particles_per_cell` entries, which is exactly what the C++ `min(...,
//! max_particles_per_cell)` clamp does.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::{
  Cfg, P_CELL_SIZE, SCAN_THREADS, SphArgs, cube_count, particle_to_cid, sph_args, whole,
};
use crate::particles::{ParticleKind, SphDevice, SphHost};

/// Marks a particle that is not in the grid at all — the C++ skips vapor with
/// `if (sph.state[i] != 0) return;` before ever computing a cell.
pub const NOT_IN_GRID: u32 = 0xFFFF_FFFF;

/// Device buffers backing the grid build.
#[derive(Debug, Clone)]
pub struct GridDevice {
  pub cell_counts: Handle,
  /// Per-block totals for the prefix scan; `SCAN_THREADS` entries.
  pub partials: Handle,
  pub cell_start: Handle,
  pub cell_cursor: Handle,
  pub particle_cell: Handle,
  pub sorted_ids: Handle,
}

impl GridDevice {
  pub fn alloc<R: Runtime>(client: &ComputeClient<R>, cfg: &Cfg) -> Self {
    let cells = cfg.num_cells as usize * size_of::<u32>();
    let particles = cfg.num_particles as usize * size_of::<u32>();
    Self {
      cell_counts: client.empty(cells),
      partials: client.empty(SCAN_THREADS as usize * size_of::<u32>()),
      cell_start: client.empty(cells),
      cell_cursor: client.empty(cells),
      particle_cell: client.empty(particles),
      sorted_ids: client.empty(particles),
    }
  }
}

#[cube(launch)]
pub fn reset_cell_counts(cell_counts: &mut [u32], #[comptime] cfg: Cfg) {
  let i = ABSOLUTE_POS;
  if i < cfg.num_cells as usize {
    cell_counts[i] = 0u32;
  }
}

/// Histogram: how many liquid particles land in each cell, and which cell each
/// particle landed in.
#[cube(launch)]
pub fn count_cells(
  sph: &SphArgs,
  cell_counts: &[Atomic<u32>],
  particle_cell: &mut [u32],
  params: &[f32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != 0u32 {
    particle_cell[i] = NOT_IN_GRID;
    terminate!();
  }

  let cid = particle_to_cid(
    sph.pos[2 * i],
    sph.pos[2 * i + 1],
    params[P_CELL_SIZE as usize],
    cfg,
  );
  particle_cell[i] = cid;
  cell_counts[cid as usize].fetch_add(1u32);
}

/// Exclusive prefix sum over the cell histogram, in three launches.
///
/// Each of the `SCAN_THREADS` units owns a contiguous block of cells: it sums
/// its block, one unit scans the per-block totals, then every unit walks its
/// block again writing the running offsets. Three launches rather than one
/// kernel with two `sync_cube()` barriers, because a cube-wide barrier costs
/// the CPU runtime ~140 ms per barrier at this cube size — it turned a 25 us
/// scan into 280 ms. Nothing here uses shared memory or a barrier, so this is
/// also the portable shape.
///
/// `cell_cursor` starts as a copy of `cell_start`; `scatter_ids` consumes it.
#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn scan_block_sums(cell_counts: &[u32], partials: &mut [u32], #[comptime] cfg: Cfg) {
  let t = ABSOLUTE_POS;
  if t >= comptime!(SCAN_THREADS as usize) {
    terminate!();
  }
  let block = comptime!(cfg.num_cells.div_ceil(SCAN_THREADS) as usize);
  let cells = comptime!(cfg.num_cells as usize);
  let begin = t * block;
  let mut sum = 0u32;
  if begin < cells {
    let mut end = begin + block;
    if end > cells {
      end = cells;
    }
    for c in begin..end {
      sum += cell_counts[c];
    }
  }
  partials[t] = sum;
}

/// Exclusive scan of the per-block totals, on one unit. `SCAN_THREADS` serial
/// adds, which is the price of not needing a barrier.
#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn scan_partials(partials: &mut [u32]) {
  if ABSOLUTE_POS != 0 {
    terminate!();
  }
  let mut running = 0u32;
  for k in 0..comptime!(SCAN_THREADS as usize) {
    let value = partials[k];
    partials[k] = running;
    running += value;
  }
}

#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn scan_write_starts(
  cell_counts: &[u32],
  partials: &[u32],
  cell_start: &mut [u32],
  cell_cursor: &mut [u32],
  #[comptime] cfg: Cfg,
) {
  let t = ABSOLUTE_POS;
  if t >= comptime!(SCAN_THREADS as usize) {
    terminate!();
  }
  let block = comptime!(cfg.num_cells.div_ceil(SCAN_THREADS) as usize);
  let cells = comptime!(cfg.num_cells as usize);
  let begin = t * block;
  if begin < cells {
    let mut end = begin + block;
    if end > cells {
      end = cells;
    }
    let mut running = partials[t];
    for c in begin..end {
      cell_start[c] = running;
      cell_cursor[c] = running;
      running += cell_counts[c];
    }
  }
}

/// Place every liquid particle into its cell's slice. The slot a particle wins
/// is race-dependent; `sort_cells` below makes the outcome deterministic.
#[cube(launch)]
pub fn scatter_ids(
  particle_cell: &[u32],
  cell_cursor: &[Atomic<u32>],
  sorted_ids: &mut [u32],
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  let cid = particle_cell[i];
  if cid == NOT_IN_GRID {
    terminate!();
  }
  let slot = cell_cursor[cid as usize].fetch_add(1u32);
  sorted_ids[slot as usize] = i as u32;
}

/// Sort each cell's slice ascending by particle id, which is the order a
/// stable sequential counting sort would have produced. Insertion sort: a cell
/// holds `particles_per_cell` entries on average, four by default.
#[cube(launch)]
pub fn sort_cells(
  cell_counts: &[u32],
  cell_start: &[u32],
  sorted_ids: &mut [u32],
  #[comptime] cfg: Cfg,
) {
  let c = ABSOLUTE_POS;
  if c >= cfg.num_cells as usize {
    terminate!();
  }
  let start = cell_start[c] as usize;
  let count = cell_counts[c];

  // `1..count` would be an empty Rust range for an empty cell, but the loop
  // bound reaches the kernel as `count - 1` and underflows, so the guard is
  // load-bearing. The inner loop is bounded rather than a `while`, for the
  // same reason: k shifts are the most an insertion can need.
  if count > 1u32 {
    for k in 1..count {
      let value = sorted_ids[start + k as usize];
      let mut j = k;
      for _ in 0..k {
        if j > 0u32 && sorted_ids[start + j as usize - 1] > value {
          sorted_ids[start + j as usize] = sorted_ids[start + j as usize - 1];
          j -= 1u32;
        }
      }
      sorted_ids[start + j as usize] = value;
    }
  }
}

/// Run the whole grid build.
#[allow(clippy::too_many_arguments)]
pub fn build<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  grid: &GridDevice,
  params: &Handle,
  cfg: Cfg,
  timed: &mut dyn FnMut(&'static str, &mut (dyn FnMut() + Send)),
) {
  let cells = cfg.num_cells as usize;
  let particles = cfg.num_particles as usize;

  timed("reset_cell_counts", &mut || {
    reset_cell_counts::launch::<R>(
      client,
      cube_count(cells),
      CubeDim::new_1d(super::CUBE_DIM),
      whole(&grid.cell_counts, cells),
      cfg,
    );
  });

  timed("count_cells", &mut || {
    count_cells::launch::<R>(
      client,
      cube_count(particles),
      CubeDim::new_1d(super::CUBE_DIM),
      sph_args(sph),
      whole(&grid.cell_counts, cells),
      whole(&grid.particle_cell, particles),
      whole(params, super::PARAM_COUNT),
      cfg,
    );
  });

  timed("scan_cell_starts", &mut || {
    let threads = SCAN_THREADS as usize;
    scan_block_sums::launch::<R>(
      client,
      cube_count(threads),
      CubeDim::new_1d(super::CUBE_DIM),
      whole(&grid.cell_counts, cells),
      whole(&grid.partials, threads),
      cfg,
    );
    scan_partials::launch::<R>(
      client,
      CubeCount::Static(1, 1, 1),
      CubeDim::new_1d(1),
      whole(&grid.partials, threads),
    );
    scan_write_starts::launch::<R>(
      client,
      cube_count(threads),
      CubeDim::new_1d(super::CUBE_DIM),
      whole(&grid.cell_counts, cells),
      whole(&grid.partials, threads),
      whole(&grid.cell_start, cells),
      whole(&grid.cell_cursor, cells),
      cfg,
    );
  });

  timed("scatter_ids", &mut || {
    scatter_ids::launch::<R>(
      client,
      cube_count(particles),
      CubeDim::new_1d(super::CUBE_DIM),
      whole(&grid.particle_cell, particles),
      whole(&grid.cell_cursor, cells),
      whole(&grid.sorted_ids, particles),
      cfg,
    );
  });

  timed("sort_cells", &mut || {
    sort_cells::launch::<R>(
      client,
      cube_count(cells),
      CubeDim::new_1d(super::CUBE_DIM),
      whole(&grid.cell_counts, cells),
      whole(&grid.cell_start, cells),
      whole(&grid.sorted_ids, particles),
      cfg,
    );
  });
}

/// Plain-Rust reference for the whole grid build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridRef {
  pub cell_counts: Vec<u32>,
  pub cell_start: Vec<u32>,
  pub sorted_ids: Vec<u32>,
}

pub fn build_ref(particles: &SphHost, cell_size: f32, cfg: &Cfg) -> GridRef {
  let cells = cfg.num_cells as usize;
  let mut cell_counts = vec![0u32; cells];
  let mut particle_cell = vec![NOT_IN_GRID; particles.len()];

  for (i, slot) in particle_cell.iter_mut().enumerate() {
    if particles.state[i] != ParticleKind::Liquid {
      continue;
    }
    let cid = super::particle_to_cid_ref(particles.pos[i], cell_size, cfg);
    *slot = cid;
    cell_counts[cid as usize] += 1;
  }

  let mut cell_start = vec![0u32; cells];
  let mut running = 0u32;
  for c in 0..cells {
    cell_start[c] = running;
    running += cell_counts[c];
  }

  // Ascending particle id, which is what the stable scatter plus per-cell
  // sort produces on the device. Only the first `running` slots are ever
  // written — vapor takes no slot — so that is the meaningful length.
  let mut cursor = cell_start.clone();
  let mut sorted_ids = vec![NOT_IN_GRID; running as usize];
  for (i, cid) in particle_cell.iter().enumerate() {
    if *cid == NOT_IN_GRID {
      continue;
    }
    let slot = &mut cursor[*cid as usize];
    sorted_ids[*slot as usize] = i as u32;
    *slot += 1;
  }

  GridRef {
    cell_counts,
    cell_start,
    sorted_ids,
  }
}

#[cfg(test)]
mod tests {
  use super::super::test_support::Harness;

  #[test]
  fn matches_reference() {
    let h = Harness::new();
    h.build_grid();
    assert_eq!(h.read_grid(), h.grid_ref());
  }

  #[test]
  fn every_liquid_particle_is_placed_exactly_once() {
    let h = Harness::new();
    h.build_grid();
    let grid = h.read_grid();

    let liquid = h.particles.liquid_count();
    assert_eq!(grid.cell_counts.iter().sum::<u32>() as usize, liquid);

    let mut seen = vec![false; h.particles.len()];
    for c in 0..grid.cell_counts.len() {
      let start = grid.cell_start[c] as usize;
      for k in 0..grid.cell_counts[c] as usize {
        let id = grid.sorted_ids[start + k] as usize;
        assert!(!seen[id], "particle {id} placed twice");
        seen[id] = true;
        // Ascending within the cell: that is what makes the build
        // deterministic.
        if k > 0 {
          assert!(grid.sorted_ids[start + k - 1] < id as u32);
        }
      }
    }
    assert_eq!(seen.iter().filter(|s| **s).count(), liquid);
  }

  #[test]
  fn rebuilding_gives_the_same_grid() {
    let h = Harness::new();
    h.build_grid();
    let first = h.read_grid();
    h.build_grid();
    assert_eq!(h.read_grid(), first);
  }
}
