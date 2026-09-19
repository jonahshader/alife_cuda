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
//! The prefix scan the counting sort needs is [`super::scan`], which the
//! organism slot allocator shares.
//!
//! Layout: cell `c` owns `sorted_ids[cell_start[c] .. cell_start[c] +
//! cell_counts[c]]`. Unlike the C++ fixed stride, nothing is dropped when a
//! cell is crowded; the neighbour kernels still read at most
//! `max_particles_per_cell` entries, which is exactly what the C++ `min(...,
//! max_particles_per_cell)` clamp does.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::scan;
use super::{
  Cfg, GridArgs, P_CELL_SIZE, SCAN_THREADS, SphArgs, cube_count, in_fluid, particle_to_cid,
  sph_args, whole,
};
use crate::particles::{SphDevice, SphHost};

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

/// Histogram: how many particles land in each cell, and which cell each
/// particle landed in. Liquid and body particles are in the grid; vapor and
/// free slots are not.
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
  if !in_fluid(sph.state[i]) {
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

/// Place every in-grid particle into its cell's slice. The slot a particle wins
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
  live: super::LiveParticles,
  timed: &mut dyn FnMut(&'static str, &mut (dyn FnMut() + Send)),
) {
  let cells = cfg.num_cells as usize;
  // The buffers are the full capacity; only the launches are the live prefix.
  let particles = cfg.num_particles as usize;
  let launched = live.get();

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
      cube_count(launched),
      CubeDim::new_1d(super::CUBE_DIM),
      sph_args(sph),
      whole(&grid.cell_counts, cells),
      whole(&grid.particle_cell, particles),
      whole(params, super::PARAM_COUNT),
      cfg,
    );
  });

  // `cell_cursor` comes out as a copy of `cell_start`; `scatter_ids`
  // consumes it.
  timed("scan_cell_starts", &mut || {
    scan::exclusive_scan(
      client,
      &grid.cell_counts,
      &grid.partials,
      &grid.cell_start,
      &grid.cell_cursor,
      scan::ScanCfg { n: cfg.num_cells },
    );
  });

  timed("scatter_ids", &mut || {
    scatter_ids::launch::<R>(
      client,
      cube_count(launched),
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
    if !particles.state[i].in_fluid() {
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
  // written — vapor and free slots take none — so that is the meaningful
  // length.
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

// --- Reading the grid: the 3x3 neighbourhood every gather kernel walks ---

/// One cell of the 3x3 neighbourhood, as the three things a gather needs from
/// it: where its slice of `sorted_ids` starts, how many entries to read, and
/// the shift that unwraps a neighbour reached across the world's x seam.
#[derive(CubeType, Clone, Copy)]
pub struct NeighbourCell {
  pub start: u32,
  pub count: u32,
  /// Added to a neighbour's x before differencing. Zero for the eight of nine
  /// visits that do not cross the seam.
  pub x_shift: f32,
}

/// The cell `(dx - 1, dy - 1)` away from `(cell_x, cell_y)`, with `dx` and `dy`
/// each in `0..3`.
///
/// x wraps; y is clipped at the world floor and ceiling, and a row outside the
/// world yields an empty cell rather than an index. The count is clamped to
/// `max_per_cell`, which is exactly what the C++ `min(...,
/// max_particles_per_cell)` does — unlike the C++ this grid drops nothing when
/// a cell is crowded, so the clamp is where the two agree again.
///
/// The callers keep the 3x3 as two nested loops and pass `dx` and `dy` rather
/// than flattening it to one `0..9`: flattened, wgpu spends 14% longer in
/// `calculate_accel` (0.263 ms against 0.230 ms, measured both ways twice).
/// The nesting is also the row-then-column order the C++ gathers in, which is
/// their float summation order, so it is not free to change either.
#[cube]
pub fn neighbour_cell(
  grid: &GridArgs,
  cell_x: i32,
  cell_y: i32,
  dx: u32,
  dy: u32,
  bounds_x: f32,
  #[comptime] cfg: Cfg,
) -> NeighbourCell {
  let yi = cell_y + dy as i32 - 1;
  let xi = cell_x + dx as i32 - 1;

  // wrap x if out of horizontal bounds
  let mut x_shift = 0.0f32;
  if xi < 0 {
    x_shift = -bounds_x;
  } else if xi >= cfg.grid_w {
    x_shift = bounds_x;
  }

  let mut start = 0u32;
  let mut count = 0u32;
  // skip if cell is out of vertical bounds
  if yi >= 0 && yi < cfg.grid_h {
    let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
    let index = (yi * cfg.grid_w + wrapped_x) as usize;
    start = grid.cell_start[index];
    count = grid.cell_counts[index];
    if count > cfg.max_per_cell {
      count = cfg.max_per_cell;
    }
  }

  NeighbourCell {
    start,
    count,
    x_shift,
  }
}

/// Move a neighbour's x across the world's seam, if that is where it was
/// reached from.
///
/// The branch is not an optimisation. An unconditional `x + x_shift` is the
/// same number for the eight cells whose shift is zero, but it moves the CUDA
/// backend's float contraction: the gathers then differ in their low bits from
/// what the hand-written loops computed — measured against their dumps, one
/// step, 4157 of 51200 densities off by up to 4 ulp. The host twin needs no
/// such guard.
#[cube]
pub fn unwrap_x(x: f32, x_shift: f32) -> f32 {
  let mut out = x;
  if x_shift != 0.0f32 {
    out += x_shift;
  }
  out
}

/// Host twin of [`neighbour_cell`]: every neighbour particle of a cell, paired
/// with the x shift that unwraps it, in the same order the kernels visit them.
pub fn neighbours<'a>(
  grid: &'a GridRef,
  cell_x: i32,
  cell_y: i32,
  bounds_x: f32,
  cfg: &'a Cfg,
) -> impl Iterator<Item = (usize, f32)> + 'a {
  (0..3i32).flat_map(move |dy| {
    (0..3i32).flat_map(move |dx| {
      let yi = cell_y + dy - 1;
      let xi = cell_x + dx - 1;

      let x_shift = if xi < 0 {
        -bounds_x
      } else if xi >= cfg.grid_w {
        bounds_x
      } else {
        0.0
      };

      let (start, count) = if yi >= 0 && yi < cfg.grid_h {
        let wrapped_x = (xi + cfg.grid_w) % cfg.grid_w;
        let index = (yi * cfg.grid_w + wrapped_x) as usize;
        (
          grid.cell_start[index] as usize,
          grid.cell_counts[index].min(cfg.max_per_cell) as usize,
        )
      } else {
        (0, 0)
      };

      grid.sorted_ids[start..start + count]
        .iter()
        .map(move |id| (*id as usize, x_shift))
    })
  })
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
  fn every_in_fluid_particle_is_placed_exactly_once() {
    let h = Harness::new();
    h.build_grid();
    let grid = h.read_grid();

    let placed = h.particles.state.iter().filter(|k| k.in_fluid()).count();
    assert_eq!(grid.cell_counts.iter().sum::<u32>() as usize, placed);

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
    assert_eq!(seen.iter().filter(|s| **s).count(), placed);
    for (i, s) in seen.iter().enumerate() {
      assert_eq!(*s, h.particles.state[i].in_fluid(), "particle {i}");
    }
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
