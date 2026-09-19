//! Exclusive prefix sum over a `u32` buffer, in three barrier-free launches.
//!
//! Each of the [`SCAN_THREADS`] units owns a contiguous block of entries: it
//! sums its block, one unit scans the per-block totals, then every unit walks
//! its block again writing the running offsets. Three launches rather than one
//! kernel with two `sync_cube()` barriers, because a cube-wide barrier costs
//! the CPU runtime ~140 ms per barrier at this cube size — it turned a 25 us
//! scan into 280 ms. Nothing here uses shared memory or a barrier, so this is
//! also the portable shape (`docs/organism.md`, decisions).
//!
//! Two consumers so far: the neighbour grid's counting sort
//! ([`super::grid`]) and the organism slot allocator
//! ([`crate::genome::slots`]).

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::{SCAN_THREADS, cube_count, whole};

/// Entry count, baked in so the per-unit block size is a compile-time
/// constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScanCfg {
  pub n: u32,
}

/// The entries one unit owns, as a half-open range.
///
/// Both block-walking passes take it from [`scan_block`], so they cannot
/// disagree about which entries a unit owns.
#[derive(CubeType, Clone, Copy)]
pub struct ScanBlock {
  pub begin: usize,
  pub end: usize,
}

/// The entries unit `t` owns. `begin` is clamped as well as `end`: a unit past
/// the last entry would otherwise get `end < begin`, and the loop bound
/// reaches the kernel as `end - begin` and underflows — the same trap
/// `grid::sort_cells` guards against.
#[cube]
fn scan_block(t: usize, #[comptime] cfg: ScanCfg) -> ScanBlock {
  let block = comptime!(cfg.n.div_ceil(SCAN_THREADS) as usize);
  let n = comptime!(cfg.n as usize);

  let mut begin = t * block;
  if begin > n {
    begin = n;
  }
  let mut end = begin + block;
  if end > n {
    end = n;
  }
  ScanBlock { begin, end }
}

#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn block_sums(values: &[u32], partials: &mut [u32], #[comptime] cfg: ScanCfg) {
  let t = ABSOLUTE_POS;
  if t >= comptime!(SCAN_THREADS as usize) {
    terminate!();
  }
  let block = scan_block(t, cfg);
  let mut sum = 0u32;
  for c in block.begin..block.end {
    sum += values[c];
  }
  partials[t] = sum;
}

/// Exclusive scan of the per-block totals, on one unit. [`SCAN_THREADS`]
/// serial adds, which is the price of not needing a barrier.
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

/// Write the running offsets. `starts` and `cursor` receive the same values:
/// a consumer that walks the offsets with an atomic cursor (the grid's
/// scatter) needs a copy it may destroy, and producing it here costs nothing
/// over producing one.
#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn write_starts(
  values: &[u32],
  partials: &[u32],
  starts: &mut [u32],
  cursor: &mut [u32],
  #[comptime] cfg: ScanCfg,
) {
  let t = ABSOLUTE_POS;
  if t >= comptime!(SCAN_THREADS as usize) {
    terminate!();
  }
  let block = scan_block(t, cfg);
  let mut running = partials[t];
  for c in block.begin..block.end {
    starts[c] = running;
    cursor[c] = running;
    running += values[c];
  }
}

/// Run the three launches. `partials` holds [`SCAN_THREADS`] entries;
/// `starts` and `cursor` hold `cfg.n` each.
pub fn exclusive_scan<R: Runtime>(
  client: &ComputeClient<R>,
  values: &Handle,
  partials: &Handle,
  starts: &Handle,
  cursor: &Handle,
  cfg: ScanCfg,
) {
  let n = cfg.n as usize;
  let threads = SCAN_THREADS as usize;
  block_sums::launch::<R>(
    client,
    cube_count(threads),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(values, n),
    whole(partials, threads),
    cfg,
  );
  scan_partials::launch::<R>(
    client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new_1d(1),
    whole(partials, threads),
  );
  write_starts::launch::<R>(
    client,
    cube_count(threads),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(values, n),
    whole(partials, threads),
    whole(starts, n),
    whole(cursor, n),
    cfg,
  );
}

/// Plain-Rust twin of [`exclusive_scan`].
pub fn exclusive_scan_ref(values: &[u32]) -> Vec<u32> {
  let mut out = Vec::with_capacity(values.len());
  let mut running = 0u32;
  for v in values {
    out.push(running);
    running += v;
  }
  out
}
