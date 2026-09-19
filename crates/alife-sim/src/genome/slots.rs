//! Deterministic free-slot allocation.
//!
//! `docs/organism.md`: "births claim `Free` particle slots and free organism
//! slots in order via a prefix scan over the free flags, never via atomics."
//! An atomic claim would hand slot ids out in whatever order the units
//! happened to arrive in, and the run would stop being reproducible.
//!
//! The primitive is a compaction, not an organism thing: given a buffer of
//! occupancy flags, produce the indices where the flag is zero, ascending, and
//! their count. The organism chunk feeds it `alive`; the bodies chunk feeds it
//! a per-particle occupancy flag.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::kernels::scan::{ScanCfg, exclusive_scan, exclusive_scan_ref};
use crate::kernels::{CUBE_DIM, SCAN_THREADS, cube_count, whole};

/// Scratch and output buffers for one flag buffer's compaction.
pub struct SlotScan {
  /// 1 where the slot is claimable, 0 where it is occupied.
  pub free_flags: Handle,
  /// Per-block totals for the scan; [`SCAN_THREADS`] entries.
  pub partials: Handle,
  pub starts: Handle,
  /// The scan's second output, which this consumer has no use for.
  cursor: Handle,
  /// The claimable slot indices, ascending, in `[0, count)`.
  pub free_ids: Handle,
  /// One `u32`: how many entries of `free_ids` are live.
  pub count: Handle,
  pub len: usize,
}

impl SlotScan {
  pub fn alloc<R: Runtime>(client: &ComputeClient<R>, len: usize) -> Self {
    // `scatter_free` finds the total at entry `n - 1`, so an empty scan
    // would underflow that index before any unit could terminate.
    assert!(len >= 1, "a slot scan needs at least one slot");
    let words = len * size_of::<u32>();
    Self {
      free_flags: client.empty(words),
      partials: client.empty(SCAN_THREADS as usize * size_of::<u32>()),
      starts: client.empty(words),
      cursor: client.empty(words),
      free_ids: client.empty(words),
      count: client.empty(size_of::<u32>()),
      len,
    }
  }
}

#[cube(launch)]
pub fn mark_free(flags: &[u32], free_flags: &mut [u32], #[comptime] cfg: ScanCfg) {
  let i = ABSOLUTE_POS;
  if i >= comptime!(cfg.n as usize) {
    terminate!();
  }
  let mut free = 0u32;
  if flags[i] == 0u32 {
    free = 1u32;
  }
  free_flags[i] = free;
}

/// Compact the free indices. The exclusive scan already decided where each
/// one goes, so this writes without atomics and the result is ascending by
/// construction.
#[cube(launch)]
pub fn scatter_free(
  free_flags: &[u32],
  starts: &[u32],
  free_ids: &mut [u32],
  count: &mut [u32],
  #[comptime] cfg: ScanCfg,
) {
  let i = ABSOLUTE_POS;
  let n = comptime!(cfg.n as usize);
  if i >= n {
    terminate!();
  }
  if free_flags[i] == 1u32 {
    free_ids[starts[i] as usize] = i as u32;
  }
  // The total is the last entry's exclusive offset plus its own flag, so the
  // unit that owns the last entry is the one that knows it.
  if i == n - 1 {
    count[0] = starts[i] + free_flags[i];
  }
}

/// Fill `scan.free_ids` and `scan.count` from an occupancy flag buffer.
pub fn claim_free_slots<R: Runtime>(client: &ComputeClient<R>, flags: &Handle, scan: &SlotScan) {
  let cfg = ScanCfg { n: scan.len as u32 };
  mark_free::launch::<R>(
    client,
    cube_count(scan.len),
    CubeDim::new_1d(CUBE_DIM),
    whole(flags, scan.len),
    whole(&scan.free_flags, scan.len),
    cfg,
  );
  exclusive_scan(
    client,
    &scan.free_flags,
    &scan.partials,
    &scan.starts,
    &scan.cursor,
    cfg,
  );
  scatter_free::launch::<R>(
    client,
    cube_count(scan.len),
    CubeDim::new_1d(CUBE_DIM),
    whole(&scan.free_flags, scan.len),
    whole(&scan.starts, scan.len),
    whole(&scan.free_ids, scan.len),
    whole(&scan.count, 1),
    cfg,
  );
}

/// Read back the live part of `free_ids`.
pub fn read_free_slots<R: Runtime>(client: &ComputeClient<R>, scan: &SlotScan) -> Vec<u32> {
  let count = {
    let bytes = client.read_one_unchecked(scan.count.clone());
    bytemuck::cast_slice::<u8, u32>(&bytes)[0] as usize
  };
  let bytes = client.read_one_unchecked(scan.free_ids.clone());
  bytemuck::cast_slice::<u8, u32>(&bytes)[..count].to_vec()
}

/// Plain-Rust twin of [`claim_free_slots`] plus [`read_free_slots`].
pub fn free_slots_ref(flags: &[u32]) -> Vec<u32> {
  let free: Vec<u32> = flags.iter().map(|f| u32::from(*f == 0)).collect();
  let starts = exclusive_scan_ref(&free);
  let mut out = vec![
    0u32;
    starts.last().copied().unwrap_or(0) as usize
      + free.last().copied().unwrap_or(0) as usize
  ];
  for (i, f) in free.iter().enumerate() {
    if *f == 1 {
      out[starts[i] as usize] = i as u32;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::genome::population::Population;
  use crate::genome::{BrainShape, Genome};
  use crate::{SimParams, rng};
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  fn run(flags: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let client = CpuRuntime::client(&CpuDevice);
    let buf = client.create_from_slice(bytemuck::cast_slice(flags));
    let scan = SlotScan::alloc(&client, flags.len());
    claim_free_slots(&client, &buf, &scan);
    (read_free_slots(&client, &scan), free_slots_ref(flags))
  }

  #[test]
  fn kernel_matches_reference() {
    // A pattern wide enough to span several scan blocks, with runs of free
    // and occupied slots and both ends occupied.
    let n = 3000;
    let flags: Vec<u32> = (0..n)
      .map(|i| {
        let r = rng::threefry4x32_20_ref([i as u32, 0, 0, 0], [0xBEEF, 0, 0, 0]);
        u32::from(rng::u01_ref(r[0]) < 0.4)
      })
      .collect();
    let (actual, expected) = run(&flags);
    assert_eq!(actual, expected);
    assert!(!actual.is_empty() && actual.len() < n);
  }

  #[test]
  fn free_slots_come_out_ascending_and_complete() {
    let flags: Vec<u32> = (0..500).map(|i| u32::from(i % 3 == 0)).collect();
    let (actual, _) = run(&flags);
    assert!(actual.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(
      actual.len(),
      flags.iter().filter(|f| **f == 0).count(),
      "every free slot is claimed"
    );
    assert!(actual.iter().all(|i| flags[*i as usize] == 0));
  }

  #[test]
  fn all_free_and_none_free_both_work() {
    let (all, _) = run(&vec![0u32; 300]);
    assert_eq!(all, (0..300).collect::<Vec<u32>>());
    let (none, _) = run(&vec![1u32; 300]);
    assert!(none.is_empty());
  }

  #[test]
  fn a_population_hands_out_its_free_slots_in_order() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = SimParams {
      max_organisms: 8,
      ..SimParams::default()
    };
    let shape = BrainShape::from_params(&params);
    let mut pop = Population::new(&client, &params, shape);
    for slot in [0usize, 1, 4] {
      pop.write_genome(slot, &Genome::seed_plant(&shape, pop.max_limbs, 0, 1));
      pop.organisms.alive[slot] = 1;
    }
    pop.upload(&client);

    let scan = SlotScan::alloc(&client, pop.max_organisms);
    claim_free_slots(&client, &pop.device.organisms.alive, &scan);
    assert_eq!(read_free_slots(&client, &scan), vec![2, 3, 5, 6, 7]);
  }
}
