//! The mutation kernels on every backend this box has.
//!
//! The unit test in `genome::mutate` checks the kernels against their
//! plain-Rust twin on the CPU runtime and demands bit equality. This checks
//! the other half: that the three backends agree with each other and with the
//! reference. Box–Muller runs `ln`, `cos` and `sqrt`, and those are the
//! library functions a backend is most likely to round differently, so the
//! cross-backend comparison is a tolerance and the within-backend one is not.

use std::sync::{Mutex, MutexGuard};

use cubecl::prelude::*;
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_cuda::{CudaDevice, CudaRuntime};

use alife_sim::SimParams;
use alife_sim::genome::mutate::{Birth, MutateCfg, MutateInputs, mutate, mutate_ref};
use alife_sim::genome::population::{PartType, Population};
use alife_sim::genome::{BrainShape, Genome};
use alife_sim::runtime::{RuntimeKind, available};

/// Bringing two CubeCL backends up at once from two test threads segfaults
/// inside the runtimes, so the tests in this file take turns — the same rule
/// `tests/parity.rs` follows.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const STEP: u32 = 3;
const SEED: u64 = 0xA11FE;

fn params() -> SimParams {
  SimParams {
    max_organisms: 12,
    max_limbs: 8,
    max_particles_per_limb: 6,
    brain_d_token: 8,
    brain_d_latent: 8,
    brain_n_latents: 2,
    brain_trunk_hidden: 6,
    // Every birth makes a structural edit, so one run covers all three.
    structural_rate: 1.0,
    ..SimParams::default()
  }
}

fn births() -> Vec<Birth> {
  (4..10)
    .map(|child| Birth {
      child,
      parent: child % 4,
    })
    .collect()
}

/// Four founders, the last one with a deeper body so a removal has
/// descendants to cascade to.
fn population<R: Runtime>(client: &ComputeClient<R>, params: &SimParams) -> Population {
  let shape = BrainShape::from_params(params);
  let mut pop = Population::new(client, params, shape);
  for slot in 0..4u32 {
    let genome = Genome::seed_plant(&shape, pop.max_limbs, slot, 100 + slot as u64);
    pop.write_genome(slot as usize, &genome);
    pop.organisms.alive[slot as usize] = 1;
  }
  let base = pop.limb_index(3, 0);
  for (limb, (part, parent)) in [
    (PartType::Stem, 0u8),
    (PartType::Stem, 1),
    (PartType::Leaf, 2),
    (PartType::Seed, 1),
  ]
  .into_iter()
  .enumerate()
  {
    let i = base + 3 + limb;
    pop.limbs.part_type[i] = part;
    pop.limbs.parent[i] = parent;
    pop.limbs.length[i] = 1 + (limb as u8 % 3);
    pop.limbs.grow_angle[i] = 0.1 * limb as f32;
    pop.limbs.identity[i] = [limb as f32, 0.5, -0.25, 1.0];
  }
  pop.upload(client);
  pop
}

/// One mutation pass on `client`, returned as `(brain, grow_angle, identity,
/// part_type)` plus the reference the same inputs produce on the host.
fn run<R: Runtime>(client: &ComputeClient<R>) -> (Population, Population) {
  let params = params();
  let mut actual = population(client, &params);
  let mut expected = population(client, &params);
  let cfg = MutateCfg::new(&actual);
  let inputs = MutateInputs::upload(client, &births(), STEP, SEED, &params);
  mutate(client, &actual.device, &inputs, cfg);
  actual.download(client);
  mutate_ref(&mut expected, &births(), STEP, SEED, &params);
  (actual, expected)
}

fn max_float_diff(a: &Population, b: &Population) -> f64 {
  let mut worst = 0.0f64;
  for (x, y) in a.brain.iter().zip(&b.brain) {
    worst = worst.max((*x as f64 - *y as f64).abs());
  }
  for (x, y) in a.limbs.grow_angle.iter().zip(&b.limbs.grow_angle) {
    worst = worst.max((*x as f64 - *y as f64).abs());
  }
  for (x, y) in a.limbs.identity.iter().zip(&b.limbs.identity) {
    for (p, q) in x.iter().zip(y) {
      worst = worst.max((*p as f64 - *q as f64).abs());
    }
  }
  worst
}

fn assert_discrete_equal(a: &Population, b: &Population, what: &str) {
  assert_eq!(a.limbs.part_type, b.limbs.part_type, "{what}: part_type");
  assert_eq!(a.limbs.length, b.limbs.length, "{what}: length");
  assert_eq!(a.limbs.parent, b.limbs.parent, "{what}: parent");
  assert_eq!(a.limbs.child_slot, b.limbs.child_slot, "{what}: child_slot");
}

/// Largest absolute difference a backend may show against the host
/// reference. Everything so far lands at 0; this is loose enough that a
/// differently rounded `ln` or `cos` passes and a wrong kernel does not, since
/// the perturbation itself is `mutation_sigma` = 0.02.
const CROSS_RUNTIME_TOLERANCE: f64 = 1e-5;

#[test]
fn every_backend_agrees_with_the_reference_and_with_the_others() {
  let _guard = serialised();
  let mut results: Vec<(RuntimeKind, Population)> = Vec::new();

  {
    let client = CpuRuntime::client(&CpuDevice);
    let (actual, expected) = run(&client);
    assert_discrete_equal(&actual, &expected, "cpu");
    assert_eq!(max_float_diff(&actual, &expected), 0.0, "cpu is not exact");
    results.push((RuntimeKind::Cpu, actual));
  }

  if available(RuntimeKind::Cuda) {
    let client = CudaRuntime::client(&CudaDevice::new(0));
    let (actual, expected) = run(&client);
    assert_discrete_equal(&actual, &expected, "cuda");
    let diff = max_float_diff(&actual, &expected);
    println!("cuda vs reference: max abs diff {diff:e}");
    assert!(
      diff <= CROSS_RUNTIME_TOLERANCE,
      "cuda vs reference: {diff:e}"
    );
    results.push((RuntimeKind::Cuda, actual));
  } else {
    println!("cuda unavailable, skipped");
  }

  if available(RuntimeKind::Wgpu) {
    let setup = alife_sim::wgpu_backend::headless_setup(wgpu::Backends::PRIMARY, None).unwrap();
    let (_device, client) = alife_sim::wgpu_backend::client_on(&setup);
    let (actual, expected) = run(&client);
    assert_discrete_equal(&actual, &expected, "wgpu");
    let diff = max_float_diff(&actual, &expected);
    println!("wgpu vs reference: max abs diff {diff:e}");
    assert!(
      diff <= CROSS_RUNTIME_TOLERANCE,
      "wgpu vs reference: {diff:e}"
    );
    results.push((RuntimeKind::Wgpu, actual));
  } else {
    println!("wgpu unavailable, skipped");
  }

  // The structural edits are integer decisions from integer RNG words, so
  // they must match exactly whatever the float library does.
  for (kind, pop) in &results[1..] {
    assert_discrete_equal(pop, &results[0].1, kind.name());
    let diff = max_float_diff(pop, &results[0].1);
    println!("{} vs cpu: max abs diff {diff:e}", kind.name());
    assert!(diff <= CROSS_RUNTIME_TOLERANCE, "{}: {diff:e}", kind.name());
  }
  assert!(results.len() > 1, "only the CPU runtime ran");
}
