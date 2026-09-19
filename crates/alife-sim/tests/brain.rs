//! The brain tick inside the whole sim, on every backend this box has.
//!
//! The kernel-versus-reference tests next to each kernel run on the CPU
//! runtime over a hand-laid fixture. This is the other half: the pass driven
//! by `Sim::step`, on a world the fluid has actually been moving, checked
//! against [`forward_population_ref`] on all three backends and for
//! run-to-run reproducibility on each.
//!
//! The tolerance is a relative one against the largest value in the field,
//! and is set per weight-storage choice: fp32 weights round only where a
//! backend contracts a multiply-add differently or evaluates `exp` its own
//! way, and fp16 weights start 2^-11 relative away from the master copy
//! before any accumulation.

use std::sync::{Mutex, MutexGuard};

use alife_sim::SimParams;
use alife_sim::brain::forward::forward_population_ref;
use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions, available};

const RUNTIMES: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];

/// Two CubeCL backends coming up at once from two test threads segfaults
/// inside the runtimes, as `parity.rs` found.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

const FOUNDERS: usize = 8;
/// Enough that the plants have been pushed around and the latents have
/// compounded over many ticks, short enough for the CPU runtime.
const STEPS: usize = 30;

/// fp32 weights: a relative tolerance a few ulp wide, which is what a
/// backend's own `exp` and its multiply-add contraction cost over
/// accumulation chains of 32 to 64 terms.
const TOL_F32: f32 = 2e-5;
/// fp16 weights: 2^-11 is the relative spacing of fp16 near 1, and a chain of
/// terms with mixed signs can land a couple of orders above that.
const TOL_F16: f32 = 2e-2;

fn params(fp16: bool) -> SimParams {
  SimParams {
    world_width: 8.0,
    world_height: 4.0,
    terrain_mode: 0,
    seed: 42,
    max_organisms: 8,
    brain_fp16: fp16 as i32,
    ..SimParams::default()
  }
}

fn run(kind: RuntimeKind, fp16: bool, steps: usize) -> AnySim {
  let params = params(fp16);
  let mut sim = AnySim::new(
    kind,
    params,
    params.resolve_seed(),
    None,
    &WgpuOptions::default(),
  )
  .expect("building the sim");
  assert_eq!(sim.spawn_founders(FOUNDERS), FOUNDERS);
  for _ in 0..steps {
    sim.step();
  }
  sim.sync();
  sim
}

/// Largest difference relative to the largest magnitude in the expected
/// field, which is [`alife_sim::kernels::test_support`]'s rule for a
/// kernel-versus-reference comparison.
fn relative_deviation(actual: &[f32], expected: &[f32]) -> f32 {
  assert_eq!(actual.len(), expected.len());
  let scale = expected.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
  actual
    .iter()
    .zip(expected)
    .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()))
    / scale
}

/// Step once more and compare that tick against the reference, starting from
/// the latents the device had going into it.
fn deviation_against_reference(sim: &mut AnySim) -> (f32, f32) {
  let mut latents = sim.brain_latents();
  sim.step();
  sim.sync();

  let sensors = sim.brain_sensors();
  let geometry = sim.read_limb_geometry();
  let expected_heads = forward_population_ref(sim.population(), &geometry, &sensors, &mut latents);

  (
    relative_deviation(&sim.brain_latents(), &latents),
    relative_deviation(&sim.brain_outputs(), &expected_heads),
  )
}

fn check(fp16: bool, tolerance: f32) {
  let _guard = serialised();
  let mut ran_any = false;

  for kind in RUNTIMES {
    if !available(kind) {
      eprintln!("skipping {}: not available on this box", kind.name());
      continue;
    }
    ran_any = true;
    let mut sim = run(kind, fp16, STEPS);
    let (latents, heads) = deviation_against_reference(&mut sim);
    eprintln!(
      "{} fp{}: max relative deviation, latents {latents:.3e}, heads {heads:.3e}",
      kind.name(),
      if fp16 { 16 } else { 32 },
    );
    assert!(
      latents <= tolerance && heads <= tolerance,
      "{}: latents {latents} heads {heads} exceed {tolerance}",
      kind.name()
    );
    assert!(
      sim.brain_outputs().iter().all(|v| v.is_finite()),
      "{}: a head is not finite",
      kind.name()
    );
  }
  assert!(ran_any, "no runtime was available at all");
}

#[test]
fn fp32_weights_match_the_reference_on_every_backend() {
  check(false, TOL_F32);
}

#[test]
fn fp16_weights_match_the_reference_on_every_backend() {
  check(true, TOL_F16);
}

/// Exact within one backend, tolerance across them: the rule
/// `docs/organism.md` sets for a kernel that calls a transcendental. The
/// brain calls `exp` in the softmax, the SiLU and the gate.
#[test]
fn the_pass_is_bit_reproducible_on_each_backend() {
  let _guard = serialised();
  for kind in RUNTIMES {
    if !available(kind) {
      continue;
    }
    let first = run(kind, false, 20);
    let second = run(kind, false, 20);
    assert_eq!(
      first.brain_latents(),
      second.brain_latents(),
      "{}: two runs at one seed left different latents",
      kind.name()
    );
    assert_eq!(
      first.brain_outputs(),
      second.brain_outputs(),
      "{}: two runs at one seed left different heads",
      kind.name()
    );
  }
}

/// The tick makes three unnormalised residual adds onto the latent stream,
/// so before `latent_norm` existed the latents grew by about 2.6x per step
/// and overflowed to infinity by step 45 at `--founders 256`. This is the
/// regression test: every latent vector leaves a tick at unit RMS, whatever
/// the tick did, and nothing downstream goes non-finite.
#[test]
fn the_latents_stay_at_unit_rms_over_a_long_run() {
  let _guard = serialised();
  let sim = run(RuntimeKind::Cpu, false, 200);
  let latents = sim.brain_latents();
  let d = sim.population().shape.d_latent;

  let mut checked = 0;
  for (o, alive) in sim.population().organisms.alive.iter().enumerate() {
    if *alive == 0 {
      continue;
    }
    for m in 0..sim.population().shape.n_latents {
      let base = o * sim.population().shape.latent_state_len() + m * d;
      let vector = &latents[base..base + d];
      assert!(
        vector.iter().all(|v| v.is_finite()),
        "organism {o} latent {m} is not finite"
      );
      let rms = (vector.iter().map(|v| v * v).sum::<f32>() / d as f32).sqrt();
      assert!(
        (rms - 1.0).abs() < 1e-3,
        "organism {o} latent {m} has RMS {rms}"
      );
      checked += 1;
    }
  }
  assert!(checked > 0, "no live organism to check");
  assert!(sim.brain_outputs().iter().all(|v| v.is_finite()));
}

/// With nothing alive the pass never runs, so every buffer it owns stays as
/// it was allocated. The other half of this — that the fluid is byte-for-byte
/// what it was before brains existed — is a dump diff against a `main` build,
/// recorded in `docs/perf.md`.
#[test]
fn a_world_with_no_organisms_has_no_brain_state() {
  let _guard = serialised();
  let params = params(false);
  let mut sim = AnySim::new(
    RuntimeKind::Cpu,
    params,
    params.resolve_seed(),
    None,
    &WgpuOptions::default(),
  )
  .expect("building the sim");
  assert_eq!(sim.organism_count(), 0);
  for _ in 0..5 {
    sim.step();
  }
  sim.sync();
  assert!(sim.brain_outputs().iter().all(|v| *v == 0.0));
  assert!(sim.brain_latents().iter().all(|v| *v == 0.0));
  assert!(sim.brain_sensors().iter().all(|v| *v == 0.0));
}
