//! Parity against the C++ binary, and same-seed reproducibility.
//!
//! The C++ is not bit-reproducible run to run — `populate_grid_indices`'s
//! atomics decide each cell's particle order, and `calculate_accel` races on
//! `sph.vel` — so parity is stated against that binary's own floor: after one
//! step from an identical state, the per-field difference between this port
//! and the C++ must be no worse than the difference between two C++ runs.
//!
//! The reference dumps are not checked in. Regenerating them is documented in
//! `crates/alife-sim/README.md`; without them these tests skip.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use alife_sim::particles::SphHost;
use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions, available};
use alife_sim::sim::InitialState;
use alife_sim::{SimParams, dump};

fn parity_dir() -> PathBuf {
  // crates/alife-sim -> workspace root -> target/parity
  Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../../target/parity")
    .canonicalize()
    .unwrap_or_else(|_| PathBuf::from("target/parity"))
}

const RUNTIMES: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];

/// Bringing two CubeCL backends up at once from two test threads segfaults
/// inside the runtimes, so the tests in this file take turns.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// How far past the C++ run-to-run floor a field may land.
///
/// Everything measures at 1.0x the floor except `evap_prob`, a sum of
/// gradient terms that nearly cancel; on the wgpu backend it sits at 3.2x,
/// from that compiler's own float contraction. A regression would miss these
/// by orders of magnitude, not by a factor of two.
fn budget_multiple(field: &str) -> f64 {
  match field {
    "evap_prob" => 5.0,
    _ => 3.0,
  }
}

/// Largest absolute difference per field, in `SphHost` declaration order.
fn field_diffs(a: &SphHost, b: &SphHost) -> Vec<(&'static str, f64)> {
  fn max_vec2(a: &[glam::Vec2], b: &[glam::Vec2]) -> f64 {
    a.iter()
      .zip(b)
      .map(|(x, y)| (*x - *y).abs().max_element() as f64)
      .fold(0.0, f64::max)
  }
  fn max_f32(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
      .zip(b)
      .map(|(x, y)| (x - y).abs() as f64)
      .fold(0.0, f64::max)
  }
  vec![
    ("pos", max_vec2(&a.pos, &b.pos)),
    ("ppos", max_vec2(&a.ppos, &b.ppos)),
    ("vel", max_vec2(&a.vel, &b.vel)),
    ("acc", max_vec2(&a.acc, &b.acc)),
    ("mass", max_f32(&a.mass, &b.mass)),
    ("density", max_f32(&a.density, &b.density)),
    ("near_density", max_f32(&a.near_density, &b.near_density)),
    ("evap_prob", max_f32(&a.evap_prob, &b.evap_prob)),
  ]
}

#[test]
fn one_step_from_a_cpp_dump_lands_at_the_cpp_run_to_run_floor() {
  let _guard = serialised();
  let dir = parity_dir();
  let (ref1, ref2a, ref2b) = (
    dir.join("ref1.bin"),
    dir.join("ref2a.bin"),
    dir.join("ref2b.bin"),
  );
  if !ref1.exists() || !ref2a.exists() || !ref2b.exists() {
    eprintln!(
      "skipping: no C++ reference dumps in {} (see crates/alife-sim/README.md)",
      dir.display()
    );
    return;
  }

  let (start_header, start) = dump::read(&ref1).expect("ref1.bin");
  let (_, expected) = dump::read(&ref2a).expect("ref2a.bin");
  // Two or three C++ runs of the same two steps: this is the noise we have to
  // beat. A third run makes the estimate less jumpy; `evap_prob`'s floor moves
  // by 30% between pairs.
  let mut runs = vec![dump::read(&ref2b).expect("ref2b.bin").1];
  let ref2c = dir.join("ref2c.bin");
  if ref2c.exists() {
    runs.push(dump::read(&ref2c).expect("ref2c.bin").1);
  }
  let mut floor = field_diffs(&expected, &runs[0]);
  for other in &runs {
    for (a, b) in [(&expected, other), (&runs[0], other)] {
      for (slot, (_, diff)) in floor.iter_mut().zip(field_diffs(a, b)) {
        slot.1 = slot.1.max(diff);
      }
    }
  }

  // The references are generated at `--terrain-mode 1 --seed 42`.
  let params = SimParams {
    terrain_mode: 1,
    seed: 42,
    ..SimParams::default()
  };

  let mut ran_any = false;
  for kind in RUNTIMES {
    if !available(kind) {
      eprintln!("skipping {}: not available on this box", kind.name());
      continue;
    }
    ran_any = true;

    let mut sim = AnySim::new(
      kind,
      params,
      params.resolve_seed(),
      Some(InitialState {
        particles: start.clone(),
        step_count: start_header.step_count,
      }),
      &WgpuOptions::default(),
    )
    .expect("building the sim");
    sim.step();
    sim.sync();

    let actual = sim.read_particles();
    for ((field, diff), (_, floor)) in field_diffs(&actual, &expected).iter().zip(&floor) {
      let budget = (floor * budget_multiple(field)).max(1e-9);
      assert!(
        *diff <= budget,
        "{}: {field} differs by {diff:.6e}, C++ run-to-run floor {floor:.6e}",
        kind.name(),
      );
    }

    // Evaporation is a coin flip per particle; the same stream has to produce
    // the same flips, so this one is exact rather than near.
    assert_eq!(
      actual.vapor_count(),
      expected.vapor_count(),
      "{}: vapor count",
      kind.name()
    );
  }
  assert!(ran_any, "no runtime was available at all");
}

#[test]
fn twenty_steps_twice_is_bit_identical() {
  let _guard = serialised();
  // Small world: reproducibility is about the grid build and the kernels, not
  // about the particle count, and this keeps the CPU runtime quick.
  let params = SimParams {
    world_width: 8.0,
    world_height: 4.0,
    terrain_mode: 1,
    seed: 42,
    ..SimParams::default()
  };

  for kind in RUNTIMES {
    if !available(kind) {
      eprintln!("skipping {}: not available on this box", kind.name());
      continue;
    }

    let run = || {
      let mut sim = AnySim::new(
        kind,
        params,
        params.resolve_seed(),
        None,
        &WgpuOptions::default(),
      )
      .expect("building the sim");
      for _ in 0..20 {
        sim.step();
      }
      sim.sync();
      sim.read_particles()
    };

    let first = run();
    let second = run();
    assert_eq!(
      first,
      second,
      "{}: two runs at one seed diverged",
      kind.name()
    );
    // A run that never moved would pass trivially.
    assert!(first.pos.iter().any(|p| p.y > 0.0));
  }
}
