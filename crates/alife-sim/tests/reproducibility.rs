//! Same-seed reproducibility of the fluid and soil, on every backend this box
//! has. `bodies.rs` checks the same with organisms in the world.

use std::sync::{Mutex, MutexGuard};

use alife_sim::SimParams;
use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions, available};

const RUNTIMES: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];

/// Bringing two CubeCL backends up at once from two test threads segfaults
/// inside the runtimes, so the tests in this file take turns.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
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
