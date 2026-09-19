//! Organisms stepping inside the whole sim, on every backend this box has.
//!
//! The kernel-versus-reference tests live next to the kernels; this is the
//! other half — that a body survives hundreds of steps of being pushed around
//! by the fluid without going numerically bad, drifting off its anchor or
//! stretching, and that the run is still bit-reproducible with organisms in
//! it.

use std::sync::{Mutex, MutexGuard};

use alife_sim::SimParams;
use alife_sim::bodies::NO_PARTICLE;
use alife_sim::particles::{ParticleKind, SphHost};
use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions, available};

const RUNTIMES: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];

/// Two CubeCL backends coming up at once from two test threads segfaults
/// inside the runtimes, as `parity.rs` found.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// A world small enough for the CPU runtime, with room above the soil for a
/// plant to stand in.
fn params() -> SimParams {
  SimParams {
    world_width: 8.0,
    world_height: 4.0,
    terrain_mode: 0,
    seed: 42,
    max_organisms: 8,
    // The life cycle off: what is under test here is the constraint pass
    // holding a body together over hundreds of steps, and a founder that
    // starved or sprouted halfway through would be testing `crate::life`
    // instead. `tests/life.rs` covers the two together.
    life_interval: 100_000,
    ..SimParams::default()
  }
}

const FOUNDERS: usize = 8;
const STEPS: usize = 200;

fn run(kind: RuntimeKind, steps: usize) -> AnySim {
  let params = params();
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

fn finite(particles: &SphHost) -> Result<(), String> {
  for i in 0..particles.len() {
    let values = [
      particles.pos[i].x,
      particles.pos[i].y,
      particles.ppos[i].x,
      particles.ppos[i].y,
      particles.vel[i].x,
      particles.vel[i].y,
      particles.mass[i],
      particles.density[i],
      particles.near_density[i],
      particles.evap_prob[i],
    ];
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
      return Err(format!(
        "particle {i} ({:?}) field {bad} is {}",
        particles.state[i], values[bad]
      ));
    }
  }
  Ok(())
}

#[test]
fn bodies_hold_their_shape_over_a_long_run() {
  let _guard = serialised();
  let mut ran_any = false;

  for kind in RUNTIMES {
    if !available(kind) {
      eprintln!("skipping {}: not available on this box", kind.name());
      continue;
    }
    ran_any = true;
    let sim = run(kind, STEPS);
    let particles = sim.read_particles();
    let name = kind.name();

    finite(&particles).unwrap_or_else(|why| panic!("{name}: {why}"));

    let (pop, bodies) = (sim.population(), sim.bodies());
    let cfg = bodies.cfg;
    let rest = sim.params().limb_segment_length;
    let mut segments = 0;

    for o in 0..FOUNDERS {
      assert_eq!(
        pop.organisms.alive[o], 1,
        "{name}: organism {o} is not alive"
      );
      let root = bodies.particle(o, 0, 0);
      assert_ne!(
        root, NO_PARTICLE,
        "{name}: organism {o} has no root particle"
      );
      let anchor = bodies.anchors[o];
      let drift = (particles.ppos[root as usize] - anchor).length();
      assert!(
        drift < 1e-3,
        "{name}: organism {o}'s root drifted {drift} off its anchor"
      );

      for limb in 0..cfg.max_limbs as usize {
        let record = pop.limb_index(o, limb);
        if !pop.limbs.part_type[record].is_present() {
          continue;
        }
        let n = (pop.limbs.length[record] as usize).min(cfg.max_particles_per_limb as usize);
        for i in 0..n {
          let id = bodies.particle(o, limb, i) as usize;
          assert_eq!(
            particles.state[id],
            ParticleKind::Body,
            "{name}: organism {o} limb {limb} particle {i} stopped being a body"
          );
          if i > 0 {
            let prev = bodies.particle(o, limb, i - 1) as usize;
            let len = (particles.ppos[id] - particles.ppos[prev]).length();
            assert!(
              (len / rest - 1.0).abs() <= 0.05,
              "{name}: organism {o} limb {limb} segment {i} is {len}, rest {rest}"
            );
            segments += 1;
          }
        }
      }
    }
    assert!(segments > 0, "{name}: no segment was checked");

    // The fluid is still a fluid around them.
    assert!(particles.liquid_count() > 0);
    assert_eq!(
      particles.count_of(ParticleKind::Body),
      bodies
        .limb_particles
        .iter()
        .filter(|id| **id != NO_PARTICLE)
        .count()
    );
  }
  assert!(ran_any, "no runtime was available at all");
}

#[test]
fn a_run_with_organisms_is_still_bit_reproducible() {
  let _guard = serialised();
  for kind in RUNTIMES {
    if !available(kind) {
      continue;
    }
    // Shorter than the shape test: what this checks is the constraint pass's
    // determinism, which shows up immediately or not at all.
    let first = run(kind, 40).read_particles();
    let second = run(kind, 40).read_particles();
    assert_eq!(
      first,
      second,
      "{}: two runs at one seed diverged",
      kind.name()
    );
  }
}
