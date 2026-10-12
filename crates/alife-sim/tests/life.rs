//! The life cycle inside the whole sim, on every backend this box has.
//!
//! The kernel-versus-reference tests live next to the kernels; this is the
//! other half — that a population left to itself gains and spends energy,
//! drinks and transpires, sprouts, seeds, germinates and dies without leaking
//! a particle slot, an organism slot or a drop of water, and that the whole
//! thing is still bit-reproducible.

use std::sync::{Mutex, MutexGuard};

use alife_sim::SimParams;
use alife_sim::bodies::NO_PARTICLE;
use alife_sim::genome::PartType;
use alife_sim::genome::population::{STAGE_PLANT, STAGE_SEED};
use alife_sim::particles::ParticleKind;
use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions, available};

const RUNTIMES: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Cuda, RuntimeKind::Wgpu];

/// Two CubeCL backends coming up at once from two test threads segfaults
/// inside the runtimes, as `reproducibility.rs` notes.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialised() -> MutexGuard<'static, ()> {
  ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// A world small enough for the CPU runtime, with room above the soil, and a
/// life cadence fast enough that a few hundred steps is many ticks.
fn params() -> SimParams {
  SimParams {
    world_width: 8.0,
    world_height: 4.0,
    terrain_mode: 0,
    seed: 42,
    max_organisms: 16,
    life_interval: 5,
    // Deliberately generous, so a founder banks a seed's worth inside a few
    // hundred steps and the whole cycle runs on the CPU runtime in seconds.
    // The shipped defaults are tuned for a population over thousands of steps
    // in the full world (`docs/perf.md`); what this checks is the mechanism,
    // and eight founders in an 8 m world are too small a sample to survive
    // them.
    upkeep_per_volume: 3.0,
    seed_threshold: 1.0,
    seed_cost: 0.5,
    ..SimParams::default()
  }
}

const FOUNDERS: usize = 8;

fn run(kind: RuntimeKind, params: SimParams, steps: usize) -> AnySim {
  run_watching_growth(kind, params, steps).0
}

/// Step a fresh world, and say whether any limb was ever *part way* into a
/// segment.
///
/// Growth is continuous, but a limb that has reached its record's length
/// stands on a whole number of segments again, so the property has to be
/// watched over the run rather than read off the end of it.
fn run_watching_growth(kind: RuntimeKind, params: SimParams, steps: usize) -> (AnySim, bool) {
  let mut sim = AnySim::new(
    kind,
    params,
    params.resolve_seed(),
    None,
    &WgpuOptions::default(),
  )
  .expect("building the sim");
  assert_eq!(sim.spawn_founders(FOUNDERS), FOUNDERS);
  let mut part_way = false;
  for _ in 0..steps {
    sim.step();
    let bodies = sim.bodies();
    part_way |=
      (0..bodies.cfg.limb_count()).any(|i| bodies.grown[i] > 0.0 && bodies.grown[i].fract() != 0.0);
  }
  sim.sync();
  (sim, part_way)
}

/// Every invariant the bookkeeping has to keep, whatever the population did.
fn check_consistent(sim: &AnySim, name: &str) {
  let particles = sim.read_particles();
  let (pop, bodies) = (sim.population(), sim.bodies());
  let cfg = bodies.cfg;

  for i in 0..particles.len() {
    for v in [
      particles.pos[i].x,
      particles.pos[i].y,
      particles.vel[i].x,
      particles.vel[i].y,
      particles.density[i],
    ] {
      assert!(v.is_finite(), "{name}: particle {i} went non-finite");
    }
  }
  for o in 0..pop.max_organisms {
    assert!(
      pop.organisms.energy[o].is_finite(),
      "{name}: organism {o} energy is {}",
      pop.organisms.energy[o]
    );
  }

  // Every claimed slot holds a body particle of the organism that claimed it,
  // and every body particle is claimed by exactly one organism.
  let mut claimed: Vec<u32> = Vec::new();
  for o in 0..cfg.max_organisms as usize {
    for id in bodies.organism_particles(o) {
      assert_eq!(
        particles.state[id as usize],
        ParticleKind::Body,
        "{name}: organism {o} holds slot {id}, which is not a body particle"
      );
      assert_eq!(
        particles.organism[id as usize], o as u32,
        "{name}: slot {id} is claimed by {o} but says otherwise"
      );
      assert_eq!(
        pop.organisms.alive[o], 1,
        "{name}: dead organism {o} still holds slot {id}"
      );
      claimed.push(id);
    }
  }
  let before = claimed.len();
  claimed.sort_unstable();
  claimed.dedup();
  assert_eq!(before, claimed.len(), "{name}: a slot was claimed twice");
  assert_eq!(
    claimed.len(),
    particles.count_of(ParticleKind::Body),
    "{name}: a body particle belongs to no organism"
  );

  // The high-water mark still covers every claimed slot.
  if let Some(highest) = claimed.last() {
    assert!(
      bodies.high_water > *highest as usize,
      "{name}: high water {} misses slot {highest}",
      bodies.high_water
    );
  }
  // And the live prefix covers every occupied slot of any kind: water an
  // organism gave back or transpired can sit above the fluid range in no
  // organism's map, and a kernel launched short of it would never see it.
  let live = sim.geometry().fluid_particles.max(bodies.high_water);
  if let Some(top) = (0..particles.len())
    .rev()
    .find(|i| particles.state[*i] != ParticleKind::Free)
  {
    assert!(
      top < live,
      "{name}: slot {top} is {:?} but the live prefix ends at {live}",
      particles.state[top]
    );
  }

  // The water ledger closes: what came in is what went out plus what is
  // held, to float rounding over the run.
  let water = sim.water_balance();
  assert!(
    water.residual().abs() < 1e-3 * (water.endowed + water.drunk as f64).max(1.0),
    "{name}: the water ledger is off by {}: {water:?}",
    water.residual()
  );
  for o in 0..pop.max_organisms {
    assert!(
      pop.organisms.water[o] >= 0.0 && pop.organisms.transpired[o] >= 0.0,
      "{name}: organism {o} holds negative water"
    );
  }

  // A limb's length and the particles it holds are the same fact: the map
  // holds `ceil(grown)` of them, and a limb never outgrows the record it is
  // growing towards.
  let mp = cfg.max_particles_per_limb as usize;
  for o in 0..cfg.max_organisms as usize {
    for l in 0..cfg.max_limbs as usize {
      let grown = bodies.grown[cfg.limb_index(o, l)];
      assert_eq!(
        bodies.limb_len(o, l),
        grown.ceil() as usize,
        "{name}: organism {o} limb {l} has grown {grown} and holds {} particles",
        bodies.limb_len(o, l)
      );
      let record = pop.limb_index(o, l);
      let full = if pop.limbs.part_type[record].is_present() {
        (pop.limbs.length[record] as usize).min(mp)
      } else {
        0
      };
      assert!(
        grown <= full as f32,
        "{name}: organism {o} limb {l} grew to {grown} past a length of {full}"
      );
    }
  }

  // A seed is one particle of type Seed and nothing else; a plant has a root.
  for o in 0..pop.max_organisms {
    if pop.organisms.alive[o] == 0 {
      assert_eq!(bodies.seed_particles[o], NO_PARTICLE, "{name}: {o} is free");
      continue;
    }
    if pop.organisms.stage[o] == STAGE_SEED {
      let seed = bodies.seed_particles[o];
      assert_ne!(seed, NO_PARTICLE, "{name}: seed {o} has no particle");
      assert_eq!(particles.part_type[seed as usize], PartType::Seed);
      assert_eq!(
        bodies.organism_particles(o).len(),
        1,
        "{name}: seed {o} has a body"
      );
      assert!(
        pop.organisms.generation[o] >= 1,
        "{name}: a seed is a child"
      );
    } else {
      assert_eq!(pop.organisms.stage[o], STAGE_PLANT);
      assert_eq!(bodies.seed_particles[o], NO_PARTICLE);
    }
  }
}

#[test]
fn a_population_turns_over_on_its_own() {
  let _guard = serialised();
  let mut ran_any = false;

  for kind in RUNTIMES {
    if !available(kind) {
      eprintln!("skipping {}: not available on this box", kind.name());
      continue;
    }
    ran_any = true;
    let name = kind.name();
    // Long enough for founders to bank `seed_threshold`, seed, and for those
    // seeds to land and germinate.
    let (sim, grew_part_way) = run_watching_growth(kind, params(), 800);
    check_consistent(&sim, name);

    let counters = sim.counters();
    assert!(counters.births > 0, "{name}: nothing was ever born");
    assert!(counters.drinks > 0, "{name}: no root ever drank");
    assert!(counters.transpired > 0, "{name}: no leaf ever transpired");
    let pop = sim.population();
    let alive = (0..pop.max_organisms)
      .filter(|o| pop.organisms.alive[*o] == 1)
      .count();
    assert!(alive > 0, "{name}: the population died out");
    assert!(
      pop
        .organisms
        .generation
        .iter()
        .zip(&pop.organisms.alive)
        .any(|(g, a)| *a == 1 && *g >= 1),
      "{name}: no child is alive, so nothing descended from a founder"
    );
    // Growth is continuous: at some point in the run a limb was part way
    // into a segment rather than standing at a whole number of them.
    assert!(
      grew_part_way,
      "{name}: every limb was always a whole number of segments long, so nothing grew"
    );

    // A child's lineage is its parent's, and a founder's is its own slot.
    for o in 0..pop.max_organisms {
      if pop.organisms.alive[o] == 0 {
        continue;
      }
      assert!(
        pop.organisms.lineage_id[o] < FOUNDERS as u32,
        "{name}: organism {o} traces to no founder"
      );
    }
  }
  assert!(ran_any, "no runtime was available at all");
}

/// Energy is the only thing that kills a plant, so a world with no light
/// empties out — and empties cleanly, giving its water back.
#[test]
fn starvation_frees_every_slot_it_took() {
  let _guard = serialised();
  let params = SimParams {
    light_top: 0.0,
    // Little enough to starve on inside the run, and under the threshold,
    // so no founder seeds on its way out.
    seed_energy: 0.5,
    ..params()
  };
  let sim = run(RuntimeKind::Cpu, params, 600);
  check_consistent(&sim, "cpu");

  let pop = sim.population();
  let alive = (0..pop.max_organisms)
    .filter(|o| pop.organisms.alive[*o] == 1)
    .count();
  assert_eq!(alive, 0, "something survived with no light");
  assert_eq!(sim.counters().deaths, FOUNDERS as u64);
  assert_eq!(
    sim.read_particles().count_of(ParticleKind::Body),
    0,
    "a dead body left particles behind"
  );
  // Every particle that died left its organic matter in the soil.
  let deposited: f32 = sim.soil().cells.organic_matter.iter().sum();
  assert!(
    deposited > 0.0,
    "the dead bodies left nothing in the soil at all"
  );
  // Every dead plant gave its pool and its tissue back as liquid, less the
  // fraction of a particle each death drops.
  let water = sim.water_balance();
  assert!(sim.counters().returned > 0, "the dead gave no water back");
  assert_eq!(water.pools + water.tissue + water.pending, 0.0);
  assert!(water.leaked < FOUNDERS as f64, "{water:?}");
  // The body range holds no body any more, so the live prefix reaches only
  // as far as the returned water does: nothing above it is occupied.
  let particles = sim.read_particles();
  let live = sim.geometry().fluid_particles.max(sim.bodies().high_water);
  assert!(
    particles.state[live..]
      .iter()
      .all(|k| *k == ParticleKind::Free)
  );
}

#[test]
fn a_life_cycle_run_is_still_bit_reproducible() {
  let _guard = serialised();
  for kind in RUNTIMES {
    if !available(kind) {
      continue;
    }
    let first = run(kind, params(), 400);
    let second = run(kind, params(), 400);
    assert_eq!(
      first.read_particles(),
      second.read_particles(),
      "{}: two runs at one seed diverged",
      kind.name()
    );
    assert_eq!(first.counters(), second.counters());
    assert_eq!(
      first.population().organisms,
      second.population().organisms,
      "{}: the populations diverged",
      kind.name()
    );
  }
}
