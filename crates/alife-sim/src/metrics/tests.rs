use std::path::PathBuf;

use cubecl_cpu::{CpuDevice, CpuRuntime};

use super::*;
use crate::SimParams;
use crate::bodies::spawn_founders;
use crate::genome::{BrainShape, Genome, PartType};
use crate::sim::Sim;

/// Fields the time series writes per soil column.
const PER_COLUMN: usize = 6;

/// A world narrow enough that the capillary test's `width / 40` gap is zero,
/// so its six columns tile the soil and every founder lands in one of them
/// (`soil::tests::a_narrow_world_has_no_gaps_between_its_columns`).
fn small_world() -> Sim<CpuRuntime> {
  let params = SimParams {
    world_width: 3.0,
    world_height: 2.0,
    smoothing_radius: 0.5,
    soil_cell_size: 0.1,
    particles_per_cell: 1,
    terrain_mode: 1,
    max_organisms: 4,
    max_particles_per_limb: 3,
    // The life cycle off for the length of this run: what is under test is
    // the writer, and a founder that starved or seeded halfway through would
    // be testing `crate::life` instead. `tests/life.rs` is where the two meet.
    life_interval: 100_000,
    ..SimParams::default()
  };
  Sim::new(CpuRuntime::client(&CpuDevice), params, 42, None)
}

fn temp_path(name: &str) -> PathBuf {
  let mut path = std::env::temp_dir();
  path.push(format!(
    "alife-metrics-{name}-{}-{:?}.csv",
    std::process::id(),
    std::thread::current().id()
  ));
  path
}

fn read_csv(path: &std::path::Path) -> (Vec<String>, Vec<Vec<String>>) {
  let text = std::fs::read_to_string(path).expect("the metrics file exists");
  let mut lines = text.lines();
  let header = lines
    .next()
    .expect("a header row")
    .split(',')
    .map(str::to_string)
    .collect();
  let rows = lines
    .map(|line| line.split(',').map(str::to_string).collect())
    .collect();
  (header, rows)
}

/// The whole writer: one run, two samplers at different cadences over it, so
/// the cadence and the trailing final-step row are both covered without
/// stepping twice.
#[test]
fn the_time_series_counts_three_founders_one_per_column() {
  let mut sim = small_world();
  assert_eq!(spawn_founders(&mut sim, 3), 3);

  let (on_cadence, off_cadence) = (temp_path("cadence"), temp_path("trailing"));
  let mut every_100 = Sampler::create(&on_cadence, 100, 0.25, sim.soil()).unwrap();
  let mut every_60 = Sampler::create(&off_cadence, 60, 0.25, sim.soil()).unwrap();

  for _ in 0..200 {
    sim.step();
    let step = sim.step_count();
    if every_100.is_sample_step(step) {
      every_100.sample(&sim).unwrap();
    }
    if every_60.is_sample_step(step) {
      every_60.sample(&sim).unwrap();
    }
  }
  // The final step is always sampled, cadence or not.
  if !every_100.is_sample_step(sim.step_count()) {
    every_100.sample(&sim).unwrap();
  }
  if !every_60.is_sample_step(sim.step_count()) {
    every_60.sample(&sim).unwrap();
  }
  every_100.flush().unwrap();
  every_60.flush().unwrap();

  let columns = sim.soil().columns();
  let (header, rows) = read_csv(&on_cadence);

  // The header names every column, in the documented order.
  assert_eq!(
    &header[..16],
    [
      "step",
      "wall_ms",
      "fluid_liquid",
      "fluid_vapor",
      "mean_density",
      "max_density",
      "alive",
      "births",
      "deaths",
      "energy_mean",
      "energy_min",
      "energy_max",
      "lineages",
      "generation_max",
      "generation_mean",
      "species",
    ]
  );
  assert_eq!(header.len(), 16 + PER_COLUMN * columns.len());
  for (c, extent) in columns.iter().enumerate() {
    let base = 16 + PER_COLUMN * c;
    assert_eq!(header[base], format!("alive_{}", extent.label));
    assert_eq!(header[base + 1], format!("root_frac_{}", extent.label));
    assert_eq!(header[base + 2], format!("height_{}", extent.label));
    assert_eq!(header[base + 3], format!("leaf_count_{}", extent.label));
    assert_eq!(header[base + 4], format!("energy_mean_{}", extent.label));
    assert_eq!(header[base + 5], format!("species_{}", extent.label));
  }

  // 200 steps at K=100: two samples, the last of which is the final step.
  assert_eq!(rows.len(), 2);
  assert_eq!(rows[0][0], "100");
  assert_eq!(rows[1][0], "200");
  for row in &rows {
    assert_eq!(row.len(), header.len(), "a row is the header's width");
  }

  // At K=60 the cadence misses the final step, which is appended.
  let (_, trailing) = read_csv(&off_cadence);
  assert_eq!(
    trailing.iter().map(|r| r[0].as_str()).collect::<Vec<_>>(),
    ["60", "120", "180", "200"]
  );

  let last = &rows[1];
  let field = |name: &str| -> &str {
    let i = header.iter().position(|h| h == name).expect(name);
    last[i].as_str()
  };
  assert_eq!(field("alive"), "3");
  // The three founders are each their own root ancestor.
  assert_eq!(field("lineages"), "3");
  // Founders are the initial condition, not births; nothing has died.
  assert_eq!(field("births"), "0");
  assert_eq!(field("deaths"), "0");
  assert_eq!(field("generation_max"), "0");
  assert_eq!(field("generation_mean"), "0.0000");
  // The founders differ only in their identity vectors, which at the default
  // `max_limbs` keeps them inside the default 0.25 threshold.
  for (a, b) in [(0, 1), (0, 2), (1, 2)] {
    assert!(sim.population().species_distance(a, b) < 0.25);
  }
  assert_eq!(field("species"), "1");
  assert_eq!(
    field("fluid_liquid").parse::<usize>().unwrap()
      + field("fluid_vapor").parse::<usize>().unwrap(),
    sim.geometry().fluid_particles
  );
  assert!(field("mean_density").parse::<f32>().unwrap() > 0.0);

  // One founder per column, and every founder accounted for.
  let alive_per_column: Vec<usize> = (0..columns.len())
    .map(|c| last[16 + PER_COLUMN * c].parse().unwrap())
    .collect();
  assert_eq!(alive_per_column.iter().sum::<usize>(), 3);
  assert!(
    alive_per_column.iter().all(|n| *n <= 1),
    "{alive_per_column:?}: a founder shares a column"
  );
  // The seed plant is 2 root, 3 stem, 1 leaf particles.
  for (c, n) in alive_per_column.iter().enumerate() {
    let root_frac: f32 = last[16 + PER_COLUMN * c + 1].parse().unwrap();
    let leaf_count: f32 = last[16 + PER_COLUMN * c + 3].parse().unwrap();
    let energy: f32 = last[16 + PER_COLUMN * c + 4].parse().unwrap();
    let species: usize = last[16 + PER_COLUMN * c + 5].parse().unwrap();
    if *n == 0 {
      assert_eq!(root_frac, 0.0);
      assert_eq!(energy, 0.0);
      assert_eq!(species, 0);
    } else {
      // The life cycle is off in this world, so a founder still holds exactly
      // the energy it was seeded with.
      assert!(
        (energy - sim.params().seed_energy).abs() < 1e-6,
        "column {c}: {energy}"
      );
      assert!(
        (root_frac - 2.0 / 6.0).abs() < 1e-5,
        "column {c}: {root_frac}"
      );
      assert_eq!(leaf_count, 1.0);
      assert_eq!(species, 1);
    }
  }

  // The summary is derived from the rows, not from the world.
  let summary = every_100.summary();
  assert!(summary.contains("alive=3 lineages=3 species=1 generation max=0"));
  assert!(summary.contains("2 samples"));

  let _ = std::fs::remove_file(&on_cadence);
  let _ = std::fs::remove_file(&off_cadence);
}

#[test]
fn a_run_with_no_organisms_writes_the_fluid_columns_and_zeros() {
  let mut sim = small_world();
  let path = temp_path("empty");
  let mut sampler = Sampler::create(&path, 5, 0.25, sim.soil()).unwrap();
  for _ in 0..5 {
    sim.step();
  }
  sampler.sample(&sim).unwrap();
  sampler.flush().unwrap();

  let (header, rows) = read_csv(&path);
  assert_eq!(rows.len(), 1);
  let row = &rows[0];
  let field = |name: &str| -> &str { &row[header.iter().position(|h| h == name).expect(name)] };
  assert_eq!(field("alive"), "0");
  assert_eq!(field("lineages"), "0");
  assert_eq!(field("species"), "0");
  assert_eq!(field("energy_mean"), "0.000000");
  assert!(field("fluid_liquid").parse::<usize>().unwrap() > 0);
  assert!(field("mean_density").parse::<f32>().unwrap() > 0.0);
  for cell in &row[16..] {
    assert!(
      cell.parse::<f32>().unwrap() == 0.0,
      "a per-column cell is not zero: {cell}"
    );
  }
  let _ = std::fs::remove_file(&path);
}

/// The noise terrain has no soil layout to split by, so it gets one column.
#[test]
fn another_terrain_gets_one_column_spanning_the_world() {
  let params = SimParams {
    world_width: 3.0,
    world_height: 2.0,
    smoothing_radius: 0.5,
    soil_cell_size: 0.1,
    particles_per_cell: 1,
    terrain_mode: 0,
    max_organisms: 4,
    max_particles_per_limb: 3,
    ..SimParams::default()
  };
  let mut sim = Sim::new(CpuRuntime::client(&CpuDevice), params, 42, None);
  assert_eq!(spawn_founders(&mut sim, 2), 2);
  let path = temp_path("noise");
  let mut sampler = Sampler::create(&path, 1, 0.25, sim.soil()).unwrap();
  sim.step();
  sampler.sample(&sim).unwrap();
  sampler.flush().unwrap();

  let (header, rows) = read_csv(&path);
  assert_eq!(header.len(), 16 + PER_COLUMN);
  assert_eq!(header[16], "alive_world");
  assert_eq!(rows[0][16], "2", "both founders are in the one column");
  let _ = std::fs::remove_file(&path);
}

/// A seed in flight holds an organism slot and has no anchor yet, so it is in
/// `alive` and in no column — rather than in the leftmost one, which is where
/// its `Vec2::ZERO` anchor would otherwise put it.
#[test]
fn a_seed_in_flight_counts_in_alive_and_in_no_column() {
  let mut sim = small_world();
  assert_eq!(spawn_founders(&mut sim, 3), 3);
  {
    // The fourth slot, as a newborn seed: alive, staged as a seed, energy and
    // anchor untouched. This is exactly what `life::apply` leaves behind
    // between the tick that emits a seed and the one that germinates it.
    let access = sim.body_access();
    access.pop.organisms.alive[3] = 1;
    access.pop.organisms.stage[3] = crate::genome::population::STAGE_SEED;
    access.pop.upload_organisms(access.client);
  }
  assert_eq!(sim.bodies().anchors[3], glam::Vec2::ZERO);

  let path = temp_path("seed-in-flight");
  let mut sampler = Sampler::create(&path, 1, 0.25, sim.soil()).unwrap();
  sim.step();
  sampler.sample(&sim).unwrap();
  sampler.flush().unwrap();

  let (header, rows) = read_csv(&path);
  let columns = sim.soil().columns();
  let field = |name: &str| -> &str { &rows[0][header.iter().position(|h| h == name).expect(name)] };
  assert_eq!(field("alive"), "4");
  let per_column: usize = (0..columns.len())
    .map(|c| rows[0][16 + PER_COLUMN * c].parse::<usize>().unwrap())
    .collect::<Vec<_>>()
    .iter()
    .sum();
  assert_eq!(per_column, 3, "the seed was binned into a column");
  // The three founders land in columns 1, 3 and 5 at this width, so the
  // leftmost column is empty — and stays empty, which is what the seed's
  // `Vec2::ZERO` anchor would otherwise break.
  assert_eq!(
    rows[0][16], "0",
    "a seed at the origin was binned into the leftmost column"
  );
  let _ = std::fs::remove_file(&path);
}

fn population_of(genomes: &[Genome], params: &SimParams, shape: &BrainShape) -> Population {
  let client = CpuRuntime::client(&CpuDevice);
  let mut pop = Population::new(&client, params, *shape);
  for (slot, genome) in genomes.iter().enumerate() {
    pop.write_genome(slot, genome);
    pop.organisms.alive[slot] = 1;
  }
  pop
}

#[test]
fn identical_organisms_are_one_species() {
  let params = SimParams {
    max_organisms: 3,
    ..SimParams::default()
  };
  let shape = BrainShape::from_params(&params);
  let genome = Genome::seed_plant(&shape, params.max_limbs as usize, 0, 7);
  let genomes = vec![genome.clone(), genome.clone(), genome];
  let pop = population_of(&genomes, &params, &shape);
  assert_eq!(cluster_species(&pop, &[0, 1, 2], 0.25), [0, 0, 0]);
}

#[test]
fn a_distance_above_the_threshold_starts_a_second_species() {
  let params = SimParams {
    max_organisms: 3,
    ..SimParams::default()
  };
  let shape = BrainShape::from_params(&params);
  let max_limbs = params.max_limbs as usize;
  let genome = Genome::seed_plant(&shape, max_limbs, 0, 7);
  let mut genomes = vec![genome.clone(), genome.clone(), genome];
  // Six added limbs: `6 / max_limbs` = 0.375, over the 0.25 threshold.
  for limb in 3..9 {
    genomes[2].limbs[limb].part_type = PartType::Stem;
    genomes[2].limbs[limb].length = 1;
  }
  let pop = population_of(&genomes, &params, &shape);
  assert!(pop.species_distance(0, 2) > 0.25);
  assert_eq!(cluster_species(&pop, &[0, 1, 2], 0.25), [0, 0, 1]);

  // A threshold above that distance collapses them back into one.
  assert_eq!(cluster_species(&pop, &[0, 1, 2], 1.0), [0, 0, 0]);
}
