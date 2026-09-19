//! The `alife` binary: headless runs and the GUI.
//!
//! Flag names match the C++ tree's `src/main.cu`, so `--help` of both lists
//! the same simulation flags.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions};
use alife_sim::{SimParams, SimParamsCli, dump, popdump};
use anyhow::{Context, Result};
use clap::Parser;

mod gui;

#[derive(Parser, Debug)]
#[command(
  name = "alife",
  version,
  about = "ALife: an SPH fluid and soil simulation"
)]
struct Cli {
  /// Path to TOML config file
  #[arg(long, default_value = "config.toml")]
  config: PathBuf,

  /// Write default config file and exit
  #[arg(long)]
  write_config: bool,

  /// Run without graphics
  #[arg(long)]
  headless: bool,

  /// Number of simulation steps to run (0 = unlimited)
  #[arg(long, default_value_t = 0)]
  iterations: u32,

  /// Write the particle state to a binary file at the end of a headless run
  #[arg(long, value_name = "PATH")]
  dump: Option<PathBuf>,

  /// Start from the particle state in a dump file instead of a fresh world
  #[arg(long, value_name = "PATH")]
  load: Option<PathBuf>,

  /// Seed this many founder plants, evenly spaced along the soil surface
  #[arg(long, default_value_t = 0, value_name = "N")]
  founders: usize,

  /// Write the population (genomes, lineage, energy, anchors) at the end of a
  /// headless run
  #[arg(long, value_name = "PATH")]
  save_pop: Option<PathBuf>,

  /// Start from a saved population instead of `--founders`; the bodies are
  /// re-grown at their anchors and seeds in flight are dropped
  #[arg(long, value_name = "PATH")]
  load_pop: Option<PathBuf>,

  /// Re-anchor every organism of one soil column in another, `<from>:<to>` by
  /// column label; repeatable, and all the moves are applied at once, so
  /// `--transplant sand:clay --transplant clay:sand` swaps the two. Needs
  /// `--load-pop`.
  #[arg(long, value_name = "FROM:TO")]
  transplant: Vec<String>,

  /// Write the evolutionary metrics time series to a CSV file (headless only)
  #[arg(long, value_name = "PATH")]
  metrics: Option<PathBuf>,

  /// Steps between metrics samples (at least 1)
  #[arg(long, default_value_t = 100, value_name = "K", value_parser = clap::value_parser!(u32).range(1..))]
  metrics_every: u32,

  /// Compute backend (default: cuda if available, else wgpu, else cpu)
  #[arg(long, value_enum)]
  runtime: Option<RuntimeKind>,

  /// Substring of the wgpu adapter name to select
  #[arg(long, value_name = "NAME")]
  adapter: Option<String>,

  /// List the wgpu adapters this machine offers and exit
  #[arg(long)]
  list_adapters: bool,

  #[command(flatten)]
  params: SimParamsCli,
}

fn main() -> Result<()> {
  tracing_subscriber::fmt()
    .with_env_filter(
      tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
    )
    .init();

  let cli = Cli::parse();

  if cli.write_config {
    std::fs::write(&cli.config, SimParams::default_config_text())
      .with_context(|| format!("writing {}", cli.config.display()))?;
    println!("Created default config at: {}", cli.config.display());
    return Ok(());
  }

  if cli.list_adapters {
    for info in alife_sim::wgpu_backend::list_adapters(wgpu::Backends::all()) {
      println!("{:?} {} ({:?})", info.device_type, info.name, info.backend);
    }
    return Ok(());
  }

  // Precedence: CLI > TOML > compiled default.
  let mut params = SimParams::default();
  if cli.config.exists() {
    let text = std::fs::read_to_string(&cli.config)
      .with_context(|| format!("reading {}", cli.config.display()))?;
    let table: toml::Table = text
      .parse()
      .with_context(|| format!("parsing {}", cli.config.display()))?;
    params.apply_toml(&table);
    println!("Loaded config from: {}", cli.config.display());
  }
  params.apply_cli(&cli.params);
  params
    .validate()
    .map_err(|msg| anyhow::anyhow!("invalid parameters: {msg}"))?;

  let initial = match &cli.load {
    Some(path) => {
      let (header, particles) =
        dump::read(path).with_context(|| format!("loading {}", path.display()))?;
      println!(
        "Loaded {} particles from {} (written after {} steps, seed {})",
        header.count,
        path.display(),
        header.step_count,
        header.seed,
      );
      Some(alife_sim::sim::InitialState {
        particles,
        step_count: header.step_count,
      })
    }
    None => None,
  };

  let wgpu_options = WgpuOptions {
    adapter: cli.adapter.clone(),
  };
  let seed = params.resolve_seed();

  if !cli.headless {
    // The GUI builds its own sim on the device the window gives it, so
    // nothing here starts a backend it would then throw away.
    return gui::run(params, initial, cli.runtime, cli.founders);
  }

  // Auto-selection builds the sim on the first backend that starts, rather
  // than probing for one and then starting it a second time.
  let mut sim = match cli.runtime {
    Some(kind) => AnySim::new(kind, params, seed, initial, &wgpu_options)?,
    None => AnySim::new_auto(params, seed, initial, &wgpu_options)?,
  };
  match &cli.load_pop {
    Some(path) => load_population(&mut sim, path, &cli.transplant)?,
    None => {
      if !cli.transplant.is_empty() {
        anyhow::bail!("--transplant needs a population to move: pass --load-pop as well");
      }
      let founders = sim.spawn_founders(cli.founders);
      if founders > 0 {
        println!("Seeded {founders} founder plants");
      }
    }
  }
  run_headless(sim, &cli)
}

/// `--load-pop`, and `--transplant` on top of it.
///
/// The moves are applied to the snapshot's anchors before the bodies are
/// grown, so a transplanted plant is laid out where it now stands rather than
/// being grown once and torn up again.
fn load_population(sim: &mut AnySim, path: &std::path::Path, moves: &[String]) -> Result<()> {
  let mut snapshot = popdump::read(path).with_context(|| format!("loading {}", path.display()))?;
  println!(
    "Loaded a population of {} plants from {} (written after {} steps, seed {})",
    snapshot.plants().len(),
    path.display(),
    snapshot.step_count,
    snapshot.seed,
  );

  let parsed: Vec<(&str, &str)> = moves
    .iter()
    .map(|spec| {
      spec
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("--transplant wants <from>:<to>, got `{spec}`"))
    })
    .collect::<Result<_>>()?;
  if !parsed.is_empty() {
    let moved = alife_sim::bodies::transplant_anchors(
      sim.soil(),
      &snapshot.organisms,
      &mut snapshot.anchors,
      &parsed,
    )
    .map_err(|msg| anyhow::anyhow!(msg))?;
    for ((from, to), slots) in parsed.iter().zip(&moved) {
      println!("Transplanted {} organisms from {from} to {to}", slots.len());
    }
  }

  let restored = sim
    .load_population(&snapshot)
    .context("installing the saved population")?;
  println!("Re-grew {restored} plants");
  Ok(())
}

fn run_headless(mut sim: AnySim, cli: &Cli) -> Result<()> {
  println!("Runtime: {}", sim.kind().name());
  println!("Using seed: {}", sim.seed());
  let geom = sim.geometry();
  println!(
    "World {} x {} m, grid {}x{} cells of {} m, {} fluid particles + {} body slots",
    geom.bounds.x,
    geom.bounds.y,
    geom.grid_width,
    geom.grid_height,
    geom.cell_size,
    geom.fluid_particles,
    geom.body_slots,
  );
  print!("Running headless");
  if cli.iterations > 0 {
    print!(" for {} iterations", cli.iterations);
  }
  println!(" (Ctrl+C to stop)");

  let stop = Arc::new(AtomicBool::new(false));
  {
    let stop = stop.clone();
    // A Ctrl+C in an unbounded run should still write the dump.
    ctrlc::set_handler(move || stop.store(true, Ordering::Relaxed))
      .context("installing the interrupt handler")?;
  }

  let mut metrics = match &cli.metrics {
    Some(path) => Some(
      alife_sim::metrics::Sampler::create(
        path,
        cli.metrics_every,
        sim.params().species_threshold,
        sim.soil(),
      )
      .with_context(|| format!("creating {}", path.display()))?,
    ),
    None => None,
  };

  let mut step = 0u32;
  while !stop.load(Ordering::Relaxed) {
    sim.step();
    step += 1;
    // Every kernel is JIT-compiled on its first launch — a few hundred
    // milliseconds on CUDA — so the first step is a warm-up and timing
    // starts after it.
    if step == 1 {
      sim.sync();
      sim.enable_timing();
    }
    if let Some(sampler) = &mut metrics
      && sampler.is_sample_step(sim.step_count())
    {
      sim.sample_metrics(sampler).context("sampling metrics")?;
    }
    if cli.iterations > 0 && step >= cli.iterations {
      break;
    }
  }
  sim.sync();

  println!("Completed {step} steps");

  if let Some(path) = &cli.dump {
    let particles = sim.read_particles();
    // The dump records the world's cumulative step count, which is what
    // `--load` needs to resume the RNG stream.
    dump::write(path, &particles, sim.step_count(), sim.seed())
      .with_context(|| format!("writing {}", path.display()))?;
    println!(
      "Wrote dump: {} ({} particles)",
      path.display(),
      particles.len()
    );
  }

  if let Some(path) = &cli.save_pop {
    sim
      .save_population(path)
      .with_context(|| format!("writing {}", path.display()))?;
    println!(
      "Wrote population: {} ({} organism slots taken)",
      path.display(),
      sim.organism_count()
    );
  }

  // The final step earns a row whether or not it landed on the cadence, so
  // the summary's "last row" is the state the run actually ended in.
  if let Some(sampler) = &mut metrics {
    if !sampler.is_sample_step(sim.step_count()) {
      sim.sample_metrics(sampler).context("sampling metrics")?;
    }
    sampler.flush().context("writing the metrics file")?;
    println!(
      "Wrote metrics: {} ({} samples)",
      sampler.path().display(),
      sampler.samples().len()
    );
  }

  print!("{}", sim.timings().report());
  if let Some(sampler) = &metrics {
    print!("{}", sampler.summary());
  }
  Ok(())
}
