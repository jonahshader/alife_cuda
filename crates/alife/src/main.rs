//! The `alife` binary: headless runs and the GUI.
//!
//! Flag names match the C++ tree's `src/main.cu`, so `--help` of both lists
//! the same simulation flags.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use alife_sim::runtime::{AnySim, RuntimeKind, WgpuOptions};
use alife_sim::{SimParams, SimParamsCli, dump};
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
  let founders = sim.spawn_founders(cli.founders);
  if founders > 0 {
    println!("Seeded {founders} founder plants");
  }
  run_headless(sim, &cli)
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

  print!("{}", sim.timings().report());
  Ok(())
}
