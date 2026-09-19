//! The evolutionary metrics time series: one CSV row every `K` steps, and a
//! summary at exit derived from the same rows.
//!
//! From `docs/organism.md`'s *Metrics*: "lineage count and phylogenetic depth
//! over time; distributions of a few genome and body traits (root fraction,
//! height, leaf count) over time, and whether they bimodalize; population,
//! births, deaths, and energy flux." The traits are reported per soil column
//! ([`crate::soil::SoilGrid::columns`]) because the first experiment asks
//! whether they split by soil.
//!
//! Sampling is host-side. Every `K` steps it reads five particle fields back
//! (`state`, `density`, `organism`, `part_type`, `ppos`) and reads the
//! population's host mirrors in place — those are the master copy while
//! births are host-side, so nothing is downloaded for them. The life-cycle
//! chunk, which moves births onto the device, owns refreshing them before a
//! sample. The sampler only reads: a run with `--metrics` steps the same
//! world as one without.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use glam::Vec2;

use crate::genome::{PartType, Population};
use crate::particles::ParticleKind;
use crate::sim::{Counters, Sim};
use crate::soa::download_field;
use crate::soil::{ColumnExtent, SoilGrid};

/// One row of the time series.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
  pub step: u32,
  /// Milliseconds since the sampler was created, which is the start of the
  /// stepping loop.
  pub wall_ms: f64,
  pub fluid_liquid: usize,
  pub fluid_vapor: usize,
  /// Over fluid particles — liquid and vapor, never a body slot or a free
  /// one.
  pub mean_density: f32,
  pub max_density: f32,
  pub alive: usize,
  pub counters: Counters,
  /// Over alive organisms; all three are 0 when nothing is alive.
  pub energy_mean: f32,
  pub energy_min: f32,
  pub energy_max: f32,
  /// Distinct `lineage_id` among alive organisms.
  pub lineages: usize,
  pub generation_max: u32,
  pub generation_mean: f32,
  /// Clusters from [`cluster_species`], over the whole alive population.
  pub species: usize,
  /// One entry per [`crate::soil::SoilGrid::columns`] entry, in that order.
  pub columns: Vec<ColumnSample>,
}

/// The traits of the organisms anchored in one soil column.
///
/// The three per-organism means are over the column's alive organisms that
/// hold at least one body particle; a column with none reports zeros rather
/// than a NaN.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ColumnSample {
  pub alive: usize,
  /// Fraction of an organism's body particles that are root type, averaged
  /// over the column's organisms.
  pub root_frac: f32,
  /// Mean of (highest body particle y - anchor y).
  pub height: f32,
  pub leaf_count: f32,
  /// How many of the population's species are represented in this column.
  pub species: usize,
}

/// Writes the time series and keeps every row for the summary.
pub struct Sampler {
  path: PathBuf,
  every: u32,
  species_threshold: f32,
  columns: Vec<ColumnExtent>,
  out: BufWriter<File>,
  start: Instant,
  readback: Duration,
  samples: Vec<Sample>,
}

impl Sampler {
  /// Create the file and write the header row.
  ///
  /// The column set is fixed here from the terrain, so the header and every
  /// row that follows have the same width.
  pub fn create(
    path: &Path,
    every: u32,
    species_threshold: f32,
    soil: &SoilGrid,
  ) -> io::Result<Self> {
    let columns = soil.columns();
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "{}", header(&columns))?;
    Ok(Self {
      path: path.to_path_buf(),
      every: every.max(1),
      species_threshold,
      columns,
      out,
      start: Instant::now(),
      readback: Duration::ZERO,
      samples: Vec::new(),
    })
  }

  /// Whether `step` is one of the sampled steps. The final step of a run is
  /// sampled too, whether or not it lands on the cadence.
  pub fn is_sample_step(&self, step: u32) -> bool {
    step.is_multiple_of(self.every)
  }

  /// Read the world and write one row.
  pub fn sample<R: Runtime>(&mut self, sim: &Sim<R>) -> io::Result<()> {
    let before = Instant::now();
    let particles = read_back(sim);
    self.readback += before.elapsed();

    let sample = measure(
      sim,
      &particles,
      &self.columns,
      self.species_threshold,
      self.start.elapsed(),
    );
    writeln!(self.out, "{}", row(&sample))?;
    self.samples.push(sample);
    Ok(())
  }

  pub fn flush(&mut self) -> io::Result<()> {
    self.out.flush()
  }

  pub fn path(&self) -> &Path {
    &self.path
  }

  pub fn samples(&self) -> &[Sample] {
    &self.samples
  }

  /// Total time spent pulling particle fields back, across every sample.
  pub fn readback_time(&self) -> Duration {
    self.readback
  }

  /// The end-of-run summary, derived from the samples already written and
  /// never recomputed from the world.
  pub fn summary(&self) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let Some(last) = self.samples.last() else {
      let _ = writeln!(out, "\n=== Metrics ===\n  no samples taken");
      return out;
    };

    let mean_density = self
      .samples
      .iter()
      .map(|s| s.mean_density as f64)
      .sum::<f64>()
      / self.samples.len() as f64;
    let _ = writeln!(
      out,
      "\n=== Metrics ({} samples, last at step {}) ===",
      self.samples.len(),
      last.step
    );
    let _ = writeln!(
      out,
      "  alive={} lineages={} species={} generation max={}",
      last.alive, last.lineages, last.species, last.generation_max
    );
    let _ = writeln!(
      out,
      "  births={} deaths={}",
      last.counters.births, last.counters.deaths
    );
    let _ = writeln!(
      out,
      "  fluid mean density over all samples: {mean_density:.4}"
    );
    let _ = writeln!(
      out,
      "  {:<12} {:>5} {:>10} {:>9}",
      "column", "alive", "root_frac", "height"
    );
    for (extent, column) in self.columns.iter().zip(&last.columns) {
      let _ = writeln!(
        out,
        "  {:<12} {:>5} {:>10.4} {:>9.4}",
        extent.label, column.alive, column.root_frac, column.height
      );
    }
    let _ = writeln!(
      out,
      "  sampler readback: {:.3}ms total over {} samples",
      self.readback.as_secs_f64() * 1e3,
      self.samples.len()
    );
    out
  }
}

/// The header row: the fixed columns, then five per soil column named after
/// its label.
fn header(columns: &[ColumnExtent]) -> String {
  let mut names: Vec<String> = [
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
  .iter()
  .map(|s| s.to_string())
  .collect();
  for extent in columns {
    for field in ["alive", "root_frac", "height", "leaf_count", "species"] {
      names.push(format!("{field}_{}", extent.label));
    }
  }
  names.join(",")
}

fn row(s: &Sample) -> String {
  let mut cells = vec![
    s.step.to_string(),
    format!("{:.3}", s.wall_ms),
    s.fluid_liquid.to_string(),
    s.fluid_vapor.to_string(),
    format!("{:.6}", s.mean_density),
    format!("{:.6}", s.max_density),
    s.alive.to_string(),
    s.counters.births.to_string(),
    s.counters.deaths.to_string(),
    format!("{:.6}", s.energy_mean),
    format!("{:.6}", s.energy_min),
    format!("{:.6}", s.energy_max),
    s.lineages.to_string(),
    s.generation_max.to_string(),
    format!("{:.4}", s.generation_mean),
    s.species.to_string(),
  ];
  for column in &s.columns {
    cells.push(column.alive.to_string());
    cells.push(format!("{:.6}", column.root_frac));
    cells.push(format!("{:.6}", column.height));
    cells.push(format!("{:.6}", column.leaf_count));
    cells.push(column.species.to_string());
  }
  cells.join(",")
}

/// The five particle fields a sample needs, over the live prefix only — the
/// free slots above the body high-water mark hold nothing to measure.
struct Particles {
  state: Vec<ParticleKind>,
  density: Vec<f32>,
  organism: Vec<u32>,
  part_type: Vec<PartType>,
  ppos: Vec<Vec2>,
}

fn read_back<R: Runtime>(sim: &Sim<R>) -> Particles {
  let client = sim.client();
  let sph = sim.device_particles();
  let n = sim.live_particles().get();
  Particles {
    state: download_field(client, &sph.state, n),
    density: download_field(client, &sph.density, n),
    organism: download_field(client, &sph.organism, n),
    part_type: download_field(client, &sph.part_type, n),
    ppos: download_field(client, &sph.ppos, n),
  }
}

/// What one organism's particles add up to.
#[derive(Clone, Copy, Default)]
struct Body {
  particles: usize,
  roots: usize,
  leaves: usize,
  top_y: f32,
}

fn measure<R: Runtime>(
  sim: &Sim<R>,
  particles: &Particles,
  columns: &[ColumnExtent],
  species_threshold: f32,
  elapsed: Duration,
) -> Sample {
  let pop = sim.population();
  let soil = sim.soil();
  let anchors = &sim.bodies().anchors;

  // --- Fluid ---
  let (mut fluid_liquid, mut fluid_vapor) = (0usize, 0usize);
  let (mut density_sum, mut max_density) = (0.0f64, 0.0f32);
  let mut bodies = vec![Body::default(); pop.max_organisms];
  for i in 0..particles.state.len() {
    match particles.state[i] {
      ParticleKind::Liquid | ParticleKind::Vapor => {
        if particles.state[i] == ParticleKind::Liquid {
          fluid_liquid += 1;
        } else {
          fluid_vapor += 1;
        }
        density_sum += particles.density[i] as f64;
        max_density = max_density.max(particles.density[i]);
      }
      ParticleKind::Body => {
        let organism = particles.organism[i] as usize;
        let Some(body) = bodies.get_mut(organism) else {
          continue;
        };
        if body.particles == 0 {
          body.top_y = particles.ppos[i].y;
        }
        body.particles += 1;
        body.top_y = body.top_y.max(particles.ppos[i].y);
        match particles.part_type[i] {
          PartType::Root => body.roots += 1,
          PartType::Leaf => body.leaves += 1,
          _ => {}
        }
      }
      ParticleKind::Free => {}
    }
  }
  let fluid = fluid_liquid + fluid_vapor;
  let mean_density = if fluid > 0 {
    (density_sum / fluid as f64) as f32
  } else {
    0.0
  };

  // --- Population ---
  let alive: Vec<usize> = (0..pop.max_organisms)
    .filter(|o| pop.organisms.alive[*o] == 1)
    .collect();
  let (mut energy_sum, mut energy_min, mut energy_max) = (0.0f64, f32::INFINITY, f32::NEG_INFINITY);
  let mut lineages: Vec<u32> = Vec::new();
  let (mut generation_sum, mut generation_max) = (0u64, 0u32);
  for &o in &alive {
    let energy = pop.organisms.energy[o];
    energy_sum += energy as f64;
    energy_min = energy_min.min(energy);
    energy_max = energy_max.max(energy);
    let lineage = pop.organisms.lineage_id[o];
    if !lineages.contains(&lineage) {
      lineages.push(lineage);
    }
    generation_sum += pop.organisms.generation[o] as u64;
    generation_max = generation_max.max(pop.organisms.generation[o]);
  }
  let n = alive.len();
  let species_of = cluster_species(pop, &alive, species_threshold);
  let species = species_of.iter().copied().max().map_or(0, |c| c + 1);

  // --- Per column ---
  let mut column_samples = vec![ColumnSample::default(); columns.len()];
  let mut column_species: Vec<Vec<usize>> = vec![Vec::new(); columns.len()];
  let mut column_bodies = vec![0usize; columns.len()];
  for (k, &o) in alive.iter().enumerate() {
    let cell = soil.cell_column(anchors[o].x);
    let Some(c) = columns.iter().position(|column| column.contains(cell)) else {
      // A gap between two columns belongs to no column, so an organism
      // anchored there is counted in `alive` and in none of the per-column
      // figures.
      continue;
    };
    column_samples[c].alive += 1;
    if !column_species[c].contains(&species_of[k]) {
      column_species[c].push(species_of[k]);
    }
    let body = bodies[o];
    if body.particles == 0 {
      continue;
    }
    column_bodies[c] += 1;
    column_samples[c].root_frac += body.roots as f32 / body.particles as f32;
    column_samples[c].height += body.top_y - anchors[o].y;
    column_samples[c].leaf_count += body.leaves as f32;
  }
  for (c, column) in column_samples.iter_mut().enumerate() {
    column.species = column_species[c].len();
    let count = column_bodies[c];
    if count > 0 {
      column.root_frac /= count as f32;
      column.height /= count as f32;
      column.leaf_count /= count as f32;
    }
  }

  Sample {
    step: sim.step_count(),
    wall_ms: elapsed.as_secs_f64() * 1e3,
    fluid_liquid,
    fluid_vapor,
    mean_density,
    max_density,
    alive: n,
    counters: sim.counters(),
    energy_mean: if n > 0 {
      (energy_sum / n as f64) as f32
    } else {
      0.0
    },
    energy_min: if n > 0 { energy_min } else { 0.0 },
    energy_max: if n > 0 { energy_max } else { 0.0 },
    lineages: lineages.len(),
    generation_max,
    generation_mean: if n > 0 {
      generation_sum as f32 / n as f32
    } else {
      0.0
    },
    species,
    columns: column_samples,
  }
}

/// Greedy clustering of `alive` by [`crate::genome::species_distance`]: each
/// organism joins the first cluster whose representative it is within
/// `threshold` of, or starts a new cluster and becomes its representative.
///
/// Returns each entry of `alive`'s cluster index, so the caller can both
/// count the clusters and ask which ones a soil column holds. Order-dependent
/// by construction — the slot order is the population's, which is
/// deterministic, so two runs of one seed cluster identically.
pub fn cluster_species(pop: &Population, alive: &[usize], threshold: f32) -> Vec<usize> {
  let mut representatives: Vec<usize> = Vec::new();
  let mut clusters = Vec::with_capacity(alive.len());
  for &o in alive {
    let found = representatives
      .iter()
      .position(|&r| pop.species_distance(r, o) < threshold);
    match found {
      Some(c) => clusters.push(c),
      None => {
        representatives.push(o);
        clusters.push(representatives.len() - 1);
      }
    }
  }
  clusters
}

#[cfg(test)]
mod tests;
