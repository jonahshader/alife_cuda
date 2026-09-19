//! Per-kernel timing, printed at the end of a headless run.
//!
//! Stands in for the C++ `TimingProfiler`, which brackets each section with a
//! CUDA event pair. Here the measurement comes from `ComputeClient::profile`,
//! which uses device timestamps where the backend has them and falls back to
//! wall time around a sync otherwise; [`KernelTimings::method`] reports which
//! one a run actually used, because the two are not comparable.

use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMethod {
  /// Device timestamps around the launch.
  Device,
  /// Wall clock around a synchronise.
  System,
}

impl TimingMethod {
  pub fn label(self) -> &'static str {
    match self {
      TimingMethod::Device => "device timestamps",
      TimingMethod::System => "wall time around sync",
    }
  }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KernelStat {
  pub samples: u64,
  pub total: Duration,
  pub min: Option<Duration>,
  pub max: Option<Duration>,
}

impl KernelStat {
  pub fn average(&self) -> Duration {
    if self.samples == 0 {
      Duration::ZERO
    } else {
      self.total / self.samples as u32
    }
  }
}

/// Timings in first-launch order, which is the order within a step.
#[derive(Debug, Clone, Default)]
pub struct KernelTimings {
  order: Vec<&'static str>,
  stats: HashMap<&'static str, KernelStat>,
  method: Option<TimingMethod>,
}

impl KernelTimings {
  pub fn record(&mut self, name: &'static str, elapsed: Duration, method: TimingMethod) {
    self.method = Some(match self.method {
      // A run that ever fell back to wall time is a wall-time run.
      Some(TimingMethod::System) => TimingMethod::System,
      _ => method,
    });
    if !self.stats.contains_key(name) {
      self.order.push(name);
    }
    let stat = self.stats.entry(name).or_default();
    stat.samples += 1;
    stat.total += elapsed;
    stat.min = Some(stat.min.map_or(elapsed, |m| m.min(elapsed)));
    stat.max = Some(stat.max.map_or(elapsed, |m| m.max(elapsed)));
  }

  pub fn method(&self) -> Option<TimingMethod> {
    self.method
  }

  pub fn is_empty(&self) -> bool {
    self.order.is_empty()
  }

  pub fn iter(&self) -> impl Iterator<Item = (&'static str, KernelStat)> + '_ {
    self.order.iter().map(|name| (*name, self.stats[name]))
  }

  /// Total average time for one step: every kernel's average, summed.
  pub fn step_average(&self) -> Duration {
    self.iter().map(|(_, stat)| stat.average()).sum()
  }

  /// The same shape the C++ `print_profiler_stats` writes.
  pub fn report(&self) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    if self.is_empty() {
      return out;
    }
    let method = self.method.map(|m| m.label()).unwrap_or("unknown");
    let _ = writeln!(out, "\n=== Kernel timings ({method}) ===");
    for (name, stat) in self.iter() {
      let _ = writeln!(
        out,
        "  {name}: avg={:.6}ms min={:.6}ms max={:.6}ms samples={}",
        stat.average().as_secs_f64() * 1e3,
        stat.min.unwrap_or_default().as_secs_f64() * 1e3,
        stat.max.unwrap_or_default().as_secs_f64() * 1e3,
        stat.samples,
      );
    }
    let _ = writeln!(
      out,
      "  total per step: {:.6}ms",
      self.step_average().as_secs_f64() * 1e3
    );
    out
  }
}
