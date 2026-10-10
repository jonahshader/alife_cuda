//! The live profiler: per-kernel time over the last second, as bars.
//!
//! The GUI counterpart of the headless end-of-run report, after the C++
//! `ProfilerGui`. Timing serialises every launch, so the sim measures only
//! while this window is expanded; collapsing it turns timing back off.
//!
//! Each bar is a kernel's milliseconds *per step* over the window — its time
//! in the window divided by the steps in it — so the bars sum to the step
//! total, a kernel launched several times a step counts every launch, and
//! one that runs every `life_interval` steps shows its amortised share. The
//! per-launch average the headless report prints is in the bar's tooltip.
//!
//! The bars are not coloured. The C++ thresholds (green under 1 ms, yellow
//! at 5, red at 16.67 — one 60 FPS frame) are frame-budget figures, and every
//! kernel here is 10-350 µs (`docs/perf.md`), so every bar would be green
//! and the colour would say nothing.

use super::rolling::Rolling;
use alife_sim::sim::Sim;
use alife_sim::timing::{KernelTimings, TimingMethod};
use cubecl_wgpu::WgpuRuntime;
use eframe::egui;
use std::time::{Duration, Instant};

/// About a second of frames: long enough to steady the numbers, short enough
/// that a slider moved now shows in the bars now.
const WINDOW: Duration = Duration::from_secs(1);

/// Fixed rather than the available width, which an auto-sized window would
/// grow into.
const BAR_WIDTH: f32 = 360.0;

/// The cumulative counters at one moment.
struct Snapshot {
  step: u32,
  /// `(kernel, launches, total time)` in first-launch order.
  kernels: Vec<(&'static str, u64, Duration)>,
}

impl Snapshot {
  fn of(step: u32, timings: &KernelTimings) -> Self {
    Self {
      step,
      kernels: timings
        .iter()
        .map(|(name, stat)| (name, stat.samples, stat.total))
        .collect(),
    }
  }
}

struct Row {
  name: &'static str,
  per_step: Duration,
  launches: u64,
  per_launch: Duration,
}

/// What the window between two snapshots says.
struct Summary {
  steps: u32,
  span: Duration,
  rows: Vec<Row>,
}

impl Summary {
  fn between(old: &Snapshot, new: &Snapshot, span: Duration) -> Option<Self> {
    let steps = new.step.checked_sub(old.step).filter(|&s| s > 0)?;
    let rows = new
      .kernels
      .iter()
      .map(|&(name, launches, total)| {
        // A kernel first launched inside the window has no baseline entry.
        let (base_launches, base_total) = old
          .kernels
          .iter()
          .find(|k| k.0 == name)
          .map_or((0, Duration::ZERO), |k| (k.1, k.2));
        let launches = launches.saturating_sub(base_launches);
        let total = total.saturating_sub(base_total);
        Row {
          name,
          per_step: total / steps,
          launches,
          per_launch: if launches == 0 {
            Duration::ZERO
          } else {
            total.div_f64(launches as f64)
          },
        }
      })
      .collect();
    Some(Self { steps, span, rows })
  }

  fn total_per_step(&self) -> Duration {
    self.rows.iter().map(|r| r.per_step).sum()
  }
}

pub(super) struct Profiler {
  history: Rolling<Snapshot>,
}

impl Profiler {
  pub(super) fn new() -> Self {
    Self {
      history: Rolling::new(WINDOW),
    }
  }

  /// Draw the window and switch the sim's timing to match it: on while
  /// expanded, off while collapsed. Called once a frame, after the frame's
  /// steps.
  pub(super) fn show(&mut self, ctx: &egui::Context, sim: &mut Sim<WgpuRuntime>) {
    let now = Instant::now();
    if sim.timing_enabled() {
      let step = sim.step_count();
      // A paused sim adds nothing to the window, so it holds its last
      // reading rather than ageing towards empty.
      if self.history.newest().is_none_or(|(_, s)| s.step != step) {
        self.history.push(now, Snapshot::of(step, sim.timings()));
      }
    }

    let method = sim.timings().method();
    let expanded = egui::Window::new("Profiler")
      .default_open(false)
      .resizable(false)
      .show(ctx, |ui| self.draw(ui, method))
      .and_then(|response| response.inner)
      .is_some();

    // Checked every frame rather than on the click, so a rebuilt world —
    // which starts with timing off — is picked back up.
    match (expanded, sim.timing_enabled()) {
      (true, false) => {
        sim.enable_timing();
        // Steps taken while timing was off are in the step count but not in
        // the timings, so nothing from before this point is a baseline.
        self.history.clear();
        self
          .history
          .push(now, Snapshot::of(sim.step_count(), sim.timings()));
      }
      (false, true) => {
        sim.disable_timing();
        // Reopening starts a fresh window rather than flashing this one.
        self.history.clear();
      }
      _ => {}
    }
  }

  fn draw(&self, ui: &mut egui::Ui, method: Option<TimingMethod>) {
    ui.label(format!(
      "Timing: {}",
      method.map_or("starting…", TimingMethod::label)
    ));
    ui.weak("Measuring serialises the launches, so the sim runs slower while this is open.");
    ui.separator();

    let summary = self
      .history
      .ends()
      .and_then(|((t0, old), (t1, new))| Summary::between(old, new, t1.duration_since(*t0)));
    let Some(summary) = summary else {
      ui.label("Waiting for timed steps…");
      return;
    };

    ui.label(format!(
      "Total per step: {:.3} ms  ({} steps over {:.2} s)",
      ms(summary.total_per_step()),
      summary.steps,
      summary.span.as_secs_f64(),
    ));
    ui.add_space(4.0);

    let widest = summary
      .rows
      .iter()
      .map(|r| r.per_step)
      .max()
      .unwrap_or_default()
      .max(Duration::from_nanos(1));
    let bar_color = ui.visuals().selection.bg_fill;
    let text_color = ui.visuals().strong_text_color();
    let font = egui::FontId::monospace(12.0);
    for row in &summary.rows {
      let (rect, response) =
        ui.allocate_exact_size(egui::vec2(BAR_WIDTH, 18.0), egui::Sense::hover());
      let fraction = (row.per_step.as_secs_f64() / widest.as_secs_f64()) as f32;
      let bar = egui::Rect::from_min_size(
        rect.min + egui::vec2(0.0, 1.0),
        egui::vec2(rect.width() * fraction, rect.height() - 2.0),
      );
      let painter = ui.painter();
      painter.rect_filled(bar, 2.0, bar_color);
      painter.text(
        rect.left_center() + egui::vec2(4.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("{:>8.4} ms  {}", ms(row.per_step), row.name),
        font.clone(),
        text_color,
      );
      response.on_hover_text(format!(
        "{} launches in the window, {:.4} ms each",
        row.launches,
        ms(row.per_launch),
      ));
    }
  }
}

fn ms(d: Duration) -> f64 {
  d.as_secs_f64() * 1e3
}

#[cfg(test)]
mod tests {
  use super::*;

  fn snap(step: u32, kernels: &[(&'static str, u64, u64)]) -> Snapshot {
    Snapshot {
      step,
      kernels: kernels
        .iter()
        .map(|&(n, l, us)| (n, l, Duration::from_micros(us)))
        .collect(),
    }
  }

  #[test]
  fn the_window_is_the_difference_per_step() {
    let old = snap(10, &[("a", 10, 1000), ("b", 20, 400)]);
    // Ten steps later: `a` once a step, `b` twice, `c` new and launched once.
    let new = snap(20, &[("a", 20, 3000), ("b", 40, 600), ("c", 1, 50)]);
    let s = Summary::between(&old, &new, Duration::from_secs(1)).unwrap();
    let got: Vec<_> = s
      .rows
      .iter()
      .map(|r| {
        (
          r.name,
          r.per_step.as_micros(),
          r.launches,
          r.per_launch.as_micros(),
        )
      })
      .collect();
    assert_eq!(
      got,
      [("a", 200, 10, 200), ("b", 20, 20, 10), ("c", 5, 1, 50)]
    );
    assert_eq!(s.total_per_step(), Duration::from_micros(225));
  }

  #[test]
  fn no_steps_is_no_summary() {
    let a = snap(10, &[("a", 10, 1000)]);
    assert!(Summary::between(&a, &a, Duration::from_secs(1)).is_none());
    // A rebuilt world counts from zero again.
    let rebuilt = snap(3, &[("a", 3, 300)]);
    assert!(Summary::between(&a, &rebuilt, Duration::from_secs(1)).is_none());
  }
}
