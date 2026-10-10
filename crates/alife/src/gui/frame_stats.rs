//! Frames per second, frame time and sim steps per second for the controls
//! panel — the C++ HUD's `dt` / `fps` readout. That one printed the last
//! frame's interval, which jitters too fast to read; this averages over the
//! last second.

use super::rolling::Rolling;
use eframe::egui;
use std::time::{Duration, Instant};

pub(super) struct FrameStats {
  frames: u64,
  /// `(frames so far, sim step)` at each frame.
  history: Rolling<(u64, u32)>,
}

impl FrameStats {
  pub(super) fn new() -> Self {
    Self {
      frames: 0,
      history: Rolling::new(Duration::from_secs(1)),
    }
  }

  /// Count a frame. Called once a frame, after the frame's steps.
  pub(super) fn tick(&mut self, step: u32) {
    self.frames += 1;
    // A rebuilt world counts its steps from zero again.
    if self
      .history
      .newest()
      .is_some_and(|&(_, (_, last))| step < last)
    {
      self.history.clear();
    }
    self.history.push(Instant::now(), (self.frames, step));
  }

  pub(super) fn show(&self, ui: &mut egui::Ui) {
    let Some(((t0, (f0, s0)), (t1, (f1, s1)))) = self.history.ends() else {
      ui.label("— fps");
      return;
    };
    let secs = t1.duration_since(*t0).as_secs_f64().max(1e-9);
    let frames = (f1 - f0) as f64;
    ui.label(format!(
      "{:.1} fps  ·  {:.2} ms/frame  ·  {:.0} steps/s",
      frames / secs,
      secs * 1e3 / frames,
      f64::from(s1 - s0) / secs,
    ));
  }
}
