//! A trailing window over snapshots of cumulative counters.
//!
//! The readouts it feeds (the profiler, the frame clock) want "the last
//! second", not "since the start": a live view should move when the sim
//! does. Every counter they read only grows, so the window's figure is the
//! newest snapshot minus the one at the window's start, and nothing has to be
//! summed per frame.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A snapshot and when it was taken.
pub(super) type Stamped<T> = (Instant, T);

pub(super) struct Rolling<T> {
  span: Duration,
  snaps: VecDeque<Stamped<T>>,
}

impl<T> Rolling<T> {
  pub(super) fn new(span: Duration) -> Self {
    Self {
      span,
      snaps: VecDeque::new(),
    }
  }

  pub(super) fn push(&mut self, at: Instant, value: T) {
    self.snaps.push_back((at, value));
    // The baseline is the newest snapshot at or before the window's start,
    // so once that much history exists the window spans at least `span`
    // rather than a little under it.
    while self.snaps.len() > 2 && at.duration_since(self.snaps[1].0) >= self.span {
      self.snaps.pop_front();
    }
  }

  pub(super) fn clear(&mut self) {
    self.snaps.clear();
  }

  pub(super) fn newest(&self) -> Option<&Stamped<T>> {
    self.snaps.back()
  }

  /// The baseline and the newest snapshot, once there are two.
  pub(super) fn ends(&self) -> Option<(&Stamped<T>, &Stamped<T>)> {
    match (self.snaps.front(), self.snaps.back()) {
      (Some(first), Some(last)) if self.snaps.len() >= 2 => Some((first, last)),
      _ => None,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn keeps_the_newest_snapshot_at_or_before_the_window_start() {
    let t0 = Instant::now();
    let ms = |n| t0 + Duration::from_millis(n);
    let mut w = Rolling::new(Duration::from_millis(1000));
    assert!(w.ends().is_none());
    w.push(ms(0), 0);
    assert!(w.ends().is_none());
    for (i, t) in [400, 800, 1200, 1600].into_iter().enumerate() {
      w.push(ms(t), i + 1);
    }
    // The window ends at 1600 and starts at 600: the baseline is 400, the
    // newest snapshot not after 600.
    let ((start, first), (end, last)) = w.ends().unwrap();
    assert_eq!((*start, *first), (ms(400), 1));
    assert_eq!((*end, *last), (ms(1600), 4));
  }

  #[test]
  fn a_long_gap_keeps_two_snapshots() {
    let t0 = Instant::now();
    let mut w = Rolling::new(Duration::from_millis(1000));
    w.push(t0, 0);
    w.push(t0 + Duration::from_secs(5), 1);
    w.push(t0 + Duration::from_secs(10), 2);
    let ((_, first), (_, last)) = w.ends().unwrap();
    assert_eq!((*first, *last), (1, 2));
  }
}
