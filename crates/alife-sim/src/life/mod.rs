//! Energy and the life cycle: what an organism gains, what it spends, and
//! what happens when it runs out.
//!
//! From `docs/organism.md`'s *World coupling and life cycle*. One tick every
//! `life_interval` steps ([`crate::sim::Sim::life_tick`]), after the brain:
//! rebuild the light grid ([`light`]), charge and credit every organism
//! ([`energy`]), then decide sprouts, seeds, germinations and deaths on the
//! host from a small readback.

use cubecl::prelude::Runtime;

use crate::sim::{Sim, run_timed};

pub mod energy;
pub mod light;

impl<R: Runtime> Sim<R> {
  /// One life-cycle tick, every `life_interval` steps.
  pub fn life_tick(&mut self) {
    self.rebuild_light();
    self.charge_energy();
  }

  /// Rebuild the per-soil-cell light grid from the bodies standing in it.
  fn rebuild_light(&mut self) {
    let live = self.live_particles();
    let Self {
      client,
      sph,
      light,
      params_buf,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (cfg, timing) = (*cfg, *timing);
    run_timed(client, timing, timings, "light_grid", &mut || {
      light::launch(client, sph, light, params_buf, cfg, live);
    });
  }

  /// Credit every organism's gain and charge its upkeep for the interval.
  fn charge_energy(&mut self) {
    let Self {
      client,
      sph,
      bodies,
      pop,
      light,
      params_buf,
      body_cfg,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (body_cfg, cfg, timing) = (*body_cfg, *cfg, *timing);
    run_timed(client, timing, timings, "organism_energy", &mut || {
      energy::launch(client, sph, bodies, pop, light, params_buf, body_cfg, cfg);
    });
  }
}
