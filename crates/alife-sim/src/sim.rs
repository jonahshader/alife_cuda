//! The simulation state and one step of it.
//!
//! The kernel order inside [`Sim::step`] is the order of the soil-coupled
//! `update_fluid(state, soil)` in the C++ tree's `src/systems/particle_fluid2.cu`.

use std::time::Instant;

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use crate::BrainShape;
use crate::SimParams;
use crate::bodies::{BodyCfg, BodyState, SimBodies};
use crate::brain::{BrainCfg, BrainState};
use crate::genome::Population;
use crate::kernels::{
  self, accel, constraints, density, evap, grid::GridDevice, limb_geometry, motion,
};
use crate::life::light::LightGrid;
use crate::life::pack::LifePack;
use crate::particles::{ParticleKind, SphDevice, SphHost};
use crate::rng::{RngCounter, threefry4x32_20_ref, u01_ref};
use crate::soil::{SoilDevice, SoilGrid, TerrainMode};
use crate::timing::{KernelTimings, TimingMethod};
use crate::world::{Cfg, WorldGeometry};

pub struct Sim<R: Runtime> {
  pub(crate) client: ComputeClient<R>,
  pub(crate) params: SimParams,
  pub(crate) geom: WorldGeometry,
  pub(crate) cfg: Cfg,
  pub(crate) sph: SphDevice,
  /// Velocities `calculate_accel` writes; swapped with `sph.vel` afterwards.
  pub(crate) vel_next: Handle,
  pub(crate) soil_device: SoilDevice,
  pub(crate) soil: SoilGrid,
  pub(crate) grid: GridDevice,
  /// The genome population. Empty until something seeds it: `--founders` at
  /// startup, and the life tick's seeds after that.
  pub(crate) pop: Population,
  /// Which particle holds which limb particle, plus the anchors.
  pub(crate) bodies: BodyState,
  pub(crate) body_cfg: BodyCfg,
  /// The brain's sensor buffer, its per-tick scratch and its outputs.
  pub(crate) brain: BrainState,
  /// Light reaching each soil cell, rebuilt every `life_interval` steps.
  pub(crate) light: LightGrid,
  /// Scratch for the life tick's one readback.
  pub(crate) life_pack: LifePack,
  pub(crate) params_buf: Handle,
  pub(crate) rng_counter: RngCounter,
  pub(crate) step_count: u32,
  pub(crate) seed: u64,
  pub(crate) counters: Counters,
  pub(crate) timings: KernelTimings,
  pub(crate) timing: TimingChoice,
}

/// Life-cycle events since this `Sim` was created, for the metrics time
/// series. Cumulative: nothing ever resets them, so a sample's delta against
/// the previous one is the rate over that interval.
///
/// Seeded founders are not births — they are the run's initial condition, not
/// something the life cycle produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
  pub births: u64,
  pub deaths: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimingChoice {
  /// Not measuring — the GUI path, where the launches should pipeline.
  Off,
  /// `ComputeClient::profile`, which uses device timestamps where it can.
  Profile,
  /// Wall clock around a synchronise, as the CubeCL spike measured.
  WallClock,
}

/// A world resumed from a dump rather than created fresh.
pub struct InitialState {
  pub particles: SphHost,
  /// Steps the world has already taken. The RNG counter is derived from it,
  /// so resuming a dump continues the same random stream rather than
  /// replaying it.
  pub step_count: u32,
}

impl<R: Runtime> Sim<R> {
  pub fn new(
    client: ComputeClient<R>,
    params: SimParams,
    seed: u64,
    initial: Option<InitialState>,
  ) -> Self {
    let mut geom = WorldGeometry::from_params(&params);
    let (mut particles, step_count) = match initial {
      Some(state) => {
        // A dump carries no organisms, so its body particles cannot be
        // resumed: the fluid it held becomes the fluid capacity and the body
        // slots are re-reserved empty on top.
        let fluid = state.particles.fluid_only();
        geom = geom.with_fluid_count(fluid.len());
        (fluid, state.step_count)
      }
      None => (initial_particles(&geom, &params, seed), 0),
    };
    particles.push_free_slots(geom.body_slots);
    debug_assert_eq!(particles.len(), geom.num_particles);
    let cfg = geom.cfg();

    let soil = SoilGrid::new(
      geom.soil_width,
      geom.soil_height,
      geom.soil_cell_size,
      TerrainMode::from_flag(params.terrain_mode),
      seed,
    );

    let sph = SphDevice::upload(&client, &particles);
    // Zeroed, not `empty`: `calculate_accel` runs over the live prefix only
    // and the whole buffer is swapped into `sph.vel` afterwards, so slots
    // above the prefix carry whatever this held. Nothing reads a `Free`
    // slot's velocity today, but uninitialised memory in a buffer that
    // reaches the dump would be a reproducibility trap waiting to spring.
    let vel_next =
      client.create_from_slice(bytemuck::cast_slice(&vec![0.0f32; particles.len() * 2]));
    let soil_device = SoilDevice::upload(&client, &soil.cells);
    let grid = GridDevice::alloc(&client, &cfg);
    let shape = BrainShape::from_params(&params);
    let pop = Population::new(&client, &params, shape);
    let bodies = BodyState::new(&client, &params, &geom);
    let body_cfg = bodies.cfg;
    let brain = BrainState::new(&client, BrainCfg::new(&shape, &params));
    let light = LightGrid::new(&client, &soil, params.light_top);
    let life_pack = LifePack::new(&client, body_cfg);
    let params_buf = upload_params(&client, &params, &geom);

    Self {
      client,
      params,
      geom,
      cfg,
      sph,
      vel_next,
      soil_device,
      soil,
      grid,
      pop,
      bodies,
      body_cfg,
      brain,
      light,
      life_pack,
      params_buf,
      // Two launches consume a counter value per step: `evaporate_particles`
      // and `move_vapor_particles`, exactly as the C++ increments twice.
      rng_counter: RngCounter::after_steps(step_count),
      step_count,
      seed,
      counters: Counters::default(),
      timings: KernelTimings::default(),
      timing: TimingChoice::Off,
    }
  }

  /// Turn per-kernel timing on. Measuring serialises the launches, which is
  /// why it is off by default; the C++ `TimingProfiler` has the same
  /// property.
  pub fn enable_timing(&mut self) {
    // One probe decides the method for the whole run, so a fallback never
    // lands in the middle of a run and mixes two scales.
    self.timing = match self.client.profile(|| (), "timing probe") {
      Ok(_) => TimingChoice::Profile,
      Err(err) => {
        tracing::warn!("device profiling unavailable ({err:?}); timing with wall clock");
        TimingChoice::WallClock
      }
    };
  }

  pub fn params(&self) -> &SimParams {
    &self.params
  }

  pub fn geometry(&self) -> &WorldGeometry {
    &self.geom
  }

  pub fn soil(&self) -> &SoilGrid {
    &self.soil
  }

  pub fn step_count(&self) -> u32 {
    self.step_count
  }

  pub fn seed(&self) -> u64 {
    self.seed
  }

  pub fn timings(&self) -> &KernelTimings {
    &self.timings
  }

  pub fn client(&self) -> &ComputeClient<R> {
    &self.client
  }

  pub fn device_particles(&self) -> &SphDevice {
    &self.sph
  }

  pub fn device_soil(&self) -> &SoilDevice {
    &self.soil_device
  }

  pub fn population(&self) -> &Population {
    &self.pop
  }

  pub fn bodies(&self) -> &BodyState {
    &self.bodies
  }

  pub fn body_cfg(&self) -> BodyCfg {
    self.body_cfg
  }

  pub fn brain(&self) -> &BrainState {
    &self.brain
  }

  /// The brain's per-limb outputs, read back to the host:
  /// `[max_organisms x max_limbs x HEAD_DIM]`, the sprout logits over child
  /// types followed by the two actuator outputs. Zero for an absent limb or a
  /// free organism slot.
  pub fn brain_outputs(&self) -> Vec<f32> {
    self.brain.read_heads(&self.client)
  }

  /// The limb geometry the last constraint pass published, read back to the
  /// host. The token features are built from it, so a reference
  /// implementation needs it as the device left it.
  pub fn read_limb_geometry(&self) -> crate::bodies::LimbGeometryHost {
    self.bodies.device.geometry.download(&self.client)
  }

  /// The persistent latent state, read back to the host:
  /// `[max_organisms x n_latents x d_latent]`.
  pub fn brain_latents(&self) -> Vec<f32> {
    crate::genome::population::read_f32(
      &self.client,
      &self.pop.device.latents,
      self.pop.max_organisms * self.pop.shape.latent_state_len(),
    )
  }

  /// Particle slots the per-particle kernels are launched over: the fluid
  /// plus however much of the body capacity has ever been claimed.
  pub fn live_particles(&self) -> kernels::LiveParticles {
    kernels::LiveParticles(self.geom.fluid_particles.max(self.bodies.high_water))
  }

  /// Organism slots currently occupied — seeds in flight included, because
  /// a seed holds its slot. The host mirror of `alive` is the master: the
  /// life tick writes it and uploads, and no kernel writes it back.
  pub fn organism_count(&self) -> usize {
    self.pop.organisms.alive.iter().filter(|a| **a == 1).count()
  }

  /// Cumulative births and deaths, which the metrics sampler records.
  pub fn counters(&self) -> Counters {
    self.counters
  }

  /// Count `n` births. A seed counts when it germinates, not when it is
  /// emitted: until it lands it has no body and no place in the population
  /// (`crate::life`).
  pub fn record_births(&mut self, n: u64) {
    self.counters.births += n;
  }

  /// Count `n` deaths; the other half of [`Self::record_births`].
  pub fn record_deaths(&mut self, n: u64) {
    self.counters.deaths += n;
  }

  /// Everything a body spawn touches, borrowed at once — `bodies::spawn` and
  /// `bodies::grow_limbs` need several of these fields together, and they are
  /// disjoint.
  pub fn body_access(&mut self) -> SimBodies<'_, R> {
    SimBodies {
      client: &self.client,
      params: &self.params,
      geom: &self.geom,
      cfg: self.cfg,
      sph: &self.sph,
      params_buf: &self.params_buf,
      pop: &mut self.pop,
      bodies: &mut self.bodies,
    }
  }

  /// Apply new parameters. Anything that changes the world's shape needs a
  /// new `Sim`; this is the slider path.
  pub fn set_params(&mut self, params: SimParams) {
    let geom = WorldGeometry::from_params(&params);
    assert_eq!(
      geom.cfg(),
      self.cfg,
      "changing the world's shape needs a new Sim, not new params"
    );
    self.params = params;
    self.params_buf = upload_params(&self.client, &self.params, &self.geom);
  }

  /// Pull the particle state back to the host.
  pub fn read_particles(&self) -> SphHost {
    self.sph.download(&self.client)
  }

  pub fn sync(&self) {
    pollster::block_on(self.client.sync()).expect("sync");
  }

  pub fn step(&mut self) {
    // Destructured so each timed closure borrows only what it launches
    // while the timings are borrowed mutably.
    // Host-side while births are: with no organism alive the two organism
    // passes are skipped outright rather than launched over zero units, so a
    // run without founders costs exactly what it did before they existed.
    let organisms = self.organism_count();
    let live = self.live_particles();

    let Self {
      client,
      sph,
      vel_next,
      soil_device,
      grid,
      pop,
      bodies,
      body_cfg,
      brain,
      light,
      params_buf,
      cfg,
      timings,
      timing,
      rng_counter,
      ..
    } = self;
    let cfg = *cfg;
    let body_cfg = *body_cfg;
    let timing = *timing;

    crate::soil::update_soil(&mut self.soil, self.params.dt);

    kernels::grid::build(client, sph, grid, params_buf, cfg, live, &mut |name, f| {
      run_timed(client, timing, timings, name, f);
    });

    run_timed(
      client,
      timing,
      timings,
      "calculate_particle_density",
      &mut || {
        density::launch(client, sph, grid, soil_device, params_buf, cfg, live);
      },
    );

    run_timed(client, timing, timings, "calculate_evap_prob", &mut || {
      evap::launch(client, sph, grid, soil_device, params_buf, cfg, live);
    });

    run_timed(client, timing, timings, "calculate_accel", &mut || {
      accel::launch(
        client,
        sph,
        vel_next,
        grid,
        soil_device,
        params_buf,
        cfg,
        live,
      );
    });
    std::mem::swap(&mut sph.vel, vel_next);

    let ctr = *rng_counter;
    run_timed(client, timing, timings, "evaporate_particles", &mut || {
      motion::launch_evaporate(client, sph, params_buf, ctr, cfg, live);
    });
    rng_counter.incr();

    run_timed(client, timing, timings, "move_particles", &mut || {
      motion::launch_move(client, sph, params_buf, cfg, live);
    });

    let ctr = *rng_counter;
    run_timed(client, timing, timings, "move_vapor_particles", &mut || {
      motion::launch_move_vapor(client, sph, params_buf, ctr, cfg, live);
    });
    rng_counter.incr();

    if organisms > 0 {
      run_timed(client, timing, timings, "project_constraints", &mut || {
        constraints::launch(client, sph, bodies, pop, params_buf, body_cfg, cfg);
      });
      run_timed(client, timing, timings, "write_limb_geometry", &mut || {
        limb_geometry::launch(
          client,
          sph,
          bodies,
          pop,
          params_buf,
          body_cfg,
          cfg.num_particles as usize,
        );
      });

      // Sense, then one brain tick. Nothing applies a head yet: the
      // life-cycle chunk reads the sprout logits and the actuator targets out
      // of `Sim::brain_outputs`.
      run_timed(client, timing, timings, "brain_sense", &mut || {
        crate::brain::sense::launch(
          client,
          sph,
          soil_device,
          bodies,
          pop,
          light,
          &brain.device,
          params_buf,
          brain.cfg,
          cfg,
        );
      });
      let weights = pop.weights();
      run_timed(client, timing, timings, "brain_tokens", &mut || {
        crate::brain::tokens::launch(
          client,
          &weights,
          bodies,
          pop,
          &brain.device,
          &pop.shape,
          brain.cfg,
        );
      });
      let shape = pop.shape;
      crate::brain::forward::launch(
        client,
        &weights,
        &brain.device,
        pop,
        &shape,
        brain.cfg,
        &mut |name, f| run_timed(client, timing, timings, name, f),
      );
    }

    self.step_count += 1;

    // The life cycle runs on its own cadence, after the brain: the sprout
    // head it reads was written by the tick that just finished. Skipped
    // outright when nothing is alive, so a run without organisms costs
    // exactly what it did before they existed.
    if organisms > 0
      && self
        .step_count
        .is_multiple_of(self.params.life_interval.max(1) as u32)
    {
      self.life_tick();
    }
  }
}

fn upload_params<R: Runtime>(
  client: &ComputeClient<R>,
  params: &SimParams,
  geom: &WorldGeometry,
) -> Handle {
  client.create_from_slice(bytemuck::cast_slice(&kernels::pack_params(params, geom)))
}

pub(crate) fn run_timed<R: Runtime>(
  client: &ComputeClient<R>,
  choice: TimingChoice,
  timings: &mut KernelTimings,
  name: &'static str,
  f: &mut (dyn FnMut() + Send),
) {
  match choice {
    TimingChoice::Off => f(),
    TimingChoice::WallClock => {
      let start = Instant::now();
      f();
      pollster::block_on(client.sync()).expect("sync");
      timings.record(name, start.elapsed(), TimingMethod::System);
    }
    TimingChoice::Profile => match client.profile(f, name) {
      Ok((_, duration)) => {
        let method = match duration.timing_method() {
          cubecl_common::profile::TimingMethod::Device => TimingMethod::Device,
          _ => TimingMethod::System,
        };
        let ticks = pollster::block_on(duration.resolve());
        timings.record(name, ticks.duration(), method);
      }
      Err(err) => panic!("profiling {name} failed: {err:?}"),
    },
  }
}

/// The initial fluid particles. The body slots are appended by the caller.
///
/// The C++ draws from `std::default_random_engine` with libstdc++'s own
/// distributions, which no other implementation reproduces, so this cannot be
/// and is not the same field at the same seed. Parity against the C++ is
/// therefore checked by loading a C++ dump (`--load`), not by seeding alike.
pub fn initial_particles(geom: &WorldGeometry, params: &SimParams, seed: u64) -> SphHost {
  let n = geom.fluid_particles;
  let mut particles = SphHost::new(n);
  let ctr = [seed as u32, (seed >> 32) as u32, 0, 0];

  for i in 0..n {
    // One block per particle: four uniforms, used as two gaussians (via
    // Box-Muller), a position and a symmetry-breaking direction.
    let a = threefry4x32_20_ref(ctr, [i as u32, 0xA11FE, 0, 0]);
    let b = threefry4x32_20_ref(ctr, [i as u32, 0xB0D1E, 0, 0]);

    let r = (-2.0f32 * u01_ref(a[0]).ln()).sqrt() * 0.001;
    let theta = std::f32::consts::TAU * u01_ref(a[1]);
    let vel = Vec2::new(r * theta.cos(), r * theta.sin());

    let pos = Vec2::new(
      u01_ref(a[2]) * geom.bounds.x,
      u01_ref(a[3]) * geom.bounds.y * 0.5,
    );

    particles.vel[i] = vel;
    particles.pos[i] = pos + vel * params.dt_predict;
    particles.ppos[i] = pos;
    particles.mass[i] = 1.0;
    particles.state[i] = ParticleKind::Liquid;
    particles.sym_break[i] = (b[0] % 256) as u8;
  }

  particles
}
