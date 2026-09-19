//! The simulation state and one step of it.
//!
//! The kernel order inside [`Sim::step`] is the order of the soil-coupled
//! `update_fluid(state, soil)` in the C++ tree's `src/systems/particle_fluid2.cu`.

use std::time::Instant;

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use crate::SimParams;
use crate::kernels::{self, accel, density, evap, grid::GridDevice, motion};
use crate::particles::{ParticleKind, SphDevice, SphHost};
use crate::rng::{RngCounter, threefry4x32_20_ref, u01_ref};
use crate::soil::{SoilDevice, SoilGrid, TerrainMode};
use crate::timing::{KernelTimings, TimingMethod};
use crate::world::{Cfg, WorldGeometry};

pub struct Sim<R: Runtime> {
  client: ComputeClient<R>,
  params: SimParams,
  geom: WorldGeometry,
  cfg: Cfg,
  sph: SphDevice,
  /// Velocities `calculate_accel` writes; swapped with `sph.vel` afterwards.
  vel_next: Handle,
  soil_device: SoilDevice,
  soil: SoilGrid,
  grid: GridDevice,
  params_buf: Handle,
  rng_counter: RngCounter,
  step_count: u32,
  seed: u64,
  timings: KernelTimings,
  timing: TimingChoice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimingChoice {
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
    let vel_next = client.empty(particles.len() * 2 * size_of::<f32>());
    let soil_device = SoilDevice::upload(&client, &soil.cells);
    let grid = GridDevice::alloc(&client, &cfg);
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
      params_buf,
      // Two launches consume a counter value per step: `evaporate_particles`
      // and `move_vapor_particles`, exactly as the C++ increments twice.
      rng_counter: RngCounter::after_steps(step_count),
      step_count,
      seed,
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
    let Self {
      client,
      sph,
      vel_next,
      soil_device,
      grid,
      params_buf,
      cfg,
      timings,
      timing,
      rng_counter,
      ..
    } = self;
    let cfg = *cfg;
    let timing = *timing;

    crate::soil::update_soil(&mut self.soil, self.params.dt);

    kernels::grid::build(client, sph, grid, params_buf, cfg, &mut |name, f| {
      run_timed(client, timing, timings, name, f);
    });

    run_timed(
      client,
      timing,
      timings,
      "calculate_particle_density",
      &mut || {
        density::launch(client, sph, grid, soil_device, params_buf, cfg);
      },
    );

    run_timed(client, timing, timings, "calculate_evap_prob", &mut || {
      evap::launch(client, sph, grid, soil_device, params_buf, cfg);
    });

    run_timed(client, timing, timings, "calculate_accel", &mut || {
      accel::launch(client, sph, vel_next, grid, soil_device, params_buf, cfg);
    });
    std::mem::swap(&mut sph.vel, vel_next);

    let ctr = *rng_counter;
    run_timed(client, timing, timings, "evaporate_particles", &mut || {
      motion::launch_evaporate(client, sph, params_buf, ctr, cfg);
    });
    rng_counter.incr();

    run_timed(client, timing, timings, "move_particles", &mut || {
      motion::launch_move(client, sph, params_buf, cfg);
    });

    let ctr = *rng_counter;
    run_timed(client, timing, timings, "move_vapor_particles", &mut || {
      motion::launch_move_vapor(client, sph, params_buf, ctr, cfg);
    });
    rng_counter.incr();

    self.step_count += 1;
  }
}

fn upload_params<R: Runtime>(
  client: &ComputeClient<R>,
  params: &SimParams,
  geom: &WorldGeometry,
) -> Handle {
  client.create_from_slice(bytemuck::cast_slice(&kernels::pack_params(params, geom)))
}

fn run_timed<R: Runtime>(
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
