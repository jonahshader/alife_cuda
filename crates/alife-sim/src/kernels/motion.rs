//! `evaporate_particles`, `move_particles` and `move_vapor_particles`: the
//! phase change and the two integrators.
//!
//! Each of these touches only its own particle, so unlike `calculate_accel`
//! they carry no read-write hazard and are ported as they stand.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::{
  Cfg, P_BOUNDS_X, P_BOUNDS_Y, P_COLLISION_DAMPING, P_CONDENSE_ALT_POWER, P_CONDENSE_RATE, P_DT,
  P_DT_PREDICT, P_EVAP_RATE, P_VAPOR_BUOYANCY, P_VAPOR_DRIFT, SphArgs,
};
use crate::particles::{ParticleKind, SphDevice, SphHost};
use crate::rng::{RngCounter, threefry4x32_20, threefry4x32_20_ref, u01, u01_ref};

/// Flip qualifying liquid particles to vapor state.
///
/// The Threefry counter arrives as four scalars rather than a buffer: it is
/// 16 bytes that change every step, and a buffer for it meant an allocation
/// and an upload per launch.
#[cube(launch)]
pub fn evaporate_particles(
  sph: &mut SphArgs,
  params: &[f32],
  ctr0: u32,
  ctr1: u32,
  ctr2: u32,
  ctr3: u32,
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != 0u32 {
    terminate!();
  }

  let prob = sph.evap_prob[i] * params[P_EVAP_RATE as usize] * params[P_DT as usize];
  if prob <= 0.0f32 {
    terminate!();
  }

  let result = threefry4x32_20(ctr0, ctr1, ctr2, ctr3, i as u32, 1u32, 0u32, 0u32);
  let roll = u01(result.x0);

  if roll < prob {
    sph.state[i] = 1u32;
    sph.vel[2 * i] = 0.0f32;
    sph.vel[2 * i + 1] = 0.5f32;
  }
}

#[cube(launch)]
pub fn move_particles(sph: &mut SphArgs, params: &[f32], #[comptime] cfg: Cfg) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != 0u32 {
    terminate!();
  }

  let dt = params[P_DT as usize];
  let dt_predict = params[P_DT_PREDICT as usize];
  let bounds_x = params[P_BOUNDS_X as usize];
  let bounds_y = params[P_BOUNDS_Y as usize];
  let damping = params[P_COLLISION_DAMPING as usize];

  let vel_x = sph.vel[2 * i];
  let vel_y = sph.vel[2 * i + 1];

  // use previous position to calculate new position
  let mut new_x = sph.ppos[2 * i] + vel_x * dt;
  let mut new_y = sph.ppos[2 * i + 1] + vel_y * dt;
  // wrap around
  new_x = (new_x + bounds_x) % bounds_x;

  // reflect y over bounds
  if new_y < 0.0f32 {
    new_y = -new_y;
    sph.vel[2 * i + 1] = -vel_y * damping;
  } else if new_y > bounds_y {
    new_y = (2.0f32 * bounds_y - new_y) - 1e-4f32;
    sph.vel[2 * i + 1] = -vel_y * damping;
  }

  // the prediction uses the pre-bounce velocity, as the C++ does
  let mut pred_x = new_x + vel_x * dt_predict;
  let mut pred_y = new_y + vel_y * dt_predict;
  // wrap around
  pred_x = (pred_x + bounds_x) % bounds_x;

  // reflect y over bounds. don't manipulate vel here
  if pred_y < 0.0f32 {
    pred_y = -pred_y;
  } else if pred_y > bounds_y {
    pred_y = (2.0f32 * bounds_y - pred_y) - 1e-4f32;
  }

  sph.ppos[2 * i] = new_x;
  sph.ppos[2 * i + 1] = new_y;
  sph.pos[2 * i] = pred_x;
  sph.pos[2 * i + 1] = pred_y;
}

/// Simple vapor physics: buoyancy, drift, condensation.
#[cube(launch)]
pub fn move_vapor_particles(
  sph: &mut SphArgs,
  params: &[f32],
  ctr0: u32,
  ctr1: u32,
  ctr2: u32,
  ctr3: u32,
  #[comptime] cfg: Cfg,
) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  if sph.state[i] != 1u32 {
    terminate!();
  }

  let dt = params[P_DT as usize];
  let bounds_x = params[P_BOUNDS_X as usize];
  let bounds_y = params[P_BOUNDS_Y as usize];

  let result = threefry4x32_20(ctr0, ctr1, ctr2, ctr3, i as u32, 2u32, 0u32, 0u32);
  let drift_x = (u01(result.x0) - 0.5f32) * 2.0f32 * params[P_VAPOR_DRIFT as usize];
  let condense_roll = u01(result.x1);

  let mut vel_x = sph.vel[2 * i];
  let mut vel_y = sph.vel[2 * i + 1];
  vel_y += params[P_VAPOR_BUOYANCY as usize] * dt;
  vel_x += drift_x * dt;
  let decay = f32::powf(0.99f32, dt * 600.0f32);
  vel_x *= decay;
  vel_y *= decay;

  let mut new_x = sph.ppos[2 * i] + vel_x * dt;
  let mut new_y = sph.ppos[2 * i + 1] + vel_y * dt;
  new_x = (new_x + bounds_x) % bounds_x;
  if new_y < 0.0f32 {
    new_y = 0.0f32;
  }
  if new_y > bounds_y {
    new_y = bounds_y;
  }

  // condensation: probability increases with altitude
  let normalized_alt = new_y / bounds_y;
  let condense_prob = f32::powf(normalized_alt, params[P_CONDENSE_ALT_POWER as usize])
    * params[P_CONDENSE_RATE as usize]
    * dt;

  if condense_roll < condense_prob {
    sph.state[i] = 0u32;
    vel_x *= 0.1f32;
    vel_y = -0.5f32;
    sph.evap_prob[i] = 0.0f32;
  }
  sph.vel[2 * i] = vel_x;
  sph.vel[2 * i + 1] = vel_y;

  // predict from the velocity actually stored (a condensed particle now falls)
  // and keep the prediction inside the world like move_particles does: a
  // condensed particle at the ceiling is read by the next grid build as a
  // liquid, and pos.y >= bounds.y would index one row past the grid
  let mut pred_x = new_x + vel_x * dt;
  let mut pred_y = new_y + vel_y * dt;
  pred_x = (pred_x + bounds_x) % bounds_x;
  if pred_y < 0.0f32 {
    pred_y = 0.0f32;
  }
  if pred_y > bounds_y - 1e-4f32 {
    pred_y = bounds_y - 1e-4f32;
  }

  sph.ppos[2 * i] = new_x;
  sph.ppos[2 * i + 1] = new_y;
  sph.pos[2 * i] = pred_x;
  sph.pos[2 * i + 1] = pred_y;
}

// --- Plain-Rust references ---

pub fn evaporate_particles_ref(
  particles: &mut SphHost,
  params: &crate::SimParams,
  ctr: RngCounter,
) {
  for i in 0..particles.len() {
    if particles.state[i] != ParticleKind::Liquid {
      continue;
    }
    let prob = particles.evap_prob[i] * params.evap_rate * params.dt;
    if prob <= 0.0 {
      continue;
    }
    let result = threefry4x32_20_ref(ctr.0, [i as u32, 1, 0, 0]);
    if u01_ref(result[0]) < prob {
      particles.state[i] = ParticleKind::Vapor;
      particles.vel[i] = glam::Vec2::new(0.0, 0.5);
    }
  }
}

pub fn move_particles_ref(
  particles: &mut SphHost,
  params: &crate::SimParams,
  geom: &crate::world::WorldGeometry,
) {
  let (bounds_x, bounds_y) = (geom.bounds.x, geom.bounds.y);
  for i in 0..particles.len() {
    if particles.state[i] != ParticleKind::Liquid {
      continue;
    }
    let vel = particles.vel[i];
    let mut new_pos = particles.ppos[i] + vel * params.dt;
    new_pos.x = (new_pos.x + bounds_x) % bounds_x;

    if new_pos.y < 0.0 {
      new_pos.y = -new_pos.y;
      particles.vel[i].y = -vel.y * params.collision_damping;
    } else if new_pos.y > bounds_y {
      new_pos.y = (2.0 * bounds_y - new_pos.y) - 1e-4;
      particles.vel[i].y = -vel.y * params.collision_damping;
    }

    let mut pred = new_pos + vel * params.dt_predict;
    pred.x = (pred.x + bounds_x) % bounds_x;
    if pred.y < 0.0 {
      pred.y = -pred.y;
    } else if pred.y > bounds_y {
      pred.y = (2.0 * bounds_y - pred.y) - 1e-4;
    }

    particles.ppos[i] = new_pos;
    particles.pos[i] = pred;
  }
}

pub fn move_vapor_particles_ref(
  particles: &mut SphHost,
  params: &crate::SimParams,
  geom: &crate::world::WorldGeometry,
  ctr: RngCounter,
) {
  let (bounds_x, bounds_y) = (geom.bounds.x, geom.bounds.y);
  for i in 0..particles.len() {
    if particles.state[i] != ParticleKind::Vapor {
      continue;
    }
    let result = threefry4x32_20_ref(ctr.0, [i as u32, 2, 0, 0]);
    let drift_x = (u01_ref(result[0]) - 0.5) * 2.0 * params.vapor_drift;
    let condense_roll = u01_ref(result[1]);

    let mut vel = particles.vel[i];
    vel.y += params.vapor_buoyancy * params.dt;
    vel.x += drift_x * params.dt;
    vel *= 0.99f32.powf(params.dt * 600.0);

    let mut new_pos = particles.ppos[i] + vel * params.dt;
    new_pos.x = (new_pos.x + bounds_x) % bounds_x;
    new_pos.y = new_pos.y.clamp(0.0, bounds_y);

    let normalized_alt = new_pos.y / bounds_y;
    let condense_prob =
      normalized_alt.powf(params.condense_altitude_power) * params.condense_rate * params.dt;

    if condense_roll < condense_prob {
      particles.state[i] = ParticleKind::Liquid;
      vel = glam::Vec2::new(vel.x * 0.1, -0.5);
      particles.evap_prob[i] = 0.0;
    }
    particles.vel[i] = vel;

    let mut pred = new_pos + vel * params.dt;
    pred.x = (pred.x + bounds_x) % bounds_x;
    pred.y = pred.y.clamp(0.0, bounds_y - 1e-4);

    particles.ppos[i] = new_pos;
    particles.pos[i] = pred;
  }
}

// --- Launch helpers ---

pub fn launch_evaporate<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  params: &Handle,
  ctr: RngCounter,
  cfg: Cfg,
) {
  evaporate_particles::launch::<R>(
    client,
    super::cube_count(cfg.num_particles as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::whole(params, super::PARAM_COUNT),
    ctr.0[0],
    ctr.0[1],
    ctr.0[2],
    ctr.0[3],
    cfg,
  );
}

pub fn launch_move<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  params: &Handle,
  cfg: Cfg,
) {
  move_particles::launch::<R>(
    client,
    super::cube_count(cfg.num_particles as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::whole(params, super::PARAM_COUNT),
    cfg,
  );
}

pub fn launch_move_vapor<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  params: &Handle,
  ctr: RngCounter,
  cfg: Cfg,
) {
  move_vapor_particles::launch::<R>(
    client,
    super::cube_count(cfg.num_particles as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    super::sph_args(sph),
    super::whole(params, super::PARAM_COUNT),
    ctr.0[0],
    ctr.0[1],
    ctr.0[2],
    ctr.0[3],
    cfg,
  );
}

#[cfg(test)]
mod tests {
  use super::super::test_support::{Harness, assert_close_vec2};
  use crate::particles::SphDevice;
  use crate::rng::RngCounter;

  /// Grade the surface signal so `evap_prob * evap_rate * dt` sweeps 0 to 2
  /// across the fixture: some particles never flip, some always do, and the
  /// ones in between exercise the actual roll.
  fn with_evap_signal(h: &mut Harness) {
    let n = h.particles.len() as f32;
    let certain = 2.0 / (h.params.evap_rate * h.params.dt);
    for i in 0..h.particles.len() {
      h.particles.evap_prob[i] = certain * i as f32 / n;
    }
    h.sph = SphDevice::upload(&h.client, &h.particles);
  }

  #[test]
  fn evaporate_matches_reference() {
    let mut h = Harness::new();
    with_evap_signal(&mut h);
    let ctr = RngCounter([3, 0, 0, 0]);

    super::launch_evaporate(&h.client, &h.sph, &h.params_buf, ctr, h.cfg);
    let actual = h.read_particles();

    let mut expected = h.particles.clone();
    super::evaporate_particles_ref(&mut expected, &h.params, ctr);

    assert_eq!(actual.state, expected.state);
    assert_close_vec2(&actual.vel, &expected.vel, 0.0, "vel after evaporation");
    assert!(
      expected.vapor_count() > h.particles.vapor_count(),
      "nothing evaporated, so the test proves nothing"
    );
  }

  #[test]
  fn move_matches_reference() {
    let h = Harness::new();
    super::launch_move(&h.client, &h.sph, &h.params_buf, h.cfg);
    let actual = h.read_particles();

    let mut expected = h.particles.clone();
    super::move_particles_ref(&mut expected, &h.params, &h.geom);

    assert_close_vec2(&actual.pos, &expected.pos, 0.0, "pos");
    assert_close_vec2(&actual.ppos, &expected.ppos, 0.0, "ppos");
    assert_close_vec2(&actual.vel, &expected.vel, 0.0, "vel");
  }

  #[test]
  fn move_vapor_matches_reference() {
    let h = Harness::new();
    let ctr = RngCounter([11, 0, 0, 0]);

    super::launch_move_vapor(&h.client, &h.sph, &h.params_buf, ctr, h.cfg);
    let actual = h.read_particles();

    let mut expected = h.particles.clone();
    super::move_vapor_particles_ref(&mut expected, &h.params, &h.geom, ctr);

    assert_eq!(actual.state, expected.state);
    assert_close_vec2(&actual.pos, &expected.pos, 1e-6, "pos");
    assert_close_vec2(&actual.ppos, &expected.ppos, 1e-6, "ppos");
    assert_close_vec2(&actual.vel, &expected.vel, 1e-6, "vel");
    assert!(h.particles.vapor_count() > 0, "no vapor in the fixture");
  }

  #[test]
  fn liquid_stays_inside_the_world() {
    let h = Harness::new();
    super::launch_move(&h.client, &h.sph, &h.params_buf, h.cfg);
    let moved = h.read_particles();
    for (i, pos) in moved.pos.iter().enumerate() {
      if moved.state[i] != crate::particles::ParticleKind::Liquid {
        continue;
      }
      assert!(
        pos.x >= 0.0 && pos.x < h.geom.bounds.x,
        "particle {i} left the world in x: {pos:?}"
      );
      assert!(
        pos.y >= 0.0 && pos.y <= h.geom.bounds.y,
        "particle {i} left the world in y: {pos:?}"
      );
    }
  }
}
