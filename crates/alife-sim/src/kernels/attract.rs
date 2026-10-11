//! `attract_single_kernel`: the GUI's mouse tool. Every particle in the grid
//! within `radius` of a world point gets `max_thrust` added to its velocity,
//! towards the point — or away from it, with a negative thrust.
//!
//! It is GUI-driven state, not a sim parameter: the window calls it once per
//! step while a mouse button is held, after the step, which is where the C++
//! `FluidSoil::update` applied it. Headless never calls it, so a run without
//! the window is the same with or without this kernel.
//!
//! One unit does all of it, as in the C++. The gather is a few hundred
//! particles once a frame, and a single unit is free of write races even when
//! the radius is wider than half the world and one particle is reached
//! through two of its images — each pull lands in turn.

use cubecl::prelude::*;

use super::grid::{GridDevice, GridRef};
use super::{Cfg, GridArgs, P_BOUNDS_X, P_CELL_SIZE, SphArgs, particle_to_cid};
use crate::particles::{SphDevice, SphHost};

/// Push the particles within `radius` of `(pos_x, pos_y)` by `max_thrust`
/// along the direction to the point. Launched on a single unit.
#[allow(clippy::too_many_arguments)]
#[cube(launch)]
pub fn attract_single(
  sph: &mut SphArgs,
  grid: &GridArgs,
  params: &[f32],
  pos_x: f32,
  pos_y: f32,
  max_thrust: f32,
  radius: f32,
  #[comptime] cfg: Cfg,
) {
  if ABSOLUTE_POS > 0 {
    terminate!();
  }
  // A non-positive radius reaches nothing, and would make the cell radius
  // below negative.
  if radius <= 0.0f32 {
    terminate!();
  }

  let cell_size = params[P_CELL_SIZE as usize];
  let bounds_x = params[P_BOUNDS_X as usize];

  // The world wraps in x, so a point off either side of it is the same point
  // as its image inside. The C++ took the point as given and clamped it into
  // the edge column, which misses the far images of a point more than a cell
  // or so outside; the C++ view tiled the world 3x3, so the cursor was often
  // there. y does not wrap and is clamped into the grid as it stands.
  let px = pos_x - f32::floor(pos_x / bounds_x) * bounds_x;
  let py = pos_y;

  let grid_index = particle_to_cid(px, py, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;

  // compute neighborhood size from c_size and radius. the columns tile the
  // width exactly, so no extra overlap is needed at the wrap seam.
  let cell_radius = f32::ceil(radius / cell_size) as i32;

  // Rows outside the world hold nothing, so the row range is clipped rather
  // than walked and skipped — same result, and a wide radius does not spend
  // its iterations above the ceiling.
  let mut y_lo = cell_y - cell_radius;
  if y_lo < 0 {
    y_lo = 0;
  }
  let mut y_hi = cell_y + cell_radius;
  if y_hi > cfg.grid_h - 1 {
    y_hi = cfg.grid_h - 1;
  }
  let rows = (y_hi - y_lo + 1) as u32;
  let columns = (2 * cell_radius + 1) as u32;

  // iterate through cell neighborhood
  for row in 0..rows {
    let yi = y_lo + row as i32;
    for column in 0..columns {
      let xi = cell_x - cell_radius + column as i32;
      // wrap x if out of horizontal bounds. Unlike the C++, which shifted by
      // one world width at most and indexed out of the grid once the radius
      // passed a whole width, this takes as many widths as the column is
      // away.
      let wrapped_x = ((xi % cfg.grid_w) + cfg.grid_w) % cfg.grid_w;
      let x_shift = ((xi - wrapped_x) / cfg.grid_w) as f32 * bounds_x;

      let index = (yi * cfg.grid_w + wrapped_x) as usize;
      let start = grid.cell_start[index];
      let mut count = grid.cell_counts[index];
      if count > cfg.max_per_cell {
        count = cfg.max_per_cell;
      }

      // iterate through particles within the cell
      for k in 0..count {
        let particle_id = grid.sorted_ids[start as usize + k as usize] as usize;
        let diff_x = px - (sph.pos[2 * particle_id] + x_shift);
        let diff_y = py - sph.pos[2 * particle_id + 1];
        let distance = f32::sqrt(diff_x * diff_x + diff_y * diff_y);
        // A particle exactly at the point has no direction to it; the C++
        // normalised the zero vector there and wrote a NaN velocity.
        if distance < radius && distance > 0.0f32 {
          sph.vel[2 * particle_id] += diff_x / distance * max_thrust;
          sph.vel[2 * particle_id + 1] += diff_y / distance * max_thrust;
        }
      }
    }
  }
}

/// Plain-Rust reference for [`attract_single`].
#[allow(clippy::too_many_arguments)]
pub fn attract_single_ref(
  particles: &mut SphHost,
  grid: &GridRef,
  cell_size: f32,
  bounds_x: f32,
  cfg: &Cfg,
  pos: glam::Vec2,
  max_thrust: f32,
  radius: f32,
) {
  if radius <= 0.0 {
    return;
  }
  let p = glam::Vec2::new(pos.x - (pos.x / bounds_x).floor() * bounds_x, pos.y);

  let grid_index = super::particle_to_cid_ref(p, cell_size, cfg);
  let cell_x = (grid_index % cfg.grid_w as u32) as i32;
  let cell_y = (grid_index / cfg.grid_w as u32) as i32;
  let cell_radius = (radius / cell_size).ceil() as i32;

  let y_lo = (cell_y - cell_radius).max(0);
  let y_hi = (cell_y + cell_radius).min(cfg.grid_h - 1);
  for yi in y_lo..=y_hi {
    for xi in cell_x - cell_radius..=cell_x + cell_radius {
      let wrapped_x = xi.rem_euclid(cfg.grid_w);
      let x_shift = ((xi - wrapped_x) / cfg.grid_w) as f32 * bounds_x;
      let index = (yi * cfg.grid_w + wrapped_x) as usize;
      let start = grid.cell_start[index] as usize;
      let count = grid.cell_counts[index].min(cfg.max_per_cell) as usize;
      for id in &grid.sorted_ids[start..start + count] {
        let pid = *id as usize;
        let mut other = particles.pos[pid];
        other.x += x_shift;
        let diff = p - other;
        let distance = diff.length();
        if distance < radius && distance > 0.0 {
          particles.vel[pid] += diff / distance * max_thrust;
        }
      }
    }
  }
}

/// Launch helper. The grid is whatever the last step built, as in the C++,
/// which applied this after `update_fluid` without rebuilding it.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  grid: &GridDevice,
  params: &cubecl_runtime::server::Handle,
  cfg: Cfg,
  pos: glam::Vec2,
  max_thrust: f32,
  radius: f32,
) {
  attract_single::launch::<R>(
    client,
    CubeCount::Static(1, 1, 1),
    CubeDim::new_1d(1),
    super::sph_args(sph),
    super::grid_args(grid, &cfg),
    super::whole(params, super::PARAM_COUNT),
    pos.x,
    pos.y,
    max_thrust,
    radius,
    cfg,
  );
}

#[cfg(test)]
mod tests {
  use super::super::test_support::{Harness, assert_close_vec2};
  use glam::Vec2;

  fn check(pos: Vec2, max_thrust: f32, radius: f32) {
    let h = Harness::new();
    h.build_grid();
    super::launch(
      &h.client,
      &h.sph,
      &h.grid,
      &h.params_buf,
      h.cfg,
      pos,
      max_thrust,
      radius,
    );
    let actual = h.read_particles();

    let mut expected = h.particles.clone();
    super::attract_single_ref(
      &mut expected,
      &h.grid_ref(),
      h.geom.cell_size,
      h.geom.bounds.x,
      &h.cfg,
      pos,
      max_thrust,
      radius,
    );

    assert_close_vec2(&actual.vel, &expected.vel, 1e-5, "vel after attract");
    // Only velocities move.
    assert_eq!(actual.pos, h.particles.pos);
    let moved = (0..h.particles.len())
      .filter(|&i| expected.vel[i] != h.particles.vel[i])
      .count();
    assert!(moved > 0, "the tool reached no particle");
    // The reference is checked against the spec too, not just the kernel
    // against it: while the radius reaches one image of each particle, every
    // nudge is along the minimum-image line to the point, towards it for a
    // positive thrust.
    let width = h.geom.bounds.x;
    if radius < 0.5 * width {
      for i in 0..h.particles.len() {
        let delta = expected.vel[i] - h.particles.vel[i];
        if delta == Vec2::ZERO {
          continue;
        }
        let mut to = pos - h.particles.pos[i];
        to.x -= (to.x / width).round() * width;
        assert!(to.length() < radius, "particle {i} nudged from outside");
        assert!(
          delta.dot(to) * max_thrust > 0.0,
          "particle {i} nudged the wrong way"
        );
      }
    }
    // Vapor and free slots are not in the grid, so the tool never sees them.
    for i in 0..h.particles.len() {
      if !h.particles.state[i].in_fluid() {
        assert_eq!(actual.vel[i], h.particles.vel[i], "particle {i}");
      }
    }
  }

  /// Near the left edge, so the neighbourhood wraps across the seam.
  #[test]
  fn attract_matches_reference_across_the_seam() {
    check(Vec2::new(0.1, 0.3), 0.15, 0.6);
  }

  /// A negative thrust repels, from a point off the right of the world —
  /// the wrap brings it back to the left edge.
  #[test]
  fn repel_matches_reference_off_the_world() {
    check(Vec2::new(2.3, 0.5), -0.2, 0.5);
  }

  /// Wider than the world (2 m in the harness), so columns are visited more
  /// than once and some neighbours are shifted by two widths.
  #[test]
  fn radius_wider_than_the_world_matches_reference() {
    check(Vec2::new(1.0, 0.4), 0.05, 2.6);
  }
}
