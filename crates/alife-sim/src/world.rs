//! World geometry: everything the parameters imply about buffer shapes.
//!
//! Derived once per configuration and then constant, so the kernels can take
//! the integer parts as comptime constants.

use glam::Vec2;

use crate::SimParams;

/// The structural constants a kernel bakes in at JIT time: the integer half of
/// [`WorldGeometry`].
///
/// Only integers live here. They change when the world is resized, which
/// already rebuilds every buffer; everything a slider can move is a runtime
/// value in the kernels' params buffer, so dragging a slider never recompiles
/// a kernel. CubeCL keys a compiled kernel on the hash of its comptime
/// arguments, hence the `Eq`/`Hash` bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cfg {
  pub grid_w: i32,
  pub grid_h: i32,
  pub num_cells: u32,
  pub num_particles: u32,
  pub max_per_cell: u32,
  pub soil_w: i32,
  pub soil_h: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldGeometry {
  pub bounds: Vec2,
  /// Neighbour grid. x wraps, so the columns must tile the width exactly:
  /// the column count rounds down and the cells widen to fit. `cell_size >=
  /// smoothing_radius`, so a 3x3 neighbourhood still covers every particle
  /// within the radius.
  pub grid_width: usize,
  pub grid_height: usize,
  pub cell_size: f32,
  pub max_particles_per_cell: u32,
  pub soil_width: usize,
  pub soil_height: usize,
  pub soil_cell_size: f32,
  /// Fluid particles: the ones a fresh world starts with, or the non-body
  /// particles a dump carried.
  pub fluid_particles: usize,
  /// Body slots reserved on top of the fluid, `max_organisms x max_limbs x
  /// max_particles_per_limb`. They start [`crate::ParticleKind::Free`].
  pub body_slots: usize,
  /// `fluid_particles + body_slots`; the length of every particle buffer.
  pub num_particles: usize,
}

impl WorldGeometry {
  pub fn from_params(params: &SimParams) -> Self {
    // The CLI validates first; this is the backstop for programmatic callers,
    // because a non-positive radius or cell size saturates the divisions
    // below into a grid allocation that hangs rather than fails.
    assert!(
      params.validate().is_ok(),
      "invalid parameters: {}",
      params.validate().unwrap_err()
    );
    let bounds = Vec2::new(params.world_width, params.world_height);

    let grid_width = ((bounds.x / params.smoothing_radius).floor() as usize).max(1);
    let cell_size = bounds.x / grid_width as f32;
    // y does not wrap and a partial top row is harmless, so it rounds up
    let grid_height = (bounds.y / cell_size).ceil() as usize;

    let fluid_particles = params.particles_per_cell.max(0) as usize * grid_width * grid_height;
    // Capacity is fixed at start, as `organism.md` requires: no buffer grows
    // once the kernels have baked `num_particles` in as a comptime constant.
    // One source for the reserved capacity: `BodyState` sizes its particle
    // map from the same `BodyCfg`, so the two cannot disagree about how many
    // slots the buffers hold (they did when this clamped to 0 and it to 1).
    let body_slots = crate::bodies::BodyCfg::from_params(params).particle_map_len();

    Self {
      bounds,
      grid_width,
      grid_height,
      cell_size,
      max_particles_per_cell: params.max_particles_per_cell.max(0) as u32,
      soil_width: (params.world_width / params.soil_cell_size).round() as usize,
      soil_height: (params.world_height / params.soil_cell_size).round() as usize,
      soil_cell_size: params.soil_cell_size,
      fluid_particles,
      body_slots,
      num_particles: fluid_particles + body_slots,
    }
  }

  pub fn num_cells(&self) -> usize {
    self.grid_width * self.grid_height
  }

  pub fn num_soil_cells(&self) -> usize {
    self.soil_width * self.soil_height
  }

  /// The comptime half of the kernel configuration.
  pub fn cfg(&self) -> Cfg {
    Cfg {
      grid_w: self.grid_width as i32,
      grid_h: self.grid_height as i32,
      num_cells: self.num_cells() as u32,
      num_particles: self.num_particles as u32,
      max_per_cell: self.max_particles_per_cell,
      soil_w: self.soil_width as i32,
      soil_h: self.soil_height as i32,
    }
  }

  /// Adopt a fluid count that came from a dump instead of from the
  /// `particles_per_cell` default. The body capacity is unchanged and still
  /// sits on top of it.
  pub fn with_fluid_count(mut self, n: usize) -> Self {
    self.fluid_particles = n;
    self.num_particles = n + self.body_slots;
    self
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn default_geometry_matches_the_cpp_headless_baseline() {
    let geom = WorldGeometry::from_params(&SimParams::default());
    assert_eq!(geom.grid_width, 160);
    assert_eq!(geom.grid_height, 80);
    assert_eq!(geom.cell_size, 0.2);
    assert_eq!(geom.fluid_particles, 51_200);
    assert_eq!(geom.soil_width, 320);
    assert_eq!(geom.soil_height, 160);
  }

  #[test]
  fn body_slots_sit_on_top_of_the_fluid() {
    let params = SimParams::default();
    let geom = WorldGeometry::from_params(&params);
    assert_eq!(geom.body_slots, 256 * 16 * 8);
    assert_eq!(geom.num_particles, geom.fluid_particles + geom.body_slots);

    // A dump replaces the fluid count and leaves the capacity alone.
    let loaded = geom.with_fluid_count(1_000);
    assert_eq!(loaded.fluid_particles, 1_000);
    assert_eq!(loaded.num_particles, 1_000 + geom.body_slots);
  }
}
