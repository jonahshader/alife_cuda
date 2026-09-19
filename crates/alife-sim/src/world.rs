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
  pub num_particles: usize,
}

impl WorldGeometry {
  pub fn from_params(params: &SimParams) -> Self {
    let bounds = Vec2::new(params.world_width, params.world_height);

    let grid_width = ((bounds.x / params.smoothing_radius).floor() as usize).max(1);
    let cell_size = bounds.x / grid_width as f32;
    // y does not wrap and a partial top row is harmless, so it rounds up
    let grid_height = (bounds.y / cell_size).ceil() as usize;

    Self {
      bounds,
      grid_width,
      grid_height,
      cell_size,
      max_particles_per_cell: params.max_particles_per_cell.max(0) as u32,
      soil_width: (params.world_width / params.soil_cell_size).round() as usize,
      soil_height: (params.world_height / params.soil_cell_size).round() as usize,
      soil_cell_size: params.soil_cell_size,
      num_particles: params.particles_per_cell.max(0) as usize * grid_width * grid_height,
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

  /// Adopt a particle count that came from a dump instead of from the
  /// `particles_per_cell` default.
  pub fn with_particle_count(mut self, n: usize) -> Self {
    self.num_particles = n;
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
    assert_eq!(geom.num_particles, 51_200);
    assert_eq!(geom.soil_width, 320);
    assert_eq!(geom.soil_height, 160);
  }
}
