//! Sampling the soil grid at a particle position.
//!
//! Ported from `calculate_soil_density_at_pos_smoothstep`,
//! `calculate_soil_properties_at_pos` and the soil-presence block inside the
//! soil-aware `calculate_evap_prob` in `src/systems/particle_fluid2.cu`.
//!
//! The bicubic and standalone-gradient variants in the C++ file are not
//! reachable from the soil-coupled path and did not come along.

use cubecl::prelude::*;

use super::{Cfg, SoilArgs, clamp_i32};
use crate::soil::{
  CLAY_CAPILLARY, CLAY_FRICTION, CLAY_POROSITY, SAND_CAPILLARY, SAND_FRICTION, SAND_POROSITY,
  SILT_CAPILLARY, SILT_FRICTION, SILT_POROSITY, SoilHost,
};

// The clamps are spelled out: `f32::clamp` is a Rust method, not something
// the kernel language offers.
#[cube]
#[allow(clippy::manual_clamp)]
pub fn smoothstep01(x: f32) -> f32 {
  let mut t = x;
  if t < 0.0f32 {
    t = 0.0f32;
  }
  if t > 1.0f32 {
    t = 1.0f32;
  }
  t * t * (3.0f32 - 2.0f32 * t)
}

/// Derivative of smoothstep, in [0,1]
#[cube]
#[allow(clippy::manual_clamp)]
pub fn smoothstep01_derivative(x: f32) -> f32 {
  let mut t = x;
  if t < 0.0f32 {
    t = 0.0f32;
  }
  if t > 1.0f32 {
    t = 1.0f32;
  }
  6.0f32 * t * (1.0f32 - t)
}

#[cube]
pub fn porosity(soil: &SoilArgs, i: usize) -> f32 {
  soil.sand_density[i] * SAND_POROSITY
    + soil.silt_density[i] * SILT_POROSITY
    + soil.clay_density[i] * CLAY_POROSITY
}

#[cube]
pub fn solid_density(soil: &SoilArgs, i: usize, target_density: f32) -> f32 {
  (1.0f32 - porosity(soil, i)) * target_density
}

#[cube]
pub fn pore_capacity(soil: &SoilArgs, i: usize, target_density: f32) -> f32 {
  porosity(soil, i) * target_density
}

/// The fraction of a cell's volume that is solid mineral: zero in air, and
/// `1 - porosity` of whichever mix fills a soil cell.
///
/// Not [`solid_density`] over `target_density`. That one reads `1` in air,
/// because an empty cell has no porosity to subtract — which is what the
/// fluid wants (it is a density offset that makes a particle in free air
/// neutral) and the opposite of what a limb's contact sensor wants. Each
/// mineral contributes its own solid fraction, so air contributes nothing and
/// the value also tells sand (0.62) from clay (0.50).
#[cube]
pub fn solid_fraction(soil: &SoilArgs, i: usize) -> f32 {
  soil.sand_density[i] * (1.0f32 - SAND_POROSITY)
    + soil.silt_density[i] * (1.0f32 - SILT_POROSITY)
    + soil.clay_density[i] * (1.0f32 - CLAY_POROSITY)
}

/// [`solid_fraction`] interpolated at a position, for the brain's soil sensor.
#[cube]
pub fn solid_fraction_at_pos(
  pos_x: f32,
  pos_y: f32,
  soil: &SoilArgs,
  soil_size: f32,
  #[comptime] cfg: Cfg,
) -> f32 {
  let c = corners_at(pos_x, pos_y, soil_size, cfg);
  smooth_bilinear(
    solid_fraction(soil, c.i00),
    solid_fraction(soil, c.i01),
    solid_fraction(soil, c.i10),
    solid_fraction(soil, c.i11),
    smoothstep01(c.dx),
    smoothstep01(c.dy),
  )
}

#[cube]
pub fn capillary_strength(soil: &SoilArgs, i: usize) -> f32 {
  soil.sand_density[i] * SAND_CAPILLARY
    + soil.silt_density[i] * SILT_CAPILLARY
    + soil.clay_density[i] * CLAY_CAPILLARY
}

#[cube]
pub fn friction(soil: &SoilArgs, i: usize) -> f32 {
  soil.sand_density[i] * SAND_FRICTION
    + soil.silt_density[i] * SILT_FRICTION
    + soil.clay_density[i] * CLAY_FRICTION
}

/// The four cell indices and two interpolation weights a position samples.
///
/// x wraps, y clamps — the same asymmetry as the particle grid.
#[derive(CubeType, Clone, Copy)]
pub struct Corners {
  pub i00: usize,
  pub i01: usize,
  pub i10: usize,
  pub i11: usize,
  pub dx: f32,
  pub dy: f32,
}

#[cube]
pub fn corners_at(pos_x: f32, pos_y: f32, soil_size: f32, #[comptime] cfg: Cfg) -> Corners {
  let half_soil_size = soil_size * 0.5f32;
  let fx = (pos_x - half_soil_size) / soil_size;
  let fy = (pos_y - half_soil_size) / soil_size;

  let x0 = f32::floor(fx) as i32;
  let y0 = f32::floor(fy) as i32;

  let dx = fx - x0 as f32;
  let dy = fy - y0 as f32;

  let xa = ((x0 % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let xb = (((x0 + 1) % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let ya = clamp_i32(y0, 0, cfg.soil_h - 1);
  let yb = clamp_i32(y0 + 1, 0, cfg.soil_h - 1);

  Corners {
    i00: (ya * cfg.soil_w + xa) as usize,
    i01: (ya * cfg.soil_w + xb) as usize,
    i10: (yb * cfg.soil_w + xa) as usize,
    i11: (yb * cfg.soil_w + xb) as usize,
    dx,
    dy,
  }
}

/// Smooth "bilinear" interpolation using smoothstep: interpolate horizontally
/// at y0 and y1 using tx, then vertically between those results using ty.
#[cube]
pub fn smooth_bilinear(c00: f32, c01: f32, c10: f32, c11: f32, tx: f32, ty: f32) -> f32 {
  let d0 = c00 * (1.0f32 - tx) + c01 * tx;
  let d1 = c10 * (1.0f32 - tx) + c11 * tx;
  d0 * (1.0f32 - ty) + d1 * ty
}

/// Solid density under a particle, as `calculate_particle_density` adds it.
#[cube]
pub fn solid_density_at_pos(
  pos_x: f32,
  pos_y: f32,
  soil: &SoilArgs,
  soil_size: f32,
  target_density: f32,
  #[comptime] cfg: Cfg,
) -> f32 {
  let c = corners_at(pos_x, pos_y, soil_size, cfg);
  smooth_bilinear(
    solid_density(soil, c.i00, target_density),
    solid_density(soil, c.i01, target_density),
    solid_density(soil, c.i10, target_density),
    solid_density(soil, c.i11, target_density),
    smoothstep01(c.dx),
    smoothstep01(c.dy),
  )
}

/// Everything `calculate_accel` needs from the soil at one position.
#[derive(CubeType, Clone, Copy)]
pub struct SoilPropertiesAtPos {
  pub solid_density: f32,
  pub solid_density_gradient_x: f32,
  pub solid_density_gradient_y: f32,
  pub capillary_strength: f32,
  pub pore_capacity: f32,
}

#[cube]
pub fn properties_at_pos(
  pos_x: f32,
  pos_y: f32,
  soil: &SoilArgs,
  soil_size: f32,
  target_density: f32,
  #[comptime] cfg: Cfg,
) -> SoilPropertiesAtPos {
  let c = corners_at(pos_x, pos_y, soil_size, cfg);

  let sd00 = solid_density(soil, c.i00, target_density);
  let sd01 = solid_density(soil, c.i01, target_density);
  let sd10 = solid_density(soil, c.i10, target_density);
  let sd11 = solid_density(soil, c.i11, target_density);

  let tx = smoothstep01(c.dx);
  let ty = smoothstep01(c.dy);
  let dtx = smoothstep01_derivative(c.dx);
  let dty = smoothstep01_derivative(c.dy);
  let inv_soil_size = 1.0f32 / soil_size;

  // solid density value + gradient: bilinear in the smoothstepped (tx, ty)
  let sd0 = sd00 * (1.0f32 - tx) + sd01 * tx; // at y0
  let sd1 = sd10 * (1.0f32 - tx) + sd11 * tx; // at y1
  let value = sd0 * (1.0f32 - ty) + sd1 * ty;
  // ∂f/∂tx and ∂f/∂ty of the bilinear form
  let psd_tx = (sd01 - sd00) * (1.0f32 - ty) + (sd11 - sd10) * ty;
  let psd_ty = sd1 - sd0;
  // Chain rule back to world space: the raw cell coordinate is
  // (pos - half_soil_size) / soil_size - floor, so its derivative wrt x is
  // 1/soil_size; tx = smoothstep(raw) contributes dtx. Hence
  //   grad_x = ∂f/∂tx · dtx · (1/soil_size), and likewise for y.

  SoilPropertiesAtPos {
    solid_density: value,
    solid_density_gradient_x: psd_tx * dtx * inv_soil_size,
    solid_density_gradient_y: psd_ty * dty * inv_soil_size,
    capillary_strength: smooth_bilinear(
      capillary_strength(soil, c.i00),
      capillary_strength(soil, c.i01),
      capillary_strength(soil, c.i10),
      capillary_strength(soil, c.i11),
      tx,
      ty,
    ),
    pore_capacity: smooth_bilinear(
      pore_capacity(soil, c.i00, target_density),
      pore_capacity(soil, c.i01, target_density),
      pore_capacity(soil, c.i10, target_density),
      pore_capacity(soil, c.i11, target_density),
      tx,
      ty,
    ),
  }
}

/// Total soil material (sand+silt+clay), which is 1.0 in soil cells and 0.0 in
/// air, with a smooth bilinear transition at the surface.
///
/// The C++ accumulates the four weighted corners rather than nesting two
/// lerps like [`smooth_bilinear`] does. The two are the same expression in
/// exact arithmetic and round differently in `f32`, so this keeps the C++
/// shape.
#[cube]
pub fn presence_at_pos(
  pos_x: f32,
  pos_y: f32,
  soil: &SoilArgs,
  soil_size: f32,
  #[comptime] cfg: Cfg,
) -> f32 {
  let c = corners_at(pos_x, pos_y, soil_size, cfg);
  let tx = smoothstep01(c.dx);
  let ty = smoothstep01(c.dy);
  let total00 = soil.sand_density[c.i00] + soil.silt_density[c.i00] + soil.clay_density[c.i00];
  let total01 = soil.sand_density[c.i01] + soil.silt_density[c.i01] + soil.clay_density[c.i01];
  let total10 = soil.sand_density[c.i10] + soil.silt_density[c.i10] + soil.clay_density[c.i10];
  let total11 = soil.sand_density[c.i11] + soil.silt_density[c.i11] + soil.clay_density[c.i11];

  let mut presence = total00 * (1.0f32 - tx) * (1.0f32 - ty);
  presence += total01 * tx * (1.0f32 - ty);
  presence += total10 * (1.0f32 - tx) * ty;
  presence += total11 * tx * ty;
  presence
}

// --- Host twins ---

pub fn smoothstep01_ref(x: f32) -> f32 {
  let t = x.clamp(0.0, 1.0);
  t * t * (3.0 - 2.0 * t)
}

pub fn smoothstep01_derivative_ref(x: f32) -> f32 {
  let t = x.clamp(0.0, 1.0);
  6.0 * t * (1.0 - t)
}

pub struct CornersRef {
  pub i00: usize,
  pub i01: usize,
  pub i10: usize,
  pub i11: usize,
  pub dx: f32,
  pub dy: f32,
}

pub fn corners_at_ref(pos: glam::Vec2, soil_size: f32, cfg: &Cfg) -> CornersRef {
  let half_soil_size = soil_size * 0.5;
  let fx = (pos.x - half_soil_size) / soil_size;
  let fy = (pos.y - half_soil_size) / soil_size;
  let x0 = fx.floor() as i32;
  let y0 = fy.floor() as i32;
  let dx = fx - x0 as f32;
  let dy = fy - y0 as f32;

  let xa = ((x0 % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let xb = (((x0 + 1) % cfg.soil_w) + cfg.soil_w) % cfg.soil_w;
  let ya = y0.clamp(0, cfg.soil_h - 1);
  let yb = (y0 + 1).clamp(0, cfg.soil_h - 1);

  CornersRef {
    i00: (ya * cfg.soil_w + xa) as usize,
    i01: (ya * cfg.soil_w + xb) as usize,
    i10: (yb * cfg.soil_w + xa) as usize,
    i11: (yb * cfg.soil_w + xb) as usize,
    dx,
    dy,
  }
}

pub fn smooth_bilinear_ref(c00: f32, c01: f32, c10: f32, c11: f32, tx: f32, ty: f32) -> f32 {
  let d0 = c00 * (1.0 - tx) + c01 * tx;
  let d1 = c10 * (1.0 - tx) + c11 * tx;
  d0 * (1.0 - ty) + d1 * ty
}

pub fn solid_density_at_pos_ref(
  pos: glam::Vec2,
  soil: &SoilHost,
  soil_size: f32,
  target_density: f32,
  cfg: &Cfg,
) -> f32 {
  let c = corners_at_ref(pos, soil_size, cfg);
  let sd = |i: usize| crate::soil::solid_density(soil, i, target_density);
  smooth_bilinear_ref(
    sd(c.i00),
    sd(c.i01),
    sd(c.i10),
    sd(c.i11),
    smoothstep01_ref(c.dx),
    smoothstep01_ref(c.dy),
  )
}

/// Host twin of [`solid_fraction`].
pub fn solid_fraction_ref(soil: &SoilHost, i: usize) -> f32 {
  soil.sand_density[i] * (1.0 - SAND_POROSITY)
    + soil.silt_density[i] * (1.0 - SILT_POROSITY)
    + soil.clay_density[i] * (1.0 - CLAY_POROSITY)
}

/// Host twin of [`solid_fraction_at_pos`].
pub fn solid_fraction_at_pos_ref(
  pos: glam::Vec2,
  soil: &SoilHost,
  soil_size: f32,
  cfg: &Cfg,
) -> f32 {
  let c = corners_at_ref(pos, soil_size, cfg);
  smooth_bilinear_ref(
    solid_fraction_ref(soil, c.i00),
    solid_fraction_ref(soil, c.i01),
    solid_fraction_ref(soil, c.i10),
    solid_fraction_ref(soil, c.i11),
    smoothstep01_ref(c.dx),
    smoothstep01_ref(c.dy),
  )
}

pub struct SoilPropertiesRef {
  pub solid_density: f32,
  pub solid_density_gradient: glam::Vec2,
  pub capillary_strength: f32,
  pub pore_capacity: f32,
}

pub fn properties_at_pos_ref(
  pos: glam::Vec2,
  soil: &SoilHost,
  soil_size: f32,
  target_density: f32,
  cfg: &Cfg,
) -> SoilPropertiesRef {
  let c = corners_at_ref(pos, soil_size, cfg);
  let sd = |i: usize| crate::soil::solid_density(soil, i, target_density);
  let (sd00, sd01, sd10, sd11) = (sd(c.i00), sd(c.i01), sd(c.i10), sd(c.i11));

  let tx = smoothstep01_ref(c.dx);
  let ty = smoothstep01_ref(c.dy);
  let dtx = smoothstep01_derivative_ref(c.dx);
  let dty = smoothstep01_derivative_ref(c.dy);
  let inv_soil_size = 1.0 / soil_size;

  let sd0 = sd00 * (1.0 - tx) + sd01 * tx;
  let sd1 = sd10 * (1.0 - tx) + sd11 * tx;
  let psd_tx = (sd01 - sd00) * (1.0 - ty) + (sd11 - sd10) * ty;
  let psd_ty = sd1 - sd0;

  SoilPropertiesRef {
    solid_density: sd0 * (1.0 - ty) + sd1 * ty,
    solid_density_gradient: glam::Vec2::new(
      psd_tx * dtx * inv_soil_size,
      psd_ty * dty * inv_soil_size,
    ),
    capillary_strength: smooth_bilinear_ref(
      crate::soil::capillary_strength(soil, c.i00),
      crate::soil::capillary_strength(soil, c.i01),
      crate::soil::capillary_strength(soil, c.i10),
      crate::soil::capillary_strength(soil, c.i11),
      tx,
      ty,
    ),
    pore_capacity: smooth_bilinear_ref(
      crate::soil::pore_capacity(soil, c.i00, target_density),
      crate::soil::pore_capacity(soil, c.i01, target_density),
      crate::soil::pore_capacity(soil, c.i10, target_density),
      crate::soil::pore_capacity(soil, c.i11, target_density),
      tx,
      ty,
    ),
  }
}

pub fn presence_at_pos_ref(pos: glam::Vec2, soil: &SoilHost, soil_size: f32, cfg: &Cfg) -> f32 {
  let c = corners_at_ref(pos, soil_size, cfg);
  let total = |i: usize| soil.sand_density[i] + soil.silt_density[i] + soil.clay_density[i];
  let tx = smoothstep01_ref(c.dx);
  let ty = smoothstep01_ref(c.dy);
  let mut presence = total(c.i00) * (1.0 - tx) * (1.0 - ty);
  presence += total(c.i01) * tx * (1.0 - ty);
  presence += total(c.i10) * (1.0 - tx) * ty;
  presence += total(c.i11) * tx * ty;
  presence
}
