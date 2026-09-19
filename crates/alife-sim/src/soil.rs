//! The soil grid: composition per cell and the terrain generators.
//!
//! Ported from the C++ tree's `src/systems/soil.cu` / `soil.cuh`.

use crate::define_soa;

// Porosity: fraction of volume that is pore space
pub const SAND_POROSITY: f32 = 0.38;
pub const SILT_POROSITY: f32 = 0.45;
pub const CLAY_POROSITY: f32 = 0.50;

// Capillary strength: suction pressure in SPH units (clay >> silt >> sand)
pub const SAND_CAPILLARY: f32 = 5.0;
pub const SILT_CAPILLARY: f32 = 100.0;
pub const CLAY_CAPILLARY: f32 = 8000.0;

pub const SAND_FRICTION: f32 = 8.0;
pub const SILT_FRICTION: f32 = 20.0;
pub const CLAY_FRICTION: f32 = 50.0;

define_soa! {
    /// Soil composition, one array per field.
    SoilHost / SoilDevice {
        sand_density: f32,
        silt_density: f32,
        clay_density: f32,
        ph: f32 = 6.5,
        organic_matter: f32,
        /// Written by nothing today: the C++ `calculate_soil_saturation`
        /// kernel that filled it was never launched and did not come along.
        /// Capillary suction reads the local saturation implied by the SPH
        /// density instead — see `kernels::accel`.
        saturation: f32,
    }
}

/// How the soil is laid out at startup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainMode {
  /// Noise heightmap with a softmax blend of sand/silt/clay.
  Noise,
  /// Capillary tube test: a shared water pool under separate soil columns.
  CapillaryTest,
}

impl TerrainMode {
  pub fn from_flag(mode: i32) -> Self {
    // The C++ `switch` treats everything that is not 1 as the noise mode.
    match mode {
      1 => TerrainMode::CapillaryTest,
      _ => TerrainMode::Noise,
    }
  }
}

/// Soil grid dimensions and contents.
#[derive(Debug, Clone)]
pub struct SoilGrid {
  pub width: usize,
  pub height: usize,
  pub cell_size: f32,
  pub cells: SoilHost,
}

impl SoilGrid {
  pub fn new(width: usize, height: usize, cell_size: f32, mode: TerrainMode, seed: u64) -> Self {
    let cells = match mode {
      TerrainMode::CapillaryTest => capillary_test(width, height),
      TerrainMode::Noise => noise_terrain(width, height, seed),
    };
    Self {
      width,
      height,
      cell_size,
      cells,
    }
  }

  pub fn len(&self) -> usize {
    self.width * self.height
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }
}

/// The soil step. Nothing happens yet: the C++ `update_soil_cuda` is an empty
/// `// TODO: implement`, and porting an empty function faithfully means
/// keeping it empty rather than inventing behavior.
pub fn update_soil(_soil: &mut SoilGrid, _dt: f32) {}

// --- Per-cell derived properties. Mirrored by `#[cube]` copies in `kernels`. ---

pub fn porosity(soil: &SoilHost, i: usize) -> f32 {
  soil.sand_density[i] * SAND_POROSITY
    + soil.silt_density[i] * SILT_POROSITY
    + soil.clay_density[i] * CLAY_POROSITY
}

pub fn solid_density(soil: &SoilHost, i: usize, target_density: f32) -> f32 {
  (1.0 - porosity(soil, i)) * target_density
}

pub fn pore_capacity(soil: &SoilHost, i: usize, target_density: f32) -> f32 {
  porosity(soil, i) * target_density
}

pub fn capillary_strength(soil: &SoilHost, i: usize) -> f32 {
  soil.sand_density[i] * SAND_CAPILLARY
    + soil.silt_density[i] * SILT_CAPILLARY
    + soil.clay_density[i] * CLAY_CAPILLARY
}

pub fn friction(soil: &SoilHost, i: usize) -> f32 {
  soil.sand_density[i] * SAND_FRICTION
    + soil.silt_density[i] * SILT_FRICTION
    + soil.clay_density[i] * CLAY_FRICTION
}

// --- Terrain generators ---

/// Capillary tube test: shared water pool at the bottom, separate soil columns
/// above. Bottom 20%: empty (no soil) — water pool. Above 20%: soil columns
/// with air gaps between them. Left half: pure sand | silt | clay. Right half:
/// gradients.
pub fn capillary_test(width: usize, height: usize) -> SoilHost {
  let mut soil = SoilHost::new(width * height);

  let pool_h = (height as f32 * 0.2) as usize; // bottom 20% is open water pool
  let terrain_h = height;
  let gap = width / 40;
  let half_w = width / 2;

  // -- Left half: three pure columns with gaps --
  let pure_col_w = (half_w - 2 * gap) / 3;
  let pure_x = [
    (0, pure_col_w),
    (pure_col_w + gap, 2 * pure_col_w + gap),
    (2 * pure_col_w + 2 * gap, half_w),
  ];

  for y in pool_h..terrain_h {
    let row = y * width;
    for x in pure_x[0].0..pure_x[0].1 {
      soil.sand_density[x + row] = 1.0;
    }
    for x in pure_x[1].0..pure_x[1].1 {
      soil.silt_density[x + row] = 1.0;
    }
    for x in pure_x[2].0..pure_x[2].1 {
      soil.clay_density[x + row] = 1.0;
    }
  }

  // -- Right half: three gradient columns with gaps --
  let grad_x0 = half_w + gap;
  let grad_col_w = (width - grad_x0 - 2 * gap) / 3;
  let grad_x = [
    (grad_x0, grad_x0 + grad_col_w),
    (grad_x0 + grad_col_w + gap, grad_x0 + 2 * grad_col_w + gap),
    (grad_x0 + 2 * grad_col_w + 2 * gap, width),
  ];

  for y in pool_h..terrain_h {
    let row = y * width;
    // sand -> silt gradient
    for x in grad_x[0].0..grad_x[0].1 {
      let t = (x - grad_x[0].0) as f32 / (grad_x[0].1 - grad_x[0].0) as f32;
      soil.sand_density[x + row] = 1.0 - t;
      soil.silt_density[x + row] = t;
    }
    // silt -> clay gradient
    for x in grad_x[1].0..grad_x[1].1 {
      let t = (x - grad_x[1].0) as f32 / (grad_x[1].1 - grad_x[1].0) as f32;
      soil.silt_density[x + row] = 1.0 - t;
      soil.clay_density[x + row] = t;
    }
    // sand -> clay gradient
    for x in grad_x[2].0..grad_x[2].1 {
      let t = (x - grad_x[2].0) as f32 / (grad_x[2].1 - grad_x[2].0) as f32;
      soil.sand_density[x + row] = 1.0 - t;
      soil.clay_density[x + row] = t;
    }
  }

  soil
}

/// Noise heightmap with a softmax blend of the three soil types.
///
/// The two noise generators are seeded from a `std::mt19937_64` drawn in the
/// same order as the C++, and `fastnoise-lite` is a port of the same
/// FastNoiseLite, so the same world seed produces the same terrain in both
/// trees — see `docs/perf.md` for the measured agreement.
pub fn noise_terrain(width: usize, height: usize, seed: u64) -> SoilHost {
  use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

  let mut soil = SoilHost::new(width * height);
  let mut rng = Mt19937_64::new(seed);

  let mut heightmap_noise = FastNoiseLite::with_seed(rng.next_u64() as i32);
  heightmap_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
  heightmap_noise.set_fractal_type(Some(FractalType::FBm));
  heightmap_noise.set_fractal_octaves(Some(6));
  heightmap_noise.set_frequency(Some(1.0 / height as f32));

  let mut soil_noise = FastNoiseLite::with_seed(rng.next_u64() as i32);
  soil_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
  soil_noise.set_fractal_type(Some(FractalType::FBm));
  soil_noise.set_fractal_octaves(Some(5));
  soil_noise.set_frequency(Some(0.01));

  // heightmap is 1d, but interpreted as 2d
  let heightmap: Vec<f32> = (0..width)
    .map(|x| heightmap_noise.get_noise_2d(x as f32, 0.0))
    .collect();

  let min_land_height = 0.1f32;
  for (x, sample) in heightmap.iter().enumerate() {
    let mut hf = *sample;
    hf = (hf * 2.0).tanh();
    let mut xf = x as f32 / width as f32;
    xf = xf * (1.0 - xf) * 4.0;
    xf = xf.sqrt();
    hf = hf * 0.5 + 0.5;
    hf *= xf;
    let h = ((min_land_height + hf * 0.8) * height as f32) as usize;
    for y in 0..h {
      let id = x + y * width;
      let mut sand = soil_noise.get_noise_3d(x as f32, y as f32, 0.0);
      let mut silt = soil_noise.get_noise_3d(x as f32 * 0.75, y as f32, 300.0);
      let mut clay = soil_noise.get_noise_3d(x as f32 * 0.5, y as f32, 600.0);

      // sharpen
      let sharpen = 50.0f32;
      sand *= sharpen;
      silt *= sharpen;
      clay *= sharpen;

      // softmax
      sand = sand.exp();
      silt = silt.exp();
      clay = clay.exp();

      let density = 1.0 / (sand + silt + clay);
      soil.sand_density[id] = sand * density;
      soil.silt_density[id] = silt * density;
      soil.clay_density[id] = clay * density;
    }
  }

  soil
}

/// `std::mt19937_64`, so the noise seeds drawn here match the C++ ones.
struct Mt19937_64 {
  state: [u64; 312],
  index: usize,
}

impl Mt19937_64 {
  fn new(seed: u64) -> Self {
    let mut state = [0u64; 312];
    state[0] = seed;
    for i in 1..312 {
      state[i] = 6364136223846793005u64
        .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 62))
        .wrapping_add(i as u64);
    }
    Self { state, index: 312 }
  }

  fn next_u64(&mut self) -> u64 {
    if self.index >= 312 {
      self.twist();
    }
    let mut x = self.state[self.index];
    self.index += 1;
    x ^= (x >> 29) & 0x5555_5555_5555_5555;
    x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
    x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
    x ^= x >> 43;
    x
  }

  fn twist(&mut self) {
    const LOWER_MASK: u64 = (1 << 31) - 1;
    const UPPER_MASK: u64 = !LOWER_MASK;
    for i in 0..312 {
      let x = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 312] & LOWER_MASK);
      let mut x_a = x >> 1;
      if !x.is_multiple_of(2) {
        x_a ^= 0xB502_6F5A_A966_19E9;
      }
      self.state[i] = self.state[(i + 156) % 312] ^ x_a;
    }
    self.index = 0;
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn mt19937_64_matches_reference_stream() {
    // The standard's own check value: the 10000th output of a default
    // constructed mt19937_64 (seed 5489).
    let mut rng = Mt19937_64::new(5489);
    let mut last = 0;
    for _ in 0..10000 {
      last = rng.next_u64();
    }
    assert_eq!(last, 9981545732273789042);
  }

  #[test]
  fn capillary_test_columns_at_default_size() {
    let soil = capillary_test(320, 160);
    // Pool rows stay empty, the pure sand column is solid above them.
    assert_eq!(soil.sand_density[31 * 320], 0.0);
    assert_eq!(soil.sand_density[32 * 320], 1.0);
    assert_eq!(soil.sand_density[47 + 100 * 320], 1.0);
    // The gap between the sand and silt columns is air.
    assert_eq!(soil.sand_density[50 + 100 * 320], 0.0);
    assert_eq!(soil.silt_density[50 + 100 * 320], 0.0);
    assert_eq!(soil.silt_density[56 + 100 * 320], 1.0);
    assert_eq!(soil.clay_density[112 + 100 * 320], 1.0);
    // ph keeps its declared initial value everywhere.
    assert_eq!(soil.ph[0], 6.5);
  }
}
