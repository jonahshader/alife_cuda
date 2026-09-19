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

/// A stretch of the world the per-column metrics are aggregated over, in soil
/// grid cells. `x0` is inclusive, `x1` exclusive.
///
/// The capillary test terrain publishes its six soil columns this way
/// ([`SoilGrid::columns`]) so the soil-specialization experiment can bin
/// organisms by the soil they are anchored in. The cells between two columns
/// are air and belong to no extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnExtent {
  pub x0: usize,
  pub x1: usize,
  /// Short identifier, used as a CSV column-name suffix, so it stays
  /// `[a-z_]`.
  pub label: &'static str,
}

impl ColumnExtent {
  pub fn contains(&self, x: usize) -> bool {
    x >= self.x0 && x < self.x1
  }

  pub fn width(&self) -> usize {
    self.x1.saturating_sub(self.x0)
  }
}

/// Soil grid dimensions and contents.
#[derive(Debug, Clone)]
pub struct SoilGrid {
  pub width: usize,
  pub height: usize,
  pub cell_size: f32,
  pub mode: TerrainMode,
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
      mode,
      cells,
    }
  }

  /// The stretches of world the per-column metrics bin organisms into.
  ///
  /// The capillary test's six soil columns; for every other terrain, one
  /// column spanning the world, because there is no soil layout to split it
  /// by.
  pub fn columns(&self) -> Vec<ColumnExtent> {
    match self.mode {
      TerrainMode::CapillaryTest => capillary_columns(self.width)
        .into_iter()
        .map(|column| column.extent)
        .collect(),
      TerrainMode::Noise => vec![ColumnExtent {
        x0: 0,
        x1: self.width,
        label: "world",
      }],
    }
  }

  /// The soil cell column a world x coordinate falls in.
  pub fn cell_column(&self, x: f32) -> usize {
    ((x / self.cell_size).max(0.0) as usize).min(self.width.saturating_sub(1))
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

/// The soil a capillary-test column is made of: the composition at its left
/// edge and at its right edge. A pure column has the same at both, a gradient
/// column interpolates between them across its width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoilMix {
  pub sand: f32,
  pub silt: f32,
  pub clay: f32,
}

const SAND: SoilMix = SoilMix {
  sand: 1.0,
  silt: 0.0,
  clay: 0.0,
};
const SILT: SoilMix = SoilMix {
  sand: 0.0,
  silt: 1.0,
  clay: 0.0,
};
const CLAY: SoilMix = SoilMix {
  sand: 0.0,
  silt: 0.0,
  clay: 1.0,
};

/// One column of the capillary test: where it sits and what it is made of.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapillaryColumn {
  pub extent: ColumnExtent,
  pub left: SoilMix,
  pub right: SoilMix,
}

/// The capillary test's six soil columns, in cells.
///
/// The one arithmetic behind both the terrain generator and
/// [`SoilGrid::columns`], so the extents the metrics bin by cannot drift from
/// the densities the generator writes.
pub fn capillary_columns(width: usize) -> [CapillaryColumn; 6] {
  let gap = width / 40;
  let half_w = width / 2;

  // -- Left half: three pure columns with gaps --
  let pure_col_w = (half_w.saturating_sub(2 * gap)) / 3;
  // -- Right half: three gradient columns with gaps --
  let grad_x0 = half_w + gap;
  let grad_col_w = width.saturating_sub(grad_x0 + 2 * gap) / 3;

  let column = |x0: usize, x1: usize, label, left, right| CapillaryColumn {
    extent: ColumnExtent {
      x0,
      x1: x1.max(x0),
      label,
    },
    left,
    right,
  };
  [
    column(0, pure_col_w, "sand", SAND, SAND),
    column(pure_col_w + gap, 2 * pure_col_w + gap, "silt", SILT, SILT),
    column(2 * pure_col_w + 2 * gap, half_w, "clay", CLAY, CLAY),
    column(grad_x0, grad_x0 + grad_col_w, "sand_silt", SAND, SILT),
    column(
      grad_x0 + grad_col_w + gap,
      grad_x0 + 2 * grad_col_w + gap,
      "silt_clay",
      SILT,
      CLAY,
    ),
    column(
      grad_x0 + 2 * grad_col_w + 2 * gap,
      width,
      "sand_clay",
      SAND,
      CLAY,
    ),
  ]
}

/// Capillary tube test: shared water pool at the bottom, separate soil columns
/// above. Bottom 20%: empty (no soil) — water pool. Above 20%: soil columns
/// with air gaps between them. Left half: pure sand | silt | clay. Right half:
/// gradients.
pub fn capillary_test(width: usize, height: usize) -> SoilHost {
  let mut soil = SoilHost::new(width * height);

  let pool_h = (height as f32 * 0.2) as usize; // bottom 20% is open water pool
  let terrain_h = height;

  // The layout is a function of the width alone; build it once, not per row.
  let columns = capillary_columns(width);
  for y in pool_h..terrain_h {
    let row = y * width;
    for column in columns {
      let extent = column.extent;
      for x in extent.x0..extent.x1 {
        // `t` never reaches 1: the last cell of a gradient is one step short
        // of the right-hand mix, as the original three separate loops were.
        let t = (x - extent.x0) as f32 / extent.width() as f32;
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        soil.sand_density[x + row] = lerp(column.left.sand, column.right.sand);
        soil.silt_density[x + row] = lerp(column.left.silt, column.right.silt);
        soil.clay_density[x + row] = lerp(column.left.clay, column.right.clay);
      }
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

  /// Values printed by a probe that makes the same calls as the C++
  /// `reset_soil`, built against the FastNoiseLite and libstdc++ the C++ tree
  /// uses. Regenerating them is described in `crates/alife-sim/README.md`.
  // The literals are the C++ probe's printout verbatim, at the nine
  // significant digits that round-trip an `f32`; truncating them to what
  // clippy considers the shortest form would lose that provenance.
  #[test]
  #[allow(clippy::excessive_precision)]
  fn noise_terrain_matches_the_cpp() {
    use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};

    let mut rng = Mt19937_64::new(42);
    let s1 = rng.next_u64();
    let s2 = rng.next_u64();
    assert_eq!(s1, 13_930_160_852_258_120_406);
    assert_eq!(s2, 11_788_048_577_503_494_824);
    assert_eq!(s1 as i32, 1_860_559_574);
    assert_eq!(s2 as i32, -1_188_756_824);

    let height = 160usize;
    let width = 320usize;

    let mut heightmap_noise = FastNoiseLite::with_seed(s1 as i32);
    heightmap_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    heightmap_noise.set_fractal_type(Some(FractalType::FBm));
    heightmap_noise.set_fractal_octaves(Some(6));
    heightmap_noise.set_frequency(Some(1.0 / height as f32));

    // (x, exact f32 bit pattern printed by the C++ probe)
    let heightmap_expected: &[(usize, u32)] = &[
      (0, 0x0000_0000),
      (1, 0x3c4a_f38c),
      (7, 0xbd4b_00a4),
      (63, 0x3e19_18ab),
      (159, 0xbf02_ec3b),
      (319, 0x3f0c_4233),
    ];
    for (x, bits) in heightmap_expected {
      let value = heightmap_noise.get_noise_2d(*x as f32, 0.0);
      assert_eq!(value.to_bits(), *bits, "heightmap noise at x={x}");
    }

    let mut soil_noise = FastNoiseLite::with_seed(s2 as i32);
    soil_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
    soil_noise.set_fractal_type(Some(FractalType::FBm));
    soil_noise.set_fractal_octaves(Some(5));
    soil_noise.set_frequency(Some(0.01));

    // Printed with nine significant digits, which round-trips an `f32`.
    let (x, y) = (100.0f32, 3.0f32);
    assert_eq!(soil_noise.get_noise_3d(x, y, 0.0), 0.0828871354_f32);
    assert_eq!(
      soil_noise.get_noise_3d(x * 0.75, y, 300.0),
      -0.524124384_f32
    );
    assert_eq!(soil_noise.get_noise_3d(x * 0.5, y, 600.0), 0.1870583_f32);

    // The whole generator, including the column heights and the softmax.
    let soil = noise_terrain(width, height, 42);
    let id = 100 + 3 * width;
    assert_eq!(soil.sand_density[id], 0.00543980114_f32);
    assert_eq!(soil.silt_density[id], 3.58505318e-16_f32);
    assert_eq!(soil.clay_density[id], 0.994560242_f32);

    // Column heights: the topmost soil cell per column, from the C++ probe.
    for (x, h) in [
      (0usize, 16usize),
      (1, 23),
      (7, 32),
      (63, 81),
      (159, 30),
      (319, 28),
    ] {
      let filled = (0..height)
        .filter(|y| {
          let i = x + y * width;
          soil.sand_density[i] + soil.silt_density[i] + soil.clay_density[i] > 0.0
        })
        .count();
      assert_eq!(filled, h, "column height at x={x}");
    }
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

  /// The published extents are the layout: every non-gap cell belongs to
  /// exactly one column, every gap cell to none, and the densities inside a
  /// column are the ones its mix asks for.
  #[test]
  fn the_six_column_extents_tile_the_soil_the_layout_writes() {
    let (width, height) = (320usize, 160usize);
    let grid = SoilGrid::new(width, height, 0.1, TerrainMode::CapillaryTest, 0);
    let columns = grid.columns();
    assert_eq!(columns.len(), 6);
    assert_eq!(
      columns.iter().map(|c| c.label).collect::<Vec<_>>(),
      [
        "sand",
        "silt",
        "clay",
        "sand_silt",
        "silt_clay",
        "sand_clay"
      ]
    );

    // Ascending, disjoint, and inside the grid.
    for pair in columns.windows(2) {
      assert!(pair[0].x1 <= pair[1].x0, "{pair:?} overlap");
    }
    assert_eq!(columns[0].x0, 0);
    assert_eq!(columns[5].x1, width);
    // At this width there really are gaps, which is what makes the columns
    // separate habitats.
    assert!(columns.windows(2).any(|p| p[0].x1 < p[1].x0));

    // A row above the pool: soil exactly where a column is.
    let row = (height as f32 * 0.2) as usize + 1;
    for x in 0..width {
      let i = x + row * width;
      let total =
        grid.cells.sand_density[i] + grid.cells.silt_density[i] + grid.cells.clay_density[i];
      match columns.iter().position(|c| c.contains(x)) {
        Some(_) => assert!((total - 1.0).abs() < 1e-6, "cell {x} sums to {total}"),
        None => assert_eq!(total, 0.0, "gap cell {x} holds soil"),
      }
    }
    // Below the pool line, nothing at all.
    for x in 0..width {
      let i = x + (row - 2) * width;
      assert_eq!(grid.cells.sand_density[i], 0.0);
    }

    // The mixes: the pure columns are one type throughout, and each gradient
    // runs from its left type to (one step short of) its right type.
    for column in capillary_columns(width) {
      let (x0, x1) = (column.extent.x0, column.extent.x1);
      let at = |x: usize| {
        let i = x + row * width;
        SoilMix {
          sand: grid.cells.sand_density[i],
          silt: grid.cells.silt_density[i],
          clay: grid.cells.clay_density[i],
        }
      };
      assert_eq!(
        at(x0),
        column.left,
        "{} at its left edge",
        column.extent.label
      );
      if column.left == column.right {
        assert_eq!(
          at(x1 - 1),
          column.left,
          "{} is not pure",
          column.extent.label
        );
      } else {
        let last = at(x1 - 1);
        let t = (column.extent.width() - 1) as f32 / column.extent.width() as f32;
        assert!(
          (last.sand - (column.left.sand + (column.right.sand - column.left.sand) * t)).abs()
            < 1e-6
        );
        assert!(
          (last.clay - (column.left.clay + (column.right.clay - column.left.clay) * t)).abs()
            < 1e-6
        );
      }
    }
  }

  #[test]
  fn every_other_terrain_is_one_column_spanning_the_world() {
    let grid = SoilGrid::new(64, 32, 0.1, TerrainMode::Noise, 42);
    let columns = grid.columns();
    assert_eq!(columns.len(), 1);
    assert_eq!(columns[0].x0, 0);
    assert_eq!(columns[0].x1, 64);
    assert_eq!(columns[0].label, "world");
    assert!(columns[0].contains(0) && columns[0].contains(63));
    assert!(!columns[0].contains(64));
  }

  #[test]
  fn a_narrow_world_has_no_gaps_between_its_columns() {
    // `gap` is `width / 40`, so a grid under 40 cells wide tiles exactly.
    // The metrics tests lean on that: every organism lands in a column.
    let grid = SoilGrid::new(30, 20, 0.1, TerrainMode::CapillaryTest, 0);
    let columns = grid.columns();
    assert!(
      columns.windows(2).all(|p| p[0].x1 == p[1].x0),
      "{columns:?} should tile"
    );
    assert_eq!(grid.cell_column(0.55), 5);
    assert_eq!(columns.iter().position(|c| c.contains(5)), Some(1));
  }
}
