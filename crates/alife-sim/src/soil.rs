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
  /// Capillary tube test: a shared water pool under separate soil columns,
  /// each filled to the very top of the world. What every
  /// `resources/parity/*.bin` was generated from, so it never changes.
  CapillaryTest,
  /// The same layout with air above the columns, so a plant anchored on a
  /// column's surface has somewhere to grow: the soil-specialization
  /// experiment's terrain.
  CapillaryField,
}

impl TerrainMode {
  pub fn from_flag(mode: i32) -> Self {
    // The C++ `switch` treats everything that is not 1 as the noise mode, and
    // mode 2 is this tree's own addition.
    match mode {
      1 => TerrainMode::CapillaryTest,
      2 => TerrainMode::CapillaryField,
      _ => TerrainMode::Noise,
    }
  }

  /// Whether this mode lays soil out as the six capillary columns.
  pub fn is_capillary(self) -> bool {
    matches!(
      self,
      TerrainMode::CapillaryTest | TerrainMode::CapillaryField
    )
  }
}

/// A pure soil composition, as `--uniform-soil` names one.
///
/// [`UniformSoil::None`] is the absence of the control, which is why it is the
/// default: the columns keep the compositions [`capillary_columns`] gives them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UniformSoil {
  #[default]
  None,
  Sand,
  Silt,
  Clay,
}

impl UniformSoil {
  /// The mix every column is made of, or `None` when the control is off.
  pub fn mix(self) -> Option<SoilMix> {
    match self {
      UniformSoil::None => None,
      UniformSoil::Sand => Some(SAND),
      UniformSoil::Silt => Some(SILT),
      UniformSoil::Clay => Some(CLAY),
    }
  }

  pub fn name(self) -> &'static str {
    match self {
      UniformSoil::None => "none",
      UniformSoil::Sand => "sand",
      UniformSoil::Silt => "silt",
      UniformSoil::Clay => "clay",
    }
  }
}

impl std::fmt::Display for UniformSoil {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.name())
  }
}

impl std::str::FromStr for UniformSoil {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "none" => Ok(UniformSoil::None),
      "sand" => Ok(UniformSoil::Sand),
      "silt" => Ok(UniformSoil::Silt),
      "clay" => Ok(UniformSoil::Clay),
      other => Err(format!("expected sand, silt, clay or none, got {other}")),
    }
  }
}

/// Everything about the soil that is decided before the first step.
///
/// The two controls apply to [`TerrainMode::CapillaryField`] only; the
/// parameter validator rejects them on any other mode, so mode 1 stays the
/// terrain the parity references were generated from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainSpec {
  pub mode: TerrainMode,
  /// Fraction of the world height the mode-2 columns rise to. The pool below
  /// them and the gaps between them are mode 1's.
  pub column_top: f32,
  /// Permutation control: which composition sits in which column position.
  /// 0 is the identity.
  pub permutation_seed: u64,
  /// Isolation control: one pure composition in every column.
  pub uniform: UniformSoil,
}

impl TerrainSpec {
  pub fn from_params(params: &crate::SimParams) -> Self {
    Self {
      mode: TerrainMode::from_flag(params.terrain_mode),
      column_top: params.column_top,
      permutation_seed: params.soil_permutation.max(0) as u64,
      uniform: params.uniform_soil,
    }
  }

  /// The plain terrain of one mode, both controls off — what every caller that
  /// is not running the experiment wants.
  pub fn plain(mode: TerrainMode) -> Self {
    Self {
      mode,
      column_top: crate::SimParams::default().column_top,
      permutation_seed: 0,
      uniform: UniformSoil::None,
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
  pub spec: TerrainSpec,
  pub cells: SoilHost,
}

impl SoilGrid {
  pub fn new(width: usize, height: usize, cell_size: f32, spec: TerrainSpec, seed: u64) -> Self {
    let cells = match spec.mode {
      TerrainMode::CapillaryTest => capillary_test(width, height),
      TerrainMode::CapillaryField => capillary_field(width, height, spec),
      TerrainMode::Noise => noise_terrain(width, height, seed),
    };
    Self {
      width,
      height,
      cell_size,
      spec,
      cells,
    }
  }

  pub fn mode(&self) -> TerrainMode {
    self.spec.mode
  }

  /// The stretches of world the per-column metrics bin organisms into.
  ///
  /// The capillary layout's six soil columns, in ascending x; for every other
  /// terrain, one column spanning the world, because there is no soil layout
  /// to split it by. A column's label names the composition standing there,
  /// not the position, so a permuted run still reports the sand column as
  /// `sand` wherever the permutation put it.
  pub fn columns(&self) -> Vec<ColumnExtent> {
    if self.spec.mode.is_capillary() {
      capillary_columns_for(self.width, self.spec)
        .into_iter()
        .map(|column| column.extent)
        .collect()
    } else {
      vec![ColumnExtent {
        x0: 0,
        x1: self.width,
        label: "world",
      }]
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

/// The six columns with the experiment's controls applied to them.
///
/// The extents are [`capillary_columns`]'s and never move: a control changes
/// *what stands where*, not where the habitats are. The label travels with the
/// composition, so the metrics keep calling the sand column `sand`.
pub fn capillary_columns_for(width: usize, spec: TerrainSpec) -> [CapillaryColumn; 6] {
  let base = capillary_columns(width);
  if spec.mode != TerrainMode::CapillaryField {
    return base;
  }

  // Isolation control: the same pure soil everywhere, so the only thing left
  // that separates the columns is the gaps between them. There is no
  // composition to name a column after any more, so the labels are positions.
  if let Some(mix) = spec.uniform.mix() {
    let mut out = base;
    for (position, column) in out.iter_mut().enumerate() {
      column.left = mix;
      column.right = mix;
      column.extent.label = POSITION_LABELS[position];
    }
    return out;
  }

  // Position control: the compositions are dealt out to the positions by a
  // permutation of 0..6, and each keeps its own label.
  let order = soil_permutation(spec.permutation_seed);
  let mut out = base;
  for (position, source) in order.into_iter().enumerate() {
    out[position].left = base[source].left;
    out[position].right = base[source].right;
    out[position].extent.label = base[source].extent.label;
  }
  out
}

/// Labels the isolation control uses, where no column has a composition of its
/// own to be named after.
const POSITION_LABELS: [&str; 6] = ["col0", "col1", "col2", "col3", "col4", "col5"];

/// Which composition stands in which column position: `order[position]` is the
/// index into [`capillary_columns`] whose soil and label move there.
///
/// Fisher–Yates over a SplitMix64 stream seeded with `seed`, walking the six
/// slots from the top down and swapping each with a uniform draw from the
/// slots at or below it. Seed 0 is the identity — it is the *absence* of the
/// control, not a permutation drawn from it — and every other seed goes
/// through the shuffle, so the same seed always deals the same layout without
/// depending on any other part of the run.
pub fn soil_permutation(seed: u64) -> [usize; 6] {
  let mut order = [0, 1, 2, 3, 4, 5];
  if seed == 0 {
    return order;
  }
  let mut state = seed;
  for i in (1..order.len()).rev() {
    // Modulo over a 64-bit draw: the bias against an `i + 1` of at most six is
    // below 2^-61, which no run will ever see.
    let j = (splitmix64(&mut state) % (i as u64 + 1)) as usize;
    order.swap(i, j);
  }
  order
}

/// SplitMix64, the reference stream. Self-contained by design: a permutation
/// has to be reproducible from its seed alone, with no simulation state in it.
fn splitmix64(state: &mut u64) -> u64 {
  *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
  let mut z = *state;
  z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
  z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
  z ^ (z >> 31)
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
  fill_columns(&mut soil, width, pool_h..terrain_h, &columns);
  soil
}

/// The capillary test with a ceiling: the columns stop at `column_top` of the
/// world height and everything above them is air, so a plant anchored on a
/// column's surface has somewhere to grow (`docs/organism.md`, decisions).
///
/// The pool below and the gaps between are mode 1's, cell for cell — the only
/// differences are where the soil stops and, when a control is on, which
/// composition stands in which column.
pub fn capillary_field(width: usize, height: usize, spec: TerrainSpec) -> SoilHost {
  let mut soil = SoilHost::new(width * height);

  let pool_h = (height as f32 * 0.2) as usize;
  // Clamped rather than validated away: a `column_top` under the pool line
  // leaves the columns empty, which is a world with no soil in it, not a
  // corrupt buffer.
  let terrain_h = ((height as f32 * spec.column_top) as usize).clamp(pool_h, height);

  let columns = capillary_columns_for(width, spec);
  fill_columns(&mut soil, width, pool_h..terrain_h, &columns);
  soil
}

/// Write the six columns' compositions into `rows`, the one place the mix
/// interpolation lives.
fn fill_columns(
  soil: &mut SoilHost,
  width: usize,
  rows: std::ops::Range<usize>,
  columns: &[CapillaryColumn; 6],
) {
  for y in rows {
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
    let grid = SoilGrid::new(
      width,
      height,
      0.1,
      TerrainSpec::plain(TerrainMode::CapillaryTest),
      0,
    );
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
    let grid = SoilGrid::new(64, 32, 0.1, TerrainSpec::plain(TerrainMode::Noise), 42);
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
    let grid = SoilGrid::new(
      30,
      20,
      0.1,
      TerrainSpec::plain(TerrainMode::CapillaryTest),
      0,
    );
    let columns = grid.columns();
    assert!(
      columns.windows(2).all(|p| p[0].x1 == p[1].x0),
      "{columns:?} should tile"
    );
    assert_eq!(grid.cell_column(0.55), 5);
    assert_eq!(columns.iter().position(|c| c.contains(5)), Some(1));
  }

  fn field(width: usize, height: usize, spec: TerrainSpec) -> SoilGrid {
    SoilGrid::new(width, height, 0.1, spec, 0)
  }

  /// Mode 2 is mode 1 with a ceiling: the same soil below `column_top`, air
  /// above it, and the same six columns to bin organisms into.
  #[test]
  fn mode_two_is_mode_one_below_the_column_top_and_empty_above() {
    let (width, height) = (320usize, 160usize);
    let spec = TerrainSpec::plain(TerrainMode::CapillaryField);
    let grid = field(width, height, spec);
    let reference = capillary_test(width, height);
    let top = (height as f32 * spec.column_top) as usize;
    assert_eq!(top, 88);

    for y in 0..height {
      for x in 0..width {
        let i = x + y * width;
        let cell = (
          grid.cells.sand_density[i],
          grid.cells.silt_density[i],
          grid.cells.clay_density[i],
        );
        if y < top {
          assert_eq!(
            cell,
            (
              reference.sand_density[i],
              reference.silt_density[i],
              reference.clay_density[i]
            ),
            "cell ({x}, {y}) differs from mode 1"
          );
        } else {
          assert_eq!(cell, (0.0, 0.0, 0.0), "cell ({x}, {y}) is above the top");
        }
      }
    }

    // The columns are unchanged, and there is now soil to stand on with air
    // over it — which is the whole point of the mode.
    let columns = grid.columns();
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
    assert_eq!(
      columns,
      field(
        width,
        height,
        TerrainSpec::plain(TerrainMode::CapillaryTest)
      )
      .columns()
    );
    assert_eq!(
      crate::bodies::soil_surface(&grid, 1.0),
      (top as f32 - 0.5) * 0.1
    );
  }

  /// The permutation control moves compositions between positions and takes
  /// their labels with them, so `columns()` still names the soil.
  #[test]
  fn a_permutation_moves_the_soil_and_its_label_together() {
    let (width, height) = (320usize, 160usize);
    let identity = TerrainSpec::plain(TerrainMode::CapillaryField);
    let mut permuted = identity;
    permuted.permutation_seed = 3;
    let order = soil_permutation(3);
    assert_ne!(order, [0, 1, 2, 3, 4, 5], "seed 3 should shuffle something");
    assert_eq!(soil_permutation(0), [0, 1, 2, 3, 4, 5]);
    // A permutation is a bijection: every composition is dealt exactly once.
    let mut seen = order;
    seen.sort_unstable();
    assert_eq!(seen, [0, 1, 2, 3, 4, 5]);
    // And it is a function of the seed alone.
    assert_eq!(soil_permutation(3), order);

    let base = capillary_columns(width);
    let moved = capillary_columns_for(width, permuted);
    let grid = field(width, height, permuted);
    let row = (height as f32 * 0.2) as usize + 1;
    for (position, column) in moved.iter().enumerate() {
      // The extents never move.
      assert_eq!(column.extent.x0, base[position].extent.x0);
      assert_eq!(column.extent.x1, base[position].extent.x1);
      // The soil and the label came from the source column together.
      let source = order[position];
      assert_eq!(column.left, base[source].left);
      assert_eq!(column.extent.label, base[source].extent.label);
      let i = column.extent.x0 + row * width;
      assert_eq!(grid.cells.sand_density[i], base[source].left.sand);
      assert_eq!(grid.cells.clay_density[i], base[source].left.clay);
    }
    assert_eq!(
      grid.columns().iter().map(|c| c.label).collect::<Vec<_>>(),
      order.map(|s| base[s].extent.label).to_vec()
    );
  }

  /// The isolation control: one composition everywhere, so a column has no
  /// soil identity left to be labelled by and is named for its position.
  #[test]
  fn the_uniform_control_fills_every_column_with_one_soil() {
    let (width, height) = (320usize, 160usize);
    let mut spec = TerrainSpec::plain(TerrainMode::CapillaryField);
    spec.uniform = UniformSoil::Silt;
    let grid = field(width, height, spec);
    assert_eq!(
      grid.columns().iter().map(|c| c.label).collect::<Vec<_>>(),
      ["col0", "col1", "col2", "col3", "col4", "col5"]
    );

    let row = (height as f32 * 0.2) as usize + 1;
    let columns = grid.columns();
    for x in 0..width {
      let i = x + row * width;
      let expected = if columns.iter().any(|c| c.contains(x)) {
        (0.0, 1.0, 0.0)
      } else {
        (0.0, 0.0, 0.0)
      };
      assert_eq!(
        (
          grid.cells.sand_density[i],
          grid.cells.silt_density[i],
          grid.cells.clay_density[i]
        ),
        expected,
        "cell {x} of the uniform terrain"
      );
    }
    // The gaps are still there: what the control removes is the soil
    // difference, not the spatial separation.
    assert!(columns.windows(2).any(|p| p[0].x1 < p[1].x0));
  }

  /// The controls are mode 2's; mode 1 ignores them, whatever a caller that
  /// got past the validator asks for.
  #[test]
  fn mode_one_is_untouched_by_the_controls() {
    let (width, height) = (320usize, 160usize);
    let mut spec = TerrainSpec::plain(TerrainMode::CapillaryTest);
    spec.permutation_seed = 7;
    spec.uniform = UniformSoil::Clay;
    spec.column_top = 0.3;
    let grid = field(width, height, spec);
    let reference = capillary_test(width, height);
    assert_eq!(grid.cells, reference);
    assert_eq!(
      grid.columns(),
      field(
        width,
        height,
        TerrainSpec::plain(TerrainMode::CapillaryTest)
      )
      .columns()
    );
  }
}
