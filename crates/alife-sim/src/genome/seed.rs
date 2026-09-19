//! The founder genome: the minimal plant a run starts from.
//!
//! Nothing here runs on the device. Founders are made once, before the first
//! step, and written into population slots; every genome after that is a
//! mutated copy made by `super::mutate` on the device.

use std::f32::consts::{FRAC_PI_2, PI};

use super::population::{Genome, LimbRecord, PartType};
use super::shape::{BrainShape, IDENTITY_DIM, InitKind};
use crate::rng::{gaussian_ref, threefry4x32_20_ref};

/// Threefry key words 1..3 for the founder streams, so a founder's brain, its
/// identity vectors and anything added later never share a draw.
const STREAM_BRAIN: u32 = 0;
const STREAM_IDENTITY: u32 = 1;

/// The root limb is its own parent, so its grow angle is read in the world
/// frame rather than against a parent segment: 0 is +x, angles increase
/// counter-clockwise. A root grows down.
pub const ROOT_DOWN: f32 = -FRAC_PI_2;

/// Every other limb's grow angle is measured against its parent segment's
/// direction, so a limb that reverses its parent uses `PI` and one that
/// continues it uses 0.
pub const REVERSE_PARENT: f32 = PI;
pub const ALONG_PARENT: f32 = 0.0;

impl Genome {
  /// The minimal plant: a root in the soil, a stem back up out of it, and one
  /// leaf on the stem.
  ///
  /// Deterministic in `(slot, seed)`: the brain and the identity vectors come
  /// from a Threefry stream keyed by both, so two runs at one seed produce
  /// identical founders and slot `n` is always the same founder.
  pub fn seed_plant(shape: &BrainShape, max_limbs: usize, slot: u32, seed: u64) -> Self {
    assert!(
      max_limbs >= 3,
      "the seed plant needs three limb records, got {max_limbs}"
    );
    let mut genome = Genome::empty(shape, max_limbs);

    genome.limbs[0] = LimbRecord {
      part_type: PartType::Root,
      length: 2,
      // Its own parent: that is what terminates a walk to the root.
      parent: 0,
      child_slot: 0,
      grow_angle: ROOT_DOWN,
      identity: [0.0; IDENTITY_DIM],
    };
    genome.limbs[1] = LimbRecord {
      part_type: PartType::Stem,
      length: 3,
      parent: 0,
      child_slot: 0,
      grow_angle: REVERSE_PARENT,
      identity: [0.0; IDENTITY_DIM],
    };
    genome.limbs[2] = LimbRecord {
      part_type: PartType::Leaf,
      length: 1,
      parent: 1,
      child_slot: 0,
      grow_angle: ALONG_PARENT,
      identity: [0.0; IDENTITY_DIM],
    };

    fill_brain(&mut genome.brain, shape, slot, seed);
    for (limb, record) in genome.limbs.iter_mut().enumerate() {
      if record.part_type.is_present() {
        record.identity = identity_vector(slot, seed, limb as u32);
      }
    }
    genome
  }
}

/// Draw one slice's worth of Gaussians at the sigma its [`InitKind`] asks
/// for: `1/sqrt(fan_in)` for a weight, zero for a bias, 1 for `latent_init`.
///
/// One Threefry block per parameter, countered by the parameter's index in
/// the row, so the draw for parameter `j` does not move when a slice before
/// it changes size.
pub fn fill_brain(row: &mut [f32], shape: &BrainShape, slot: u32, seed: u64) {
  assert_eq!(row.len(), shape.param_count());
  let key = [slot, STREAM_BRAIN, seed as u32, (seed >> 32) as u32];
  for slice in shape.slices() {
    let sigma = slice.sigma();
    for j in slice.range.clone() {
      row[j] = if slice.init == InitKind::Bias {
        0.0
      } else {
        sigma * draw(key, j as u32)
      };
    }
  }
}

/// A limb's identity vector: Gaussian, sigma 1.
pub fn identity_vector(slot: u32, seed: u64, limb: u32) -> [f32; IDENTITY_DIM] {
  let key = [slot, STREAM_IDENTITY, seed as u32, (seed >> 32) as u32];
  let mut out = [0.0; IDENTITY_DIM];
  for (k, v) in out.iter_mut().enumerate() {
    *v = draw(key, limb * IDENTITY_DIM as u32 + k as u32);
  }
  out
}

fn draw(key: [u32; 4], index: u32) -> f32 {
  let r = threefry4x32_20_ref([index, 0, 0, 0], key);
  gaussian_ref(r[0], r[1])
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SimParams;

  fn shape() -> BrainShape {
    BrainShape::default()
  }

  #[test]
  fn the_seed_plant_is_a_root_a_stem_and_a_leaf() {
    let genome = Genome::seed_plant(&shape(), 16, 0, 1);
    assert_eq!(genome.limbs.len(), 16);

    assert_eq!(genome.limbs[0].part_type, PartType::Root);
    assert_eq!(genome.limbs[0].length, 2);
    assert_eq!(genome.limbs[0].parent, 0);
    assert_eq!(genome.limbs[0].grow_angle, ROOT_DOWN);

    assert_eq!(genome.limbs[1].part_type, PartType::Stem);
    assert_eq!(genome.limbs[1].length, 3);
    assert_eq!(genome.limbs[1].parent, 0);
    assert_eq!(genome.limbs[1].child_slot, 0);

    assert_eq!(genome.limbs[2].part_type, PartType::Leaf);
    assert_eq!(genome.limbs[2].length, 1);
    assert_eq!(genome.limbs[2].parent, 1);
    assert_eq!(genome.limbs[2].child_slot, 0);

    for record in &genome.limbs[3..] {
      assert_eq!(record.part_type, PartType::Absent);
      assert_eq!(record.identity, [0.0; IDENTITY_DIM]);
    }
  }

  #[test]
  fn the_same_slot_and_seed_give_the_same_founder() {
    let a = Genome::seed_plant(&shape(), 16, 5, 42);
    let b = Genome::seed_plant(&shape(), 16, 5, 42);
    assert_eq!(a, b);

    assert_ne!(Genome::seed_plant(&shape(), 16, 6, 42), a);
    assert_ne!(Genome::seed_plant(&shape(), 16, 5, 43), a);
  }

  #[test]
  fn biases_are_zero_and_weights_scale_with_fan_in() {
    let shape = shape();
    let genome = Genome::seed_plant(&shape, 16, 0, 7);
    for slice in shape.slices() {
      let values = &genome.brain[slice.range.clone()];
      match slice.init {
        InitKind::Bias => assert!(values.iter().all(|v| *v == 0.0), "{}", slice.name),
        _ => {
          let n = values.len() as f32;
          let mean = values.iter().sum::<f32>() / n;
          let sigma = (values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
          // A 256-to-1024 element sample, so the tolerance is generous;
          // what it catches is a slice initialized at the wrong scale.
          let expected = slice.sigma();
          assert!(
            (sigma / expected - 1.0).abs() < 0.25,
            "{}: sigma {sigma} against the expected {expected}",
            slice.name
          );
        }
      }
    }
  }

  #[test]
  fn identity_vectors_differ_between_limbs() {
    let genome = Genome::seed_plant(&shape(), 16, 3, 9);
    assert_ne!(genome.limbs[0].identity, genome.limbs[1].identity);
    assert_ne!(genome.limbs[1].identity, genome.limbs[2].identity);
    assert!(
      genome.limbs[0]
        .identity
        .iter()
        .all(|v| v.is_finite() && *v != 0.0)
    );
  }

  #[test]
  fn a_founder_installs_into_a_population_slot() {
    use super::super::population::Population;
    use cubecl::prelude::Runtime;
    use cubecl_cpu::{CpuDevice, CpuRuntime};

    let params = SimParams {
      max_organisms: 4,
      ..SimParams::default()
    };
    let shape = BrainShape::from_params(&params);
    let client = CpuRuntime::client(&CpuDevice);
    let mut pop = Population::new(&client, &params, shape);

    let genome = Genome::seed_plant(&shape, pop.max_limbs, 2, 11);
    pop.write_genome(2, &genome);

    assert_eq!(pop.brain_row(2), genome.brain.as_slice());
    assert_eq!(
      pop.limbs.part_type[pop.limb_index(2, 1)],
      PartType::Stem,
      "the stem landed in the right record"
    );
    // Slot 2 only: nothing leaks into a neighbour.
    assert!(pop.brain_row(1).iter().all(|v| *v == 0.0));
    assert_eq!(pop.latent_row(2), &genome.brain[shape.latent_init()]);
  }
}
