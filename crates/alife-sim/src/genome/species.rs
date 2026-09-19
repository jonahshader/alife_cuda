//! Genome distance over the discrete section.
//!
//! `docs/organism.md`: "Species are defined by distance in this section", and
//! that distance is "number of differing part types plus normalized identity
//! distance, computed on the host for metrics". Host-side because it feeds the
//! evolutionary metrics, which sample rather than run every step.
//!
//! An absent record counts as a type of its own, so a limb one organism has
//! and the other does not is one differing record — the same cost as a limb
//! both have at different types.

use super::population::{LimbHost, Population};

/// Distance between two organisms' discrete sections.
///
/// Differing part types, plus the mean L2 distance between identity vectors
/// over the limbs both organisms have, all divided by `max_limbs`. So one
/// added limb costs `1 / max_limbs`, and the count term dominates the
/// identity term whenever the bodies differ at all.
pub fn species_distance(limbs: &LimbHost, max_limbs: usize, a: usize, b: usize) -> f32 {
  assert!(max_limbs > 0);
  let mut differing = 0usize;
  let mut identity_sum = 0.0f32;
  let mut shared = 0usize;

  for l in 0..max_limbs {
    let (ia, ib) = (a * max_limbs + l, b * max_limbs + l);
    let (ta, tb) = (limbs.part_type[ia], limbs.part_type[ib]);
    if ta != tb {
      differing += 1;
    }
    if ta.is_present() && tb.is_present() {
      let (va, vb) = (limbs.identity[ia], limbs.identity[ib]);
      identity_sum += va
        .iter()
        .zip(vb.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt();
      shared += 1;
    }
  }

  let mean_identity = if shared > 0 {
    identity_sum / shared as f32
  } else {
    0.0
  };
  (differing as f32 + mean_identity) / max_limbs as f32
}

impl Population {
  /// [`species_distance`] between two of this population's slots.
  pub fn species_distance(&self, a: usize, b: usize) -> f32 {
    species_distance(&self.limbs, self.max_limbs, a, b)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SimParams;
  use crate::genome::population::PartType;
  use crate::genome::{BrainShape, Genome};
  use cubecl::prelude::Runtime;
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  const MAX_LIMBS: usize = 16;

  fn population() -> Population {
    let params = SimParams {
      max_organisms: 4,
      max_limbs: MAX_LIMBS as i32,
      ..SimParams::default()
    };
    let shape = BrainShape::from_params(&params);
    let client = CpuRuntime::client(&CpuDevice);
    let mut pop = Population::new(&client, &params, shape);
    for slot in 0..4 {
      // The same founder in every slot, so a test changes exactly what it
      // means to change.
      pop.write_genome(slot, &Genome::seed_plant(&shape, MAX_LIMBS, 0, 4));
    }
    pop
  }

  #[test]
  fn an_organism_is_zero_from_itself() {
    let pop = population();
    for slot in 0..4 {
      assert_eq!(pop.species_distance(slot, slot), 0.0);
    }
    // Identical genomes in different slots, too.
    assert_eq!(pop.species_distance(0, 1), 0.0);
  }

  #[test]
  fn distance_is_symmetric() {
    let mut pop = population();
    let i = pop.limb_index(1, 3);
    pop.limbs.part_type[i] = PartType::Leaf;
    pop.limbs.identity[i] = [0.3, -1.0, 2.0, 0.0];
    let j = pop.limb_index(2, 1);
    pop.limbs.identity[j] = [5.0, 5.0, 5.0, 5.0];

    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
      assert_eq!(pop.species_distance(a, b), pop.species_distance(b, a));
    }
  }

  #[test]
  fn one_added_limb_costs_one_over_max_limbs() {
    let mut pop = population();
    let i = pop.limb_index(1, 5);
    pop.limbs.part_type[i] = PartType::Leaf;
    pop.limbs.parent[i] = 1;
    // The new limb is present in only one of the two, so it contributes to
    // the count term and not to the identity term.
    assert_eq!(pop.species_distance(0, 1), 1.0 / MAX_LIMBS as f32);
  }

  #[test]
  fn a_retype_costs_the_same_as_an_add() {
    let mut pop = population();
    let i = pop.limb_index(1, 5);
    pop.limbs.part_type[i] = PartType::Leaf;
    let added = pop.species_distance(0, 1);
    let j = pop.limb_index(2, 2);
    pop.limbs.part_type[j] = PartType::Seed;
    assert_eq!(pop.species_distance(0, 2), added);
  }

  #[test]
  fn identity_alone_moves_the_distance_below_one_limb() {
    let mut pop = population();
    let i = pop.limb_index(1, 0);
    let mut shifted = pop.limbs.identity[i];
    shifted[0] += 3.0;
    pop.limbs.identity[i] = shifted;

    // Three limbs present in both, one of them 3.0 away: mean 1.0, over 16.
    let d = pop.species_distance(0, 1);
    assert!((d - (3.0 / 3.0) / MAX_LIMBS as f32).abs() < 1e-6, "{d}");
    assert!(d < 1.0 / MAX_LIMBS as f32 + 1e-6);
  }

  #[test]
  fn two_empty_organisms_are_zero_apart() {
    let pop = {
      let params = SimParams {
        max_organisms: 2,
        max_limbs: MAX_LIMBS as i32,
        ..SimParams::default()
      };
      let client = CpuRuntime::client(&CpuDevice);
      Population::new(&client, &params, BrainShape::from_params(&params))
    };
    assert_eq!(pop.species_distance(0, 1), 0.0);
  }
}
