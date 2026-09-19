//! Mutation: what a birth does to the copy of its parent's genome.
//!
//! From `docs/organism.md`'s *Evolution operators* and *Genome buffers*. The
//! child's row starts as a copy of the parent's; the copy and the
//! perturbation are the same pass, so nothing else has to run between them.
//!
//! Two launches rather than one. The continuous section is one independent
//! draw per parameter and parallelizes over `newborns x param_count`; the
//! discrete section is a short serial walk over one organism's limb records,
//! one unit per newborn. Splitting them keeps 18,890 parameters off a single
//! unit's serial loop.
//!
//! Every draw is Threefry keyed by `(child slot, step, stream, seed)` and
//! countered by the draw's index within its stream, so a mutation depends on
//! nothing but the child's slot and the step — not on how many organisms were
//! born alongside it, nor on the order the newborn list happens to be in.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::population::{LimbArgs, PartType, Population, PopulationDevice, limb_args};
use crate::SimParams;
use crate::kernels::{CUBE_DIM, cube_count, whole};
use crate::rng::{gaussian, gaussian_ref, threefry4x32_20, threefry4x32_20_ref, u01, u01_ref};

/// Threefry stream ids. Each one is its own counter space, so the brain draws
/// can grow without moving the structural ones.
const STREAM_BRAIN: u32 = 0;
const STREAM_ANGLE: u32 = 1;
const STREAM_IDENTITY: u32 = 2;
const STREAM_STRUCT: u32 = 3;
const STREAM_NEW_IDENTITY: u32 = 4;

/// Slots of the mutation knob buffer.
pub const K_MUTATION_SIGMA: u32 = 0;
pub const K_IDENTITY_SIGMA: u32 = 1;
pub const K_ANGLE_SIGMA: u32 = 2;
pub const K_STRUCTURAL_RATE: u32 = 3;
pub const KNOB_COUNT: usize = 4;

/// Slots of the 4-word RNG buffer: the step, and the run's seed folded to one
/// word.
const R_STEP: u32 = 0;
const R_SEED: u32 = 1;

/// The three structural edits, in the order a uniform roll selects them.
const EDIT_ADD: u32 = 0;
const EDIT_REMOVE: u32 = 1;

/// The shapes the mutation kernels bake in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MutateCfg {
  pub param_count: u32,
  pub max_limbs: u32,
  /// Cap on a limb's `length`, so a structural add cannot ask for a chain the
  /// particle system has no room for.
  pub max_particles_per_limb: u32,
  /// Floats per identity vector; [`super::IDENTITY_DIM`].
  pub identity_dim: u32,
}

impl MutateCfg {
  pub fn new(pop: &Population) -> Self {
    // `pick(roll, n)` computes `n - 1`, so a zero here would wrap on the
    // device; `SimParams::validate` floors both, this is the backstop for
    // callers that build a population from unvalidated params.
    assert!(
      pop.max_limbs >= 1 && pop.max_particles_per_limb >= 1,
      "max_limbs and max_particles_per_limb must be at least 1"
    );
    Self {
      param_count: pop.shape.param_count() as u32,
      max_limbs: pop.max_limbs as u32,
      max_particles_per_limb: pop.max_particles_per_limb as u32,
      identity_dim: super::IDENTITY_DIM as u32,
    }
  }
}

/// One birth: which slot the child lands in, and whose genome it copies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Birth {
  pub child: u32,
  pub parent: u32,
}

// --- Device helpers ---

/// One Gaussian from stream `stream`, draw `index`, for child slot `child`.
#[cube]
fn draw_gaussian(child: u32, step: u32, seed: u32, #[comptime] stream: u32, index: u32) -> f32 {
  let r = threefry4x32_20(index, 0u32, 0u32, 0u32, child, step, stream, seed);
  gaussian(r.x0, r.x1)
}

/// `floor(roll * n)`, clamped into `0..n`. `u01` is closed at one, so the
/// clamp is reachable and not merely defensive.
#[cube]
fn pick(roll: f32, n: u32) -> u32 {
  let mut v = (roll * n as f32) as u32;
  if v >= n {
    v = n - 1u32;
  }
  v
}

// --- The continuous section ---

/// Copy each newborn's parent brain row and perturb it.
///
/// One unit per `(newborn, parameter)`. The parent rows are only read and the
/// child rows only written, and a newborn's slot was free — so it was nobody's
/// parent — which is what makes the in-place read and write safe.
#[cube(launch)]
pub fn mutate_brain(
  brain: &mut [f32],
  newborns: &[u32],
  parents: &[u32],
  count: &[u32],
  rng: &[u32],
  knobs: &[f32],
  #[comptime] cfg: MutateCfg,
) {
  let t = ABSOLUTE_POS;
  let params = comptime!(cfg.param_count as usize);
  if t >= count[0] as usize * params {
    terminate!();
  }
  let entry = t / params;
  let j = t % params;

  let child = newborns[entry] as usize;
  let parent = parents[entry] as usize;
  let noise = draw_gaussian(
    newborns[entry],
    rng[R_STEP as usize],
    rng[R_SEED as usize],
    STREAM_BRAIN,
    j as u32,
  );
  brain[child * params + j] = brain[parent * params + j] + knobs[K_MUTATION_SIGMA as usize] * noise;
}

// --- The discrete section ---

/// Copy each newborn's parent limb records, perturb them, and with
/// probability `structural_rate` apply exactly one structural edit.
///
/// One unit per newborn, serial over that organism's `max_limbs` records.
#[cube(launch)]
#[allow(clippy::needless_range_loop)]
pub fn mutate_structure(
  limbs: &mut LimbArgs,
  newborns: &[u32],
  parents: &[u32],
  count: &[u32],
  rng: &[u32],
  knobs: &[f32],
  #[comptime] cfg: MutateCfg,
) {
  let e = ABSOLUTE_POS;
  if e >= count[0] as usize {
    terminate!();
  }
  let ml = comptime!(cfg.max_limbs as usize);
  let dim = comptime!(cfg.identity_dim as usize);
  let child = newborns[e];
  let cbase = child as usize * ml;
  let pbase = parents[e] as usize * ml;
  let step = rng[R_STEP as usize];
  let seed = rng[R_SEED as usize];

  for l in 0..ml {
    limbs.part_type[cbase + l] = limbs.part_type[pbase + l];
    limbs.length[cbase + l] = limbs.length[pbase + l];
    limbs.parent[cbase + l] = limbs.parent[pbase + l];
    limbs.child_slot[cbase + l] = limbs.child_slot[pbase + l];
    limbs.grow_angle[cbase + l] = limbs.grow_angle[pbase + l];
    for k in 0..dim {
      limbs.identity[(cbase + l) * dim + k] = limbs.identity[(pbase + l) * dim + k];
    }
  }

  // Only records that hold a limb are perturbed: an absent record's angle and
  // identity are not the genome of anything, and a structural add overwrites
  // them outright.
  for l in 0..ml {
    if limbs.part_type[cbase + l] != 0u32 {
      limbs.grow_angle[cbase + l] +=
        knobs[K_ANGLE_SIGMA as usize] * draw_gaussian(child, step, seed, STREAM_ANGLE, l as u32);
      for k in 0..dim {
        let index = (l * dim + k) as u32;
        limbs.identity[(cbase + l) * dim + k] += knobs[K_IDENTITY_SIGMA as usize]
          * draw_gaussian(child, step, seed, STREAM_IDENTITY, index);
      }
    }
  }

  // Both blocks are drawn whether or not the edit happens, so the stream a
  // child consumes does not depend on its own rolls.
  let b0 = threefry4x32_20(0u32, 0u32, 0u32, 0u32, child, step, STREAM_STRUCT, seed);
  let b1 = threefry4x32_20(1u32, 0u32, 0u32, 0u32, child, step, STREAM_STRUCT, seed);
  if u01(b0.x0) >= knobs[K_STRUCTURAL_RATE as usize] {
    terminate!();
  }
  let edit = pick(u01(b0.x1), 3u32);
  let pick_roll = u01(b0.x2);
  let type_roll = u01(b1.x0);
  let length_roll = u01(b1.x1);
  let angle_roll = u01(b1.x2);

  if edit == EDIT_ADD {
    // The first absent record, and a uniformly chosen present limb to hang it
    // from. Counting down leaves `slot` at the lowest absent index.
    let mut slot = 0u32;
    let mut has_slot = false;
    let mut present = 0u32;
    for l in 0..ml {
      let back = ml - 1 - l;
      if limbs.part_type[cbase + back] == 0u32 {
        slot = back as u32;
        has_slot = true;
      }
      if limbs.part_type[cbase + l] != 0u32 {
        present += 1u32;
      }
    }
    if present > 0u32 && has_slot {
      let target = pick(pick_roll, present);
      let mut seen = 0u32;
      let mut chosen = 0u32;
      for l in 0..ml {
        if limbs.part_type[cbase + l] != 0u32 {
          if seen == target {
            chosen = l as u32;
          }
          seen += 1u32;
        }
      }
      // The root is its own parent, so it is not one of its own children.
      let mut kids = 0u32;
      for l in 0..ml {
        if l as u32 != chosen
          && limbs.part_type[cbase + l] != 0u32
          && limbs.parent[cbase + l] == chosen
        {
          kids += 1u32;
        }
      }
      let at = cbase + slot as usize;
      limbs.part_type[at] = 1u32 + pick(type_roll, 4u32);
      limbs.length[at] = 1u32 + pick(length_roll, cfg.max_particles_per_limb);
      limbs.parent[at] = chosen;
      limbs.child_slot[at] = kids;
      limbs.grow_angle[at] = (angle_roll * 2.0f32 - 1.0f32) * core::f32::consts::PI;
      for k in 0..dim {
        limbs.identity[at * dim + k] =
          draw_gaussian(child, step, seed, STREAM_NEW_IDENTITY, k as u32);
      }
    }
  } else {
    // Remove and retype both act on a uniformly chosen present non-root limb.
    // The root is neither removed nor retyped: an organism without one has no
    // anchor and no frame to measure any other limb's angle against.
    let mut candidates = 0u32;
    for l in 1..ml {
      if limbs.part_type[cbase + l] != 0u32 {
        candidates += 1u32;
      }
    }
    if candidates > 0u32 {
      let target = pick(pick_roll, candidates);
      let mut seen = 0u32;
      let mut victim = 0u32;
      for l in 1..ml {
        if limbs.part_type[cbase + l] != 0u32 {
          if seen == target {
            victim = l as u32;
          }
          seen += 1u32;
        }
      }
      if edit == EDIT_REMOVE {
        limbs.part_type[cbase + victim as usize] = 0u32;
        // A limb whose parent went absent goes with it. A parent can sit at
        // any index, so one sweep is not enough; `ml` sweeps cover the
        // longest possible chain.
        for _ in 0..ml {
          for l in 1..ml {
            let mut p = limbs.parent[cbase + l] as usize;
            if p >= ml {
              p = ml - 1;
            }
            if limbs.part_type[cbase + l] != 0u32 && limbs.part_type[cbase + p] == 0u32 {
              limbs.part_type[cbase + l] = 0u32;
            }
          }
        }
      } else {
        limbs.part_type[cbase + victim as usize] = 2u32 + pick(type_roll, 3u32);
      }
    }
  }
}

// --- Host side ---

/// The per-launch inputs the two kernels share.
pub struct MutateInputs {
  pub newborns: Handle,
  pub parents: Handle,
  pub count: Handle,
  pub rng: Handle,
  pub knobs: Handle,
  /// Entries the newborn list can hold — the launch bound. The live count
  /// lives on the device, so the host does not have to know it.
  pub capacity: usize,
}

impl MutateInputs {
  pub fn upload<R: Runtime>(
    client: &ComputeClient<R>,
    births: &[Birth],
    step: u32,
    seed: u64,
    params: &SimParams,
  ) -> Self {
    let newborns: Vec<u32> = births.iter().map(|b| b.child).collect();
    let parents: Vec<u32> = births.iter().map(|b| b.parent).collect();
    Self {
      newborns: client.create_from_slice(bytemuck::cast_slice(&newborns)),
      parents: client.create_from_slice(bytemuck::cast_slice(&parents)),
      count: client.create_from_slice(bytemuck::cast_slice(&[births.len() as u32])),
      rng: client.create_from_slice(bytemuck::cast_slice(&rng_words(step, seed))),
      knobs: client.create_from_slice(bytemuck::cast_slice(&knob_values(params))),
      capacity: births.len(),
    }
  }
}

/// The 4-word RNG buffer. The seed folds to one word because the key already
/// spends three words on the child slot, the step and the stream.
pub fn rng_words(step: u32, seed: u64) -> [u32; 4] {
  [step, seed as u32 ^ (seed >> 32) as u32, 0, 0]
}

pub fn knob_values(params: &SimParams) -> [f32; KNOB_COUNT] {
  [
    params.mutation_sigma,
    params.identity_sigma,
    params.angle_sigma,
    params.structural_rate,
  ]
}

/// Run both mutation kernels over a newborn list.
pub fn mutate<R: Runtime>(
  client: &ComputeClient<R>,
  device: &PopulationDevice,
  inputs: &MutateInputs,
  cfg: MutateCfg,
) {
  if inputs.capacity == 0 {
    return;
  }
  let brain_len = device.limbs.len() / cfg.max_limbs as usize * cfg.param_count as usize;
  mutate_brain::launch::<R>(
    client,
    cube_count(inputs.capacity * cfg.param_count as usize),
    CubeDim::new_1d(CUBE_DIM),
    whole(&device.brain, brain_len),
    whole(&inputs.newborns, inputs.capacity),
    whole(&inputs.parents, inputs.capacity),
    whole(&inputs.count, 1),
    whole(&inputs.rng, 4),
    whole(&inputs.knobs, KNOB_COUNT),
    cfg,
  );
  mutate_structure::launch::<R>(
    client,
    cube_count(inputs.capacity),
    CubeDim::new_1d(CUBE_DIM),
    limb_args(&device.limbs),
    whole(&inputs.newborns, inputs.capacity),
    whole(&inputs.parents, inputs.capacity),
    whole(&inputs.count, 1),
    whole(&inputs.rng, 4),
    whole(&inputs.knobs, KNOB_COUNT),
    cfg,
  );
}

// --- Plain-Rust reference ---

fn draw_gaussian_ref(child: u32, step: u32, seed: u32, stream: u32, index: u32) -> f32 {
  let r = threefry4x32_20_ref([index, 0, 0, 0], [child, step, stream, seed]);
  gaussian_ref(r[0], r[1])
}

fn pick_ref(roll: f32, n: u32) -> u32 {
  ((roll * n as f32) as u32).min(n - 1)
}

/// Plain-Rust twin of [`mutate_brain`] and [`mutate_structure`], over the host
/// copies of the population buffers.
pub fn mutate_ref(
  pop: &mut Population,
  births: &[Birth],
  step: u32,
  seed: u64,
  params: &SimParams,
) {
  let words = rng_words(step, seed);
  let (step_word, seed_word) = (words[R_STEP as usize], words[R_SEED as usize]);
  let knobs = knob_values(params);
  let ml = pop.max_limbs;
  let dim = super::IDENTITY_DIM;
  let param_count = pop.shape.param_count();
  let max_len = pop.max_particles_per_limb as u32;

  for birth in births {
    let child = birth.child;
    let cbase = child as usize * ml;
    let pbase = birth.parent as usize * ml;

    for j in 0..param_count {
      let noise = draw_gaussian_ref(child, step_word, seed_word, STREAM_BRAIN, j as u32);
      pop.brain[child as usize * param_count + j] = pop.brain
        [birth.parent as usize * param_count + j]
        + knobs[K_MUTATION_SIGMA as usize] * noise;
    }

    for l in 0..ml {
      pop.limbs.part_type[cbase + l] = pop.limbs.part_type[pbase + l];
      pop.limbs.length[cbase + l] = pop.limbs.length[pbase + l];
      pop.limbs.parent[cbase + l] = pop.limbs.parent[pbase + l];
      pop.limbs.child_slot[cbase + l] = pop.limbs.child_slot[pbase + l];
      pop.limbs.grow_angle[cbase + l] = pop.limbs.grow_angle[pbase + l];
      pop.limbs.identity[cbase + l] = pop.limbs.identity[pbase + l];
    }

    for l in 0..ml {
      if pop.limbs.part_type[cbase + l].is_present() {
        pop.limbs.grow_angle[cbase + l] += knobs[K_ANGLE_SIGMA as usize]
          * draw_gaussian_ref(child, step_word, seed_word, STREAM_ANGLE, l as u32);
        for k in 0..dim {
          let index = (l * dim + k) as u32;
          pop.limbs.identity[cbase + l][k] += knobs[K_IDENTITY_SIGMA as usize]
            * draw_gaussian_ref(child, step_word, seed_word, STREAM_IDENTITY, index);
        }
      }
    }

    let key = [child, step_word, STREAM_STRUCT, seed_word];
    let b0 = threefry4x32_20_ref([0, 0, 0, 0], key);
    let b1 = threefry4x32_20_ref([1, 0, 0, 0], key);
    if u01_ref(b0[0]) >= knobs[K_STRUCTURAL_RATE as usize] {
      continue;
    }
    let edit = pick_ref(u01_ref(b0[1]), 3);
    let pick_roll = u01_ref(b0[2]);
    let type_roll = u01_ref(b1[0]);
    let length_roll = u01_ref(b1[1]);
    let angle_roll = u01_ref(b1[2]);
    let types = &pop.limbs.part_type;

    if edit == EDIT_ADD {
      let slot = (0..ml).find(|l| !types[cbase + l].is_present());
      let present: Vec<usize> = (0..ml).filter(|l| types[cbase + l].is_present()).collect();
      if let (Some(slot), false) = (slot, present.is_empty()) {
        let chosen = present[pick_ref(pick_roll, present.len() as u32) as usize] as u32;
        let kids = (0..ml)
          .filter(|l| {
            *l as u32 != chosen
              && types[cbase + l].is_present()
              && pop.limbs.parent[cbase + l] as u32 == chosen
          })
          .count() as u32;
        let at = cbase + slot;
        pop.limbs.part_type[at] = PartType::from_byte((1 + pick_ref(type_roll, 4)) as u8);
        pop.limbs.length[at] = (1 + pick_ref(length_roll, max_len)) as u8;
        pop.limbs.parent[at] = chosen as u8;
        pop.limbs.child_slot[at] = kids as u8;
        pop.limbs.grow_angle[at] = (angle_roll * 2.0 - 1.0) * std::f32::consts::PI;
        for k in 0..dim {
          pop.limbs.identity[at][k] =
            draw_gaussian_ref(child, step_word, seed_word, STREAM_NEW_IDENTITY, k as u32);
        }
      }
    } else {
      let candidates: Vec<usize> = (1..ml).filter(|l| types[cbase + l].is_present()).collect();
      if !candidates.is_empty() {
        let victim = candidates[pick_ref(pick_roll, candidates.len() as u32) as usize];
        if edit == EDIT_REMOVE {
          pop.limbs.part_type[cbase + victim] = PartType::Absent;
          for _ in 0..ml {
            for l in 1..ml {
              let p = (pop.limbs.parent[cbase + l] as usize).min(ml - 1);
              if pop.limbs.part_type[cbase + l].is_present()
                && !pop.limbs.part_type[cbase + p].is_present()
              {
                pop.limbs.part_type[cbase + l] = PartType::Absent;
              }
            }
          }
        } else {
          pop.limbs.part_type[cbase + victim] =
            PartType::from_byte((2 + pick_ref(type_roll, 3)) as u8);
        }
      }
    }
  }
}

#[cfg(test)]
pub mod tests {
  use super::*;
  use crate::genome::{BrainShape, Genome};
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  pub fn test_params() -> SimParams {
    SimParams {
      max_organisms: 12,
      max_limbs: 8,
      max_particles_per_limb: 6,
      brain_d_token: 8,
      brain_d_latent: 8,
      brain_n_latents: 2,
      brain_trunk_hidden: 6,
      // High enough that every structural branch runs over a handful of
      // births instead of once in ten.
      structural_rate: 1.0,
      ..SimParams::default()
    }
  }

  /// Four founders in slots 0..4, so the births below have varied parents.
  pub fn seeded_population<R: Runtime>(
    client: &ComputeClient<R>,
    params: &SimParams,
  ) -> Population {
    let shape = BrainShape::from_params(params);
    let mut pop = Population::new(client, params, shape);
    for slot in 0..4u32 {
      let genome = Genome::seed_plant(&shape, pop.max_limbs, slot, 100 + slot as u64);
      pop.write_genome(slot as usize, &genome);
      pop.organisms.alive[slot as usize] = 1;
    }
    // A fourth founder with a fuller body, so a remove has descendants to
    // cascade to and an add has to count existing children.
    let base = pop.limb_index(3, 0);
    for (limb, (part, parent)) in [
      (PartType::Stem, 0u8),
      (PartType::Stem, 1),
      (PartType::Leaf, 2),
      (PartType::Leaf, 2),
      (PartType::Seed, 1),
    ]
    .into_iter()
    .enumerate()
    {
      let i = base + 3 + limb;
      pop.limbs.part_type[i] = part;
      pop.limbs.parent[i] = parent;
      pop.limbs.length[i] = 1 + (limb as u8 % 3);
      pop.limbs.child_slot[i] = limb as u8 % 2;
      pop.limbs.grow_angle[i] = 0.1 * limb as f32;
      pop.limbs.identity[i] = [limb as f32, 0.5, -0.25, 1.0];
    }
    pop.upload(client);
    pop
  }

  /// Every parent appears more than once and every free slot is used, so the
  /// kernel's per-entry indexing is exercised rather than assumed.
  pub fn births() -> Vec<Birth> {
    vec![
      Birth {
        child: 4,
        parent: 0,
      },
      Birth {
        child: 5,
        parent: 3,
      },
      Birth {
        child: 6,
        parent: 3,
      },
      Birth {
        child: 7,
        parent: 1,
      },
      Birth {
        child: 8,
        parent: 2,
      },
      Birth {
        child: 9,
        parent: 3,
      },
    ]
  }

  /// Run both paths over one population and return `(device, reference)`.
  pub fn run_both<R: Runtime>(
    client: &ComputeClient<R>,
    params: &SimParams,
    step: u32,
    seed: u64,
  ) -> (Population, Population) {
    let mut actual = seeded_population(client, params);
    let mut expected = seeded_population(client, params);
    let births = births();

    let cfg = MutateCfg::new(&actual);
    let inputs = MutateInputs::upload(client, &births, step, seed, params);
    mutate(client, &actual.device, &inputs, cfg);
    actual.download(client);

    mutate_ref(&mut expected, &births, step, seed, params);
    (actual, expected)
  }

  fn assert_same(actual: &Population, expected: &Population, what: &str) {
    for (i, (a, e)) in actual.brain.iter().zip(&expected.brain).enumerate() {
      assert_eq!(
        a.to_bits(),
        e.to_bits(),
        "{what}: brain[{i}] device {a} reference {e}"
      );
    }
    assert_eq!(actual.limbs.part_type, expected.limbs.part_type, "{what}");
    assert_eq!(actual.limbs.length, expected.limbs.length, "{what}");
    assert_eq!(actual.limbs.parent, expected.limbs.parent, "{what}");
    assert_eq!(actual.limbs.child_slot, expected.limbs.child_slot, "{what}");
    for (i, (a, e)) in actual
      .limbs
      .grow_angle
      .iter()
      .zip(&expected.limbs.grow_angle)
      .enumerate()
    {
      assert_eq!(a.to_bits(), e.to_bits(), "{what}: grow_angle[{i}]");
    }
    for (i, (a, e)) in actual
      .limbs
      .identity
      .iter()
      .zip(&expected.limbs.identity)
      .enumerate()
    {
      assert_eq!(a, e, "{what}: identity[{i}]");
    }
  }

  #[test]
  fn kernel_matches_reference() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    // Several steps, because the structural edit a birth makes is chosen by
    // the step as well as the slot; one step would exercise one branch.
    for step in 0..8u32 {
      let (actual, expected) = run_both(&client, &params, step, 0xA11FE);
      assert_same(&actual, &expected, &format!("step {step}"));
    }
  }

  #[test]
  fn the_child_starts_as_a_copy_of_its_parent() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = SimParams {
      mutation_sigma: 0.0,
      identity_sigma: 0.0,
      angle_sigma: 0.0,
      structural_rate: 0.0,
      ..test_params()
    };
    let (actual, _) = run_both(&client, &params, 0, 7);
    for birth in births() {
      assert_eq!(
        actual.brain_row(birth.child as usize),
        actual.brain_row(birth.parent as usize),
        "child {} is not a copy of parent {}",
        birth.child,
        birth.parent
      );
      let (c, p) = (
        actual.limb_range(birth.child as usize),
        actual.limb_range(birth.parent as usize),
      );
      assert_eq!(
        actual.limbs.part_type[c.clone()],
        actual.limbs.part_type[p.clone()]
      );
      assert_eq!(
        actual.limbs.parent[c.clone()],
        actual.limbs.parent[p.clone()]
      );
      assert_eq!(actual.limbs.identity[c], actual.limbs.identity[p]);
    }
  }

  #[test]
  fn untouched_slots_are_left_alone() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    let before = seeded_population(&client, &params);
    let (after, _) = run_both(&client, &params, 3, 5);
    for slot in [10usize, 11] {
      assert_eq!(after.brain_row(slot), before.brain_row(slot));
      assert_eq!(
        after.limbs.part_type[after.limb_range(slot)],
        before.limbs.part_type[before.limb_range(slot)]
      );
    }
  }

  #[test]
  fn the_root_is_never_removed_or_retyped() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    for step in 0..64u32 {
      let (_, expected) = run_both(&client, &params, step, 31);
      for birth in births() {
        let root = expected.limb_index(birth.child as usize, 0);
        assert_eq!(
          expected.limbs.part_type[root],
          PartType::Root,
          "step {step}, child {}",
          birth.child
        );
      }
    }
  }

  #[test]
  fn every_structural_edit_happens_over_enough_steps() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    let base = seeded_population(&client, &params);
    let (mut added, mut removed, mut retyped) = (0, 0, 0);

    for step in 0..64u32 {
      let (_, after) = run_both(&client, &params, step, 77);
      for birth in births() {
        let before_types = &base.limbs.part_type[base.limb_range(birth.parent as usize)];
        let after_types = &after.limbs.part_type[after.limb_range(birth.child as usize)];
        let before_present = before_types.iter().filter(|t| t.is_present()).count();
        let after_present = after_types.iter().filter(|t| t.is_present()).count();
        if after_present > before_present {
          added += 1;
        } else if after_present < before_present {
          removed += 1;
        } else if before_types != after_types {
          retyped += 1;
        }
      }
    }
    assert!(
      added > 0 && removed > 0 && retyped > 0,
      "{added} {removed} {retyped}"
    );
  }

  #[test]
  fn a_removed_limb_takes_its_descendants_with_it() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    for step in 0..64u32 {
      let (_, after) = run_both(&client, &params, step, 13);
      for birth in births() {
        let range = after.limb_range(birth.child as usize);
        let types = &after.limbs.part_type[range.clone()];
        let parents = &after.limbs.parent[range];
        for l in 1..params.max_limbs as usize {
          if types[l].is_present() {
            assert!(
              types[parents[l] as usize].is_present(),
              "step {step}: limb {l} of child {} outlived its parent",
              birth.child
            );
          }
        }
      }
    }
  }

  #[test]
  fn an_added_limb_stays_inside_its_caps() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    for step in 0..64u32 {
      let (_, after) = run_both(&client, &params, step, 99);
      for birth in births() {
        let range = after.limb_range(birth.child as usize);
        for i in range {
          if after.limbs.part_type[i].is_present() {
            assert!(after.limbs.length[i] >= 1);
            assert!(after.limbs.length[i] <= params.max_particles_per_limb as u8);
            assert!((after.limbs.parent[i] as i32) < params.max_limbs);
            assert!((after.limbs.part_type[i] as u8) <= 4);
          }
        }
      }
    }
  }

  #[test]
  fn mutation_is_deterministic_in_slot_and_step() {
    let client = CpuRuntime::client(&CpuDevice);
    let params = test_params();
    let (a, _) = run_both(&client, &params, 5, 21);
    let (b, _) = run_both(&client, &params, 5, 21);
    assert_same(&a, &b, "same step and seed");

    let (c, _) = run_both(&client, &params, 6, 21);
    assert_ne!(a.brain, c.brain, "a different step must draw differently");
  }
}
