//! Energy and the life cycle: what an organism gains, what it spends, and
//! what happens when it runs out.
//!
//! From `docs/organism.md`'s *World coupling and life cycle*. One tick every
//! `life_interval` steps ([`crate::sim::Sim::life_tick`]), after the brain,
//! in this order:
//!
//! 1. rebuild the light grid ([`light`]);
//! 2. credit and charge every organism ([`energy`]);
//! 3. one packed readback ([`pack`]) — energies, seed positions and speeds,
//!    body tops, and one sprout choice per limb;
//! 4. decide, on the host: who dies, who germinates, how much every limb
//!    grows, who seeds;
//! 5. free the dead, place the seeds, run the growth and mutation kernels.
//!
//! **What crosses the bus, per tick.** One `read` of
//! `max_organisms x (6 + max_limbs)` floats — 20 KB at the defaults — plus
//! the free-slot scan's two reads when something is claimed, the same reads a
//! `read_free_slots` costs anywhere, and one `download_limbs` (150 KB) when
//! something was born. Nothing else is read back, and nothing is read back at
//! all on a tick where no organism is alive, because `Sim::step` skips the
//! whole tick then.
//!
//! **Why the host decides.** Every decision here is a short serial walk over
//! `max_organisms` slots in slot order, which is deterministic by
//! construction; the work that is actually per-particle — light, energy,
//! layout, placement, freeing, mutation — is a kernel. Moving the decisions
//! to the device would buy a readback and cost the one place where the whole
//! life cycle can be read top to bottom.

use glam::Vec2;

use cubecl::prelude::Runtime;

use crate::bodies::{self, GrowRequest, NO_PARTICLE};
use crate::genome::mutate::{self, Birth, MutateCfg, MutateInputs};
use crate::genome::population::{STAGE_PLANT, STAGE_SEED};
use crate::genome::slots::free_slots_ref;
use crate::kernels::soil_sample::solid_fraction_at_pos_ref;
use crate::kernels::spawn::{Placement, launch_free, launch_place};
use crate::sim::{Sim, run_timed};

pub mod energy;
pub mod light;
pub mod pack;

/// What the host decided this tick, before any of it is applied.
#[derive(Debug, Default)]
struct Decisions {
  /// Plants whose energy ran out: they die and become organic matter.
  dying: Vec<usize>,
  /// Seeds that ran out of time: their particle and slot go back, and
  /// neither counts as a death, because emitting one never counted as a
  /// birth.
  expired: Vec<usize>,
  /// Seeds that landed: `(slot, where its particle came to rest)`.
  germinating: Vec<(usize, Vec2)>,
  /// What every limb that grew this tick has grown to, in segments:
  /// `(flat (organism, limb) index, length)`.
  lengthened: Vec<(usize, f32)>,
  /// Limbs whose `ceil(grown)` rose, so the map owes them a particle —
  /// germination roots first, then the tick's growth.
  growth: Vec<GrowRequest>,
  /// `(parent, child, where the seed particle starts)`.
  seeds: Vec<(usize, usize, Vec2)>,
}

impl<R: Runtime> Sim<R> {
  /// One life-cycle tick, every `life_interval` steps.
  pub fn life_tick(&mut self) {
    self.rebuild_light();
    self.charge_energy();

    let packed = self.read_life_pack();
    let decisions = self.decide(&packed);
    self.apply(decisions);
  }

  /// Rebuild the per-soil-cell light grid from the bodies standing in it.
  fn rebuild_light(&mut self) {
    let live = self.live_particles();
    let Self {
      client,
      sph,
      soil_device,
      light,
      params_buf,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (cfg, timing) = (*cfg, *timing);
    run_timed(client, timing, timings, "light_grid", &mut || {
      light::launch(client, sph, soil_device, light, params_buf, cfg, live);
    });
  }

  /// Credit every organism's gain and charge its upkeep for the interval.
  fn charge_energy(&mut self) {
    let Self {
      client,
      sph,
      bodies,
      pop,
      light,
      params_buf,
      body_cfg,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (body_cfg, cfg, timing) = (*body_cfg, *cfg, *timing);
    run_timed(client, timing, timings, "organism_energy", &mut || {
      energy::launch(client, sph, bodies, pop, light, params_buf, body_cfg, cfg);
    });
  }

  fn read_life_pack(&mut self) -> Vec<f32> {
    let num_particles = self.geom.num_particles;
    pack::read(
      &self.client,
      &self.sph,
      &self.bodies,
      &self.pop,
      &self.brain.device,
      self.brain.cfg,
      &self.life_pack,
      num_particles,
    )
  }

  /// Read the packed buffer and work out what happens, touching nothing.
  fn decide(&mut self, packed: &[f32]) -> Decisions {
    let p = self.params;
    let body = self.body_cfg;
    let width = pack::stride(body);
    let mp = body.max_particles_per_limb as usize;
    let max_organisms = body.max_organisms as usize;
    let step = self.step_count;
    let soil_cell = self.geom.soil_cell_size;
    let world = self.geom.cfg();
    let mut out = Decisions::default();

    // Claimed as we go, so two seeds emitted in one tick cannot take the same
    // slot. The reference twin of `slots::claim_free_slots`, over the host
    // mirror, gives exactly the ids the device scan would and costs no
    // readback.
    let mut free_slots = free_slots_ref(&self.pop.organisms.alive);
    free_slots.reverse(); // pop() takes the lowest id

    // The tick's growth, held back so germination roots go into the batch
    // first and the wave order stays one pass.
    let mut grew: Vec<GrowRequest> = Vec::new();
    for o in 0..max_organisms {
      if self.pop.organisms.alive[o] == 0 {
        continue;
      }
      let base = o * width;
      // The energy kernel wrote this on the device; the host mirror is what
      // the metrics sampler reads, so it is refreshed here rather than by a
      // download of its own.
      let mut energy = packed[base + pack::LIFE_ENERGY];
      self.pop.organisms.energy[o] = energy;

      if self.pop.organisms.stage[o] == STAGE_SEED {
        if step.saturating_sub(self.pop.organisms.birth_step[o]) >= p.seed_lifetime as u32 {
          out.expired.push(o);
          continue;
        }
        let at = Vec2::new(
          packed[base + pack::LIFE_SEED_X],
          packed[base + pack::LIFE_SEED_Y],
        );
        if at.x == pack::NO_SEED_X {
          continue;
        }
        let speed = packed[base + pack::LIFE_SEED_SPEED];
        let in_soil = solid_fraction_at_pos_ref(at, &self.soil.cells, soil_cell, &world) > 0.0;
        // "On the floor" is within one soil cell of it: `move_particles`
        // reflects at y = 0, so a resting particle never reads exactly zero.
        let on_floor = at.y <= soil_cell;
        if speed < p.germinate_speed && (in_soil || on_floor) {
          out.germinating.push((o, at));
        }
        continue;
      }

      // A plant. Energy below zero is death, and a dead plant neither grows
      // nor seeds.
      if energy < 0.0 {
        out.dying.push(o);
        continue;
      }

      // Charged before the batch: `grow_limbs` drops a limb off the end of
      // a batch when the particle capacity is short, and this pays for one
      // that was dropped. The capacity is
      // `max_organisms x max_limbs x max_particles_per_limb`, so it is only
      // short when every slot already holds a full body — at which point an
      // over-charged tick is the least of it.
      energy -= allocate_growth(
        &self.pop,
        &self.bodies,
        &p,
        o,
        energy,
        &mut out.lengthened,
        &mut grew,
      );

      if energy >= p.seed_threshold
        && let Some(child) = free_slots.pop()
      {
        let child = child as usize;
        let at = Vec2::new(
          packed[base + pack::LIFE_TOP_X],
          packed[base + pack::LIFE_TOP_Y],
        );
        out.seeds.push((o, child, at));
        energy -= p.seed_cost;
      }
      self.pop.organisms.energy[o] = energy;
    }

    // Germination is a root one segment long: the seedling stands on its
    // anchor with one particle in the soil, and the rest of it is growth.
    for (o, _) in &out.germinating {
      if let Some(limb) = root_limb(&self.pop, *o) {
        let record = self.pop.limb_index(*o, limb);
        let full = (self.pop.limbs.length[record] as usize).min(mp) as f32;
        let grown = full.min(1.0);
        if grown > 0.0 {
          out.lengthened.push((body.limb_index(*o, limb), grown));
          out.growth.push(GrowRequest {
            organism: *o,
            limb,
            target: 1,
          });
        }
      }
    }
    out.growth.extend(grew);
    out
  }

  /// Apply what [`Self::decide`] worked out.
  fn apply(&mut self, decisions: Decisions) {
    let p = self.params;
    let step = self.step_count;

    // --- Deaths: the particles go back, and the body goes into the soil ---
    let mut death_ids: Vec<u32> = Vec::new();
    for o in &decisions.dying {
      death_ids.extend(self.bodies.organism_particles(*o));
    }
    if !death_ids.is_empty() {
      let where_they_stood = launch_free(&self.client, &self.sph, &death_ids);
      let world = self.geom.cfg();
      let cell_size = self.geom.soil_cell_size;
      // Summed on the host, in slot order, because a float add per soil cell
      // on the device would depend on the order the units arrived in.
      for at in where_they_stood {
        let cell = light::soil_cell_index_ref(at, cell_size, &world);
        self.soil.cells.organic_matter[cell] += p.organic_matter_per_particle;
      }
    }
    for o in &decisions.dying {
      self.clear_slot(*o);
      self.record_deaths(1);
    }

    // --- Seeds that ran out of time, and the particles of seeds that
    // germinated: both just go back, neither is a death ---
    let mut discard_ids: Vec<u32> = Vec::new();
    for o in &decisions.expired {
      discard_ids.extend(self.bodies.organism_particles(*o));
    }
    for (o, _) in &decisions.germinating {
      let seed = self.bodies.seed_particles[*o];
      if seed != NO_PARTICLE {
        discard_ids.push(seed);
      }
    }
    if !discard_ids.is_empty() {
      launch_free(&self.client, &self.sph, &discard_ids);
    }
    for o in &decisions.expired {
      self.clear_slot(*o);
    }

    // --- Germination: the seed particle is released and the root grows from
    // where it came to rest ---
    for (o, at) in &decisions.germinating {
      self.bodies.seed_particles[*o] = NO_PARTICLE;
      self.bodies.anchors[*o] = *at;
      self.pop.organisms.stage[*o] = STAGE_PLANT;
      self.pop.organisms.energy[*o] = p.seed_energy;
      // A seed becomes an organism here, not when it was emitted: until it
      // lands it is a package in flight, with no body and no budget.
      self.record_births(1);
    }

    // --- Seed particles are claimed before any seed becomes a birth ---
    // `claim_particles` hands back as many slots as are free, which can be
    // fewer than asked once the world is near its particle capacity. A seed
    // with no particle is not a birth: the trailing seeds are dropped here,
    // their parents refunded, and nothing below sees them.
    let mut seeds = decisions.seeds;
    let ids = if seeds.is_empty() {
      Vec::new()
    } else {
      let access = self.body_access();
      bodies::claim_particles(&access, seeds.len())
    };
    for (parent, _, _) in seeds.drain(ids.len()..) {
      self.pop.organisms.energy[parent] += p.seed_cost;
    }

    // --- Newborn slots, so the device sees them before anything grows ---
    let births: Vec<Birth> = seeds
      .iter()
      .map(|(parent, child, _)| Birth {
        child: *child as u32,
        parent: *parent as u32,
      })
      .collect();
    for (parent, child, _) in &seeds {
      let organisms = &mut self.pop.organisms;
      organisms.alive[*child] = 1;
      organisms.stage[*child] = STAGE_SEED;
      organisms.parent_id[*child] = *parent as u32;
      organisms.birth_step[*child] = step;
      organisms.lineage_id[*child] = organisms.lineage_id[*parent];
      organisms.generation[*child] = organisms.generation[*parent] + 1;
      organisms.energy[*child] = 0.0;
    }

    // The organisms SoA is host-mastered except for `energy`, which was just
    // refreshed from the packed read, so this is the whole of it going up.
    // The brain tensor is *not*: the mutation kernel below writes rows the
    // host master does not have, and a full upload would undo them.
    self.pop.upload_organisms(&self.client);

    // --- The seed particles themselves ---
    if !seeds.is_empty() {
      let mut placement = Placement::default();
      for ((_, child, at), id) in seeds.iter().zip(&ids) {
        placement.ids.push(*id);
        placement.positions.push(at.x);
        placement.positions.push(at.y);
        placement.organism.push(*child as u32);
        // A seed is not in any limb, so `limb` and `index_in_limb` are read
        // by nothing; the constraint pass and the sensors both go through the
        // limb map, which has no entry for it.
        placement.limb.push(0);
        placement.index_in_limb.push(0);
        placement
          .part_type
          .push(crate::genome::PartType::Seed as u32);
        self.bodies.seed_particles[*child] = *id;
      }
      launch_place(&self.client, &self.sph, &placement, None);
      self.bodies.upload(&self.client);
    }

    // --- Growth: every limb's new length, and the particles the ones that
    // crossed a whole segment now owe. `grow_limbs` reconciles the two when
    // the particle capacity is short. ---
    for (record, grown) in &decisions.lengthened {
      self.bodies.grown[*record] = *grown;
    }
    if !decisions.growth.is_empty() {
      bodies::grow_limbs(self, &decisions.growth);
    } else if !decisions.lengthened.is_empty() {
      // A tick where every limb grew inside the segment it was already in:
      // no particle to claim, but the constraint pass reads `grown`.
      self.bodies.upload(&self.client);
    }

    // --- Mutation, and the rest of what a birth is ---
    if !births.is_empty() {
      let cfg = MutateCfg::new(&self.pop);
      let inputs = MutateInputs::upload(&self.client, &births, step, self.seed, &p);
      mutate::mutate(&self.client, &self.pop.device, &inputs, cfg);
      mutate::launch_reset_latents(
        &self.client,
        &self.pop.device,
        &inputs,
        &self.pop.shape,
        self.pop.max_organisms,
      );
      self.pop.narrow_shadow(&self.client);
      // The child's limb records now exist only on the device. The host needs
      // them back to decide what it can grow and to measure species distance.
      self.pop.download_limbs(&self.client);
    }

    // Deaths can empty the top of the body range, and the per-particle
    // kernels should stop being launched over it.
    self.bodies.recompute_high_water();
  }

  /// Give an organism slot back: nothing alive, no body, no anchor.
  fn clear_slot(&mut self, organism: usize) {
    self.bodies.release(organism);
    self.bodies.anchors[organism] = Vec2::ZERO;
    let organisms = &mut self.pop.organisms;
    organisms.alive[organism] = 0;
    organisms.energy[organism] = 0.0;
    organisms.stage[organism] = STAGE_SEED;
  }
}

/// Grow one organism's limbs by one tick's worth of energy, and say what it
/// cost.
///
/// `docs/organism.md`, *Growth*. A limb is **growing** when it is present,
/// short of its record's length, and — unless it is the root — less far along
/// than its parent; that one rule is the whole of parents-before-children,
/// and it needs no ordering because every limb's progress is read as it stood
/// at the start of the tick. The organism spends `energy - seed_cost`, capped
/// at what its growing limbs can absorb between them, so growth never starves
/// reproduction, and the split is a softmax over the limbs' growth biases —
/// a bias slows a limb relative to its siblings rather than deleting it. A
/// share above a limb's own `growth_rate` is capped there and redistributed
/// once over the limbs that were not capped.
///
/// What is charged is what was converted: a share the parent clamp, the
/// record's length or a second-order rate cap cut short is not spent, so no
/// energy vanishes into growth that never happened.
///
/// Writes nothing: the new lengths go to `lengthened` and the limbs that
/// crossed a whole segment — so the map owes them a particle — to `growth`.
fn allocate_growth(
  pop: &crate::genome::Population,
  bodies: &crate::bodies::BodyState,
  params: &crate::SimParams,
  organism: usize,
  energy: f32,
  lengthened: &mut Vec<(usize, f32)>,
  growth: &mut Vec<GrowRequest>,
) -> f32 {
  let cfg = bodies.cfg;
  let mp = cfg.max_particles_per_limb as usize;
  let max_limbs = pop.max_limbs;

  // Progress as it stands, per limb: 0 where there is nothing to grow.
  let full = |limb: usize| -> f32 {
    let record = pop.limb_index(organism, limb);
    if pop.limbs.part_type[record].is_present() {
      (pop.limbs.length[record] as usize).min(mp) as f32
    } else {
      0.0
    }
  };
  let grown = |limb: usize| bodies.grown[cfg.limb_index(organism, limb)];
  let progress = |limb: usize| {
    let target = full(limb);
    if target > 0.0 {
      grown(limb) / target
    } else {
      1.0
    }
  };

  let growing: Vec<usize> = (0..max_limbs)
    .filter(|limb| {
      if full(*limb) == 0.0 || progress(*limb) >= 1.0 {
        return false;
      }
      let parent = pop.limbs.parent[pop.limb_index(organism, *limb)] as usize;
      parent == *limb || (parent < max_limbs && progress(*limb) < progress(parent))
    })
    .collect();
  if growing.is_empty() {
    return 0.0;
  }

  let rate = |limb: usize| pop.limbs.growth_rate[pop.limb_index(organism, limb)].max(0.0);
  let capacity: f32 = growing.iter().map(|limb| rate(*limb)).sum();
  let spend = (energy - params.seed_cost).clamp(0.0, capacity);
  if spend <= 0.0 {
    return 0.0;
  }

  // Softmax over the biases, shifted by the largest so a big bias cannot
  // overflow the exponential.
  let bias = |limb: usize| pop.limbs.growth_bias[pop.limb_index(organism, limb)];
  let top = growing
    .iter()
    .map(|limb| bias(*limb))
    .fold(f32::NEG_INFINITY, f32::max);
  let weight: Vec<f32> = growing
    .iter()
    .map(|limb| (bias(*limb) - top).exp())
    .collect();
  let total: f32 = weight.iter().sum();
  if total <= 0.0 || !total.is_finite() {
    // Only a bias that is not a number gets here, and a body with one grows
    // nothing rather than dividing by it.
    return 0.0;
  }
  let mut share: Vec<f32> = weight.iter().map(|w| spend * w / total).collect();

  // Cap at each limb's own rate and redistribute the excess once, by the same
  // softmax over the limbs that had room.
  let mut excess = 0.0f32;
  let mut room = 0.0f32;
  for (i, limb) in growing.iter().enumerate() {
    if share[i] > rate(*limb) {
      excess += share[i] - rate(*limb);
      share[i] = rate(*limb);
    } else {
      room += weight[i];
    }
  }
  if excess > 0.0 && room > 0.0 {
    for (i, limb) in growing.iter().enumerate() {
      if share[i] < rate(*limb) {
        share[i] = (share[i] + excess * weight[i] / room).min(rate(*limb));
      }
    }
  }

  let mut charged = 0.0f32;
  for (i, limb) in growing.iter().enumerate() {
    let target = full(*limb);
    let mut length = grown(*limb) + share[i] * params.length_per_energy;
    let parent = pop.limbs.parent[pop.limb_index(organism, *limb)] as usize;
    if parent != *limb {
      // A child ends the tick no further along than its parent stood at the
      // start of it.
      length = length.min(progress(parent) * target);
    }
    length = length.clamp(0.0, target);
    if length <= grown(*limb) {
      continue;
    }
    charged += (length - grown(*limb)) / params.length_per_energy;
    lengthened.push((cfg.limb_index(organism, *limb), length));
    let held = bodies.limb_len(organism, *limb);
    let wants = length.ceil() as usize;
    if wants > held {
      growth.push(GrowRequest {
        organism,
        limb: *limb,
        target: wants,
      });
    }
  }
  charged.min(spend)
}

/// The limb that is its own parent. Mutation never removes or retypes the
/// root, so a present genome has exactly one; a corrupt one has none and
/// grows nothing.
fn root_limb(pop: &crate::genome::Population, organism: usize) -> Option<usize> {
  (0..pop.max_limbs).find(|limb| {
    let record = pop.limb_index(organism, *limb);
    pop.limbs.part_type[record].is_present() && pop.limbs.parent[record] as usize == *limb
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::SimParams;
  use crate::bodies::BodyState;
  use crate::genome::{BrainShape, Genome, Population};
  use crate::world::WorldGeometry;
  use cubecl::prelude::Runtime;
  use cubecl_cpu::{CpuDevice, CpuRuntime};

  fn params() -> SimParams {
    SimParams {
      world_width: 4.0,
      world_height: 3.0,
      smoothing_radius: 0.5,
      soil_cell_size: 0.25,
      particles_per_cell: 1,
      max_organisms: 2,
      max_limbs: 4,
      max_particles_per_limb: 4,
      seed_cost: 1.0,
      length_per_energy: 1.0,
      growth_rate: 0.1,
      ..SimParams::default()
    }
  }

  /// The founder plant — a 2-particle root, a 3-particle stem, a 1-particle
  /// leaf — in slot 0, with nothing grown yet.
  fn seedling(params: &SimParams) -> (Population, BodyState) {
    let client = CpuRuntime::client(&CpuDevice);
    let shape = BrainShape::from_params(params);
    let mut pop = Population::new(&client, params, shape);
    let genome = Genome::seed_plant(&shape, pop.max_limbs, 0, 1, params);
    pop.write_genome(0, &genome);
    pop.organisms.alive[0] = 1;
    let geom = WorldGeometry::from_params(params);
    let bodies = BodyState::new(&client, params, &geom);
    (pop, bodies)
  }

  /// Everything one call decides, as `(spend, lengths, requests)`.
  fn allocate(
    pop: &Population,
    bodies: &BodyState,
    params: &SimParams,
    energy: f32,
  ) -> (f32, Vec<(usize, f32)>, Vec<GrowRequest>) {
    let (mut lengths, mut requests) = (Vec::new(), Vec::new());
    let spend = allocate_growth(pop, bodies, params, 0, energy, &mut lengths, &mut requests);
    (spend, lengths, requests)
  }

  /// Growth is funded by what is left over `seed_cost` and capped by what the
  /// growing limbs can absorb, so it can neither starve reproduction nor run
  /// away.
  #[test]
  fn the_spend_sits_between_the_seed_reserve_and_the_limbs_rates() {
    let params = params();
    let (pop, mut bodies) = seedling(&params);
    bodies.grown[0] = 1.0; // the root, half its length

    // Below the reserve: nothing grows at all.
    assert_eq!(allocate(&pop, &bodies, &params, 0.9).0, 0.0);
    assert!(allocate(&pop, &bodies, &params, 1.0).1.is_empty());

    // Just above it: the whole surplus is spent. The root and the stem are
    // growing (the stem is behind the root), the leaf is not (it is level
    // with the stem).
    let (spend, lengths, _) = allocate(&pop, &bodies, &params, 1.05);
    assert!((spend - 0.05).abs() < 1e-6, "{spend}");
    assert_eq!(lengths.len(), 2);

    // Far above it: capped at the two growing limbs' rates.
    let (spend, _, _) = allocate(&pop, &bodies, &params, 99.0);
    assert!((spend - 2.0 * params.growth_rate).abs() < 1e-6, "{spend}");
  }

  /// The split is a softmax over the biases, and a limb that cannot take its
  /// share hands the excess back to the others.
  #[test]
  fn a_capped_share_is_redistributed_over_the_limbs_with_room() {
    let mut params = params();
    params.length_per_energy = 1.0;
    let (mut pop, mut bodies) = seedling(&params);
    bodies.grown[0] = 1.0;
    // The root is capped low and biased high: its share overflows and the
    // stem takes the remainder.
    pop.limbs.growth_rate[0] = 0.01;
    pop.limbs.growth_bias[0] = 3.0;
    pop.limbs.growth_rate[1] = 1.0;

    let (spend, lengths, _) = allocate(&pop, &bodies, &params, 1.5);
    assert!((spend - 0.5).abs() < 1e-6, "{spend}");
    let root = lengths.iter().find(|(r, _)| *r == 0).unwrap().1 - 1.0;
    let stem = lengths.iter().find(|(r, _)| *r == 1).unwrap().1;
    assert!((root - 0.01).abs() < 1e-6, "the root took {root}");
    assert!((stem - 0.49).abs() < 1e-5, "the stem took {stem}");
  }

  /// A child ends the tick no further along than its parent, whatever it was
  /// allocated — the whole of parents-before-children.
  #[test]
  fn a_child_never_outgrows_its_parent() {
    let mut params = params();
    params.length_per_energy = 100.0;
    let (mut pop, mut bodies) = seedling(&params);
    // The root a quarter of the way along, the stem behind it, and the stem
    // biased hard enough to take almost everything.
    bodies.grown[0] = 0.5;
    pop.limbs.growth_bias[1] = 5.0;

    let (charged, lengths, _) = allocate(&pop, &bodies, &params, 2.0);
    let stem = lengths.iter().find(|(r, _)| *r == 1).unwrap().1;
    // The root's progress is 0.25, so the stem stops at a quarter of its own
    // three segments.
    assert!((stem - 0.75).abs() < 1e-5, "the stem reached {stem}");
    // And what the clamp cut off was not paid for: the charge is exactly
    // the length the two limbs gained, at the conversion rate.
    let root = lengths.iter().find(|(r, _)| *r == 0).unwrap().1;
    let converted = ((stem - 0.0) + (root - 0.5)) / params.length_per_energy;
    assert!(
      (charged - converted).abs() < 1e-6,
      "charged {charged} for {converted}"
    );
    assert!(
      charged < 2.0 * params.growth_rate,
      "the cut share was charged"
    );
  }

  /// A particle is owed only when the limb crosses a whole segment; inside
  /// one, growth is the rest length lengthening and nothing else.
  #[test]
  fn a_particle_is_claimed_when_the_length_crosses_a_segment() {
    let mut params = params();
    params.length_per_energy = 1.0;
    let (mut pop, mut bodies) = seedling(&params);
    // The root alone, so the whole spend lands on one limb.
    pop.limbs.part_type[1] = crate::genome::PartType::Absent;
    pop.limbs.part_type[2] = crate::genome::PartType::Absent;
    bodies.grown[0] = 0.5;
    bodies.limb_particles[0] = 7; // the root holds its first particle

    // 0.5 -> 0.55: still inside the first segment.
    let (_, lengths, requests) = allocate(&pop, &bodies, &params, 1.05);
    assert_eq!(lengths.len(), 1);
    assert!((lengths[0].1 - 0.55).abs() < 1e-6, "{:?}", lengths[0]);
    assert!(requests.is_empty());

    // Past 1: the second particle is owed.
    bodies.grown[0] = 0.95;
    let (_, _, requests) = allocate(&pop, &bodies, &params, 1.1);
    assert_eq!(
      requests,
      vec![GrowRequest {
        organism: 0,
        limb: 0,
        target: 2,
      }]
    );
  }
}
