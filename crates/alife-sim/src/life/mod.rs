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
//! 4. decide, on the host: who dies, who germinates, who sprouts, who seeds;
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
  /// Limbs to grow — germination roots first, then sprouts.
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
      light,
      params_buf,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (cfg, timing) = (*cfg, *timing);
    run_timed(client, timing, timings, "light_grid", &mut || {
      light::launch(client, sph, light, params_buf, cfg, live);
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

    let mut sprouts: Vec<GrowRequest> = Vec::new();
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

      // A plant. Energy below zero is death, and a dead plant neither
      // sprouts nor seeds.
      if energy < 0.0 {
        out.dying.push(o);
        continue;
      }

      if energy >= p.sprout_cost
        && let Some(limb) = pick_sprout(
          &self.pop,
          &self.bodies,
          o,
          &packed[base + pack::LIFE_FIXED..base + width],
        )
      {
        // Charged here rather than after the growth batch: `grow_limbs`
        // drops whole limbs off the end of a batch when the particle
        // capacity is short, and this pays for one that was dropped. The
        // capacity is `max_organisms x max_limbs x max_particles_per_limb`,
        // so it is only short when every slot already holds a full body —
        // at which point an over-charged sprout is the least of it.
        sprouts.push(GrowRequest { organism: o, limb });
        energy -= p.sprout_cost;
      }

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

    // Germination roots before sprouts, so a seedling's root is in the same
    // batch as everything else and the wave order stays one pass.
    for (o, _) in &out.germinating {
      if let Some(limb) = root_limb(&self.pop, *o) {
        out.growth.push(GrowRequest { organism: *o, limb });
      }
    }
    out.growth.extend(sprouts);
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

    // --- Growth: germination roots and sprouts, one batch ---
    if !decisions.growth.is_empty() {
      bodies::grow_limbs(self, &decisions.growth);
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

/// Which limb this organism should sprout, if any.
///
/// The genome is the heritable body plan and the sprout head decides *when*
/// its records grow, not what they are (`docs/organism.md`, decisions,
/// 2026-09-19). So: the first grown limb whose head does not say "none" and
/// which still has an ungrown child record, and that record — lowest child
/// slot first — is what grows. At most one per organism per tick.
#[allow(clippy::needless_range_loop)]
fn pick_sprout(
  pop: &crate::genome::Population,
  bodies: &crate::bodies::BodyState,
  organism: usize,
  choices: &[f32],
) -> Option<usize> {
  let max_limbs = pop.max_limbs;
  for limb in 0..max_limbs {
    if bodies.particle(organism, limb, 0) == NO_PARTICLE {
      continue;
    }
    // Index 0 of the sprout logits is "none".
    if choices[limb] == 0.0 {
      continue;
    }
    let mut best: Option<(u8, usize)> = None;
    for child in 0..max_limbs {
      if child == limb {
        continue;
      }
      let record = pop.limb_index(organism, child);
      if !pop.limbs.part_type[record].is_present()
        || pop.limbs.parent[record] as usize != limb
        || bodies.particle(organism, child, 0) != NO_PARTICLE
      {
        continue;
      }
      let slot = pop.limbs.child_slot[record];
      if best.is_none_or(|(b, _)| slot < b) {
        best = Some((slot, child));
      }
    }
    if let Some((_, child)) = best {
      return Some(child);
    }
  }
  None
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
