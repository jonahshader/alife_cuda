//! Energy, water and the life cycle: what an organism gains, what it spends,
//! and what happens when it runs out.
//!
//! From `docs/organism.md`'s *World coupling and life cycle* and *Water*. One
//! tick every `life_interval` steps ([`crate::sim::Sim::life_tick`]), after
//! the brain, in this order:
//!
//! 1. rebuild the light grid ([`light`]);
//! 2. let the roots drink ([`drink`]): the host uploads who is thirsty, and
//!    liquid particles by a thirsty root leave the fluid;
//! 3. credit and charge every organism ([`energy`]): the drinks go into the
//!    pool, light income spends from it;
//! 4. one packed readback ([`pack`]) — energies, pools, transpired water and
//!    drinks, seed positions and speeds, body tops, and per limb the worst its
//!    base joint was bent over the interval and the brain's growth logit;
//! 5. decide, on the host: who dies, who germinates, how much every limb
//!    grows and what water that locks up, who seeds, how much water each
//!    death and break gives back and how much vapor each plant transpires;
//! 6. free the dead, place the returned water, the vapor and the seeds, run
//!    the growth and mutation kernels.
//!
//! **What crosses the bus, per tick.** One upload of the thirst flags
//! (`max_organisms` words) and one `read` of `max_organisms x (9 + 2 *
//! max_limbs)` floats — 41 KB at the defaults — plus the free-slot scan's two
//! reads when something is claimed, the same reads a `read_free_slots` costs
//! anywhere, and one `download_limbs` (150 KB) when something was born.
//! Nothing else is read back, and nothing is read back at all on a tick where
//! no organism is alive, because `Sim::step` skips the whole tick then.
//!
//! **Why the host decides.** Every decision here is a short serial walk over
//! `max_organisms` slots in slot order, which is deterministic by
//! construction; the work that is actually per-particle — light, drinking,
//! energy, layout, placement, freeing, mutation — is a kernel. Moving the decisions
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
use crate::particles::ParticleKind;
use crate::sim::{Sim, run_timed};

pub mod drink;
pub mod energy;
pub mod light;
pub mod pack;

/// The two terms of the water ledger the rest of the sim cannot reconstruct
/// (`docs/organism.md`, *Water*): water that entered the world from nowhere
/// and water that left it for nowhere, in particles, since this `Sim` was
/// created.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WaterLedger {
  /// Founders' pools and tissue, a loaded population's, and the first root
  /// segment of every seed that germinates — the initial conditions and the
  /// one piece of body that is not grown from the pool.
  pub endowed: f64,
  /// The fraction of a particle each death or break drops when its water is
  /// floored to whole particles, plus any returned particle that found no
  /// free slot.
  pub leaked: f64,
}

/// The whole water ledger at one moment: what came in, what went out, and
/// what is held. [`Self::residual`] is zero up to float rounding.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct WaterBalance {
  pub endowed: f64,
  /// Liquid particles the roots drank.
  pub drunk: u64,
  /// Vapor particles emitted at leaves.
  pub transpired: u64,
  /// Liquid particles given back by deaths and breaks.
  pub returned: u64,
  pub leaked: f64,
  /// Sum of the organisms' pools.
  pub pools: f64,
  /// Sum of `water_per_volume x volume` over every grown limb.
  pub tissue: f64,
  /// Sum of the transpired-but-not-yet-emitted accumulators.
  pub pending: f64,
}

impl WaterBalance {
  /// In minus out minus held: what the ledger cannot account for.
  pub fn residual(&self) -> f64 {
    self.endowed + self.drunk as f64
      - self.transpired as f64
      - self.returned as f64
      - self.leaked
      - self.pools
      - self.tissue
      - self.pending
  }
}

/// What the host decided this tick, before any of it is applied.
#[derive(Debug, Default)]
struct Decisions {
  /// Plants whose energy ran out: they die and become organic matter.
  dying: Vec<usize>,
  /// What each dying plant gives back as liquid, parallel to `dying`: its
  /// pool, its untranspired accumulator and the water in its tissue.
  dying_water: Vec<f32>,
  /// The tissue water of the limbs breaking off, per organism, in slot
  /// order: `(organism, water)`. An organism whose breaking limbs had not
  /// grown is not in it.
  shed_water: Vec<(usize, f32)>,
  /// Vapor each plant transpires this tick: `(organism, particles, where)`.
  emitting: Vec<(usize, u32, Vec2)>,
  /// Liquid particles drunk this tick, every organism together.
  drinks: u64,
  /// Seeds that ran out of time: their particle and slot go back, and
  /// neither counts as a death, because emitting one never counted as a
  /// birth.
  expired: Vec<usize>,
  /// Seeds that landed: `(slot, where its particle came to rest)`.
  germinating: Vec<(usize, Vec2)>,
  /// Limbs whose base joint broke, and everything that was hanging off one:
  /// `(organism, limb)`, in slot then limb order. Their particles go into the
  /// soil as organic matter and their `grown` back to zero.
  breaking: Vec<(usize, usize)>,
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
    self.drink();
    self.charge_energy();

    let packed = self.read_life_pack();
    let decisions = self.decide(&packed);
    self.apply(decisions);
  }

  /// Rebuild the per-soil-cell light grid from the bodies standing in it.
  fn rebuild_light(&mut self) {
    let Self {
      client,
      sph,
      soil_device,
      light,
      bodies,
      pop,
      params_buf,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (cfg, timing) = (*cfg, *timing);
    run_timed(client, timing, timings, "light_grid", &mut || {
      light::launch(
        client,
        sph,
        soil_device,
        light,
        bodies,
        pop,
        params_buf,
        cfg,
      );
    });
  }

  /// Which organisms may drink this tick: a plant whose pool is below its
  /// capacity, `water_capacity_per_volume` times its volume. The flag gates
  /// *starting* drinks, not finishing them: every liquid particle in reach of
  /// the organism's roots rolls against the same flag this tick, so a pool
  /// just under capacity can end the tick over it by the whole tick's drinks
  /// — on average `drink_rate` times the liquid particles in reach, a
  /// particle or so at the defaults. It then does not drink again until it
  /// is back under.
  fn thirsty_flags(&self) -> Vec<u32> {
    let p = &self.params;
    (0..self.pop.max_organisms)
      .map(|o| {
        let organisms = &self.pop.organisms;
        if organisms.alive[o] == 0 || organisms.stage[o] != STAGE_PLANT {
          return 0;
        }
        let volume = bodies::organism_volume(&self.pop, &self.bodies, p.limb_segment_length, o);
        u32::from(organisms.water[o] < p.water_capacity_per_volume * volume)
      })
      .collect()
  }

  /// Let every liquid particle by a thirsty root decide whether it is drunk.
  fn drink(&mut self) {
    let thirsty = self.thirsty_flags();
    if thirsty.iter().all(|flag| *flag == 0) {
      // Nobody can drink, so no particle's decision could differ from "no":
      // the gather over every liquid particle is skipped, and the drink
      // counters stay at the zero the pack kernel left them.
      return;
    }
    let thirsty = self
      .client
      .create_from_slice(bytemuck::cast_slice(&thirsty));
    let live = self.live_particles();
    let (step, seed) = (self.step_count, self.seed);
    let Self {
      client,
      sph,
      grid,
      bodies,
      drinks,
      params_buf,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (cfg, timing) = (*cfg, *timing);
    run_timed(client, timing, timings, "drink_particles", &mut || {
      drink::launch(
        client, sph, grid, bodies, &thirsty, drinks, params_buf, step, seed, cfg, live,
      );
    });
  }

  /// Credit every organism's drinks and light income and charge its upkeep
  /// for the interval.
  fn charge_energy(&mut self) {
    let Self {
      client,
      sph,
      bodies,
      pop,
      light,
      drinks,
      params_buf,
      body_cfg,
      cfg,
      timings,
      timing,
      ..
    } = self;
    let (body_cfg, cfg, timing) = (*body_cfg, *cfg, *timing);
    run_timed(client, timing, timings, "organism_energy", &mut || {
      energy::launch(
        client, sph, bodies, pop, light, drinks, params_buf, body_cfg, cfg,
      );
    });
  }

  fn read_life_pack(&mut self) -> Vec<f32> {
    let num_particles = self.geom.num_particles;
    pack::read(
      &self.client,
      &self.sph,
      &self.bodies,
      &self.pop,
      &self.brain,
      &self.drinks,
      &self.life_pack,
      num_particles,
    )
  }

  /// Fill these organisms' pools to capacity when `fill`, and enter what
  /// they now hold — pool, tissue and accumulator — in the ledger as water
  /// that came from nowhere. For founders and a loaded population, whose
  /// bodies are grown whole as an initial condition rather than out of a
  /// pool. Uploads the organism SoA.
  pub fn endow_water(&mut self, organisms: &[usize], fill: bool) {
    let p = self.params;
    for &o in organisms {
      let volume = bodies::organism_volume(&self.pop, &self.bodies, p.limb_segment_length, o);
      if fill {
        self.pop.organisms.water[o] = p.water_capacity_per_volume * volume;
        self.pop.organisms.transpired[o] = 0.0;
      }
      self.water_ledger.endowed += (self.pop.organisms.water[o]
        + self.pop.organisms.transpired[o]
        + p.water_per_volume * volume) as f64;
    }
    self.pop.upload_organisms(&self.client);
  }

  /// The water ledger as it stands, from the host mirrors — which are the
  /// truth between ticks, because every tick ends by uploading them.
  pub fn water_balance(&self) -> WaterBalance {
    let p = &self.params;
    let organisms = &self.pop.organisms;
    let mut balance = WaterBalance {
      endowed: self.water_ledger.endowed,
      drunk: self.counters.drinks,
      transpired: self.counters.transpired,
      returned: self.counters.returned,
      leaked: self.water_ledger.leaked,
      ..WaterBalance::default()
    };
    for o in 0..self.pop.max_organisms {
      if organisms.alive[o] == 0 {
        continue;
      }
      balance.pools += organisms.water[o] as f64;
      balance.pending += organisms.transpired[o] as f64;
      let volume = bodies::organism_volume(&self.pop, &self.bodies, p.limb_segment_length, o);
      balance.tissue += (p.water_per_volume * volume) as f64;
    }
    balance
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
      out.drinks += packed[base + pack::LIFE_DRINKS] as u64;
      // The energy kernel wrote these on the device; the host mirror is what
      // the metrics sampler reads, so it is refreshed here rather than by a
      // download of its own.
      let mut energy = packed[base + pack::LIFE_ENERGY];
      let mut water = packed[base + pack::LIFE_WATER];
      let mut transpired = packed[base + pack::LIFE_TRANSPIRED];
      self.pop.organisms.energy[o] = energy;
      self.pop.organisms.water[o] = water;
      self.pop.organisms.transpired[o] = transpired;

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
      // nor seeds; every drop it holds goes back to the fluid.
      if energy < 0.0 {
        let volume = bodies::organism_volume(&self.pop, &self.bodies, p.limb_segment_length, o);
        out.dying.push(o);
        out
          .dying_water
          .push(water + transpired + p.water_per_volume * volume);
        continue;
      }

      // What broke over the interval, before anything is spent growing it:
      // a limb that is about to be shed does not grow this tick, and next
      // tick it starts again from nothing. Its tissue water goes back to the
      // fluid where it fell.
      let broken = broken_limbs(&self.pop, &self.bodies, &p, o, &packed[base..]);
      let shed: f32 = broken
        .iter()
        .map(|limb| {
          bodies::limb_volume_of(&self.pop, &self.bodies, p.limb_segment_length, o, *limb)
        })
        .sum();
      if shed > 0.0 {
        out.shed_water.push((o, p.water_per_volume * shed));
      }
      out.breaking.extend(broken.iter().map(|limb| (o, *limb)));

      // Charged before the batch: `grow_limbs` drops a limb off the end of
      // a batch when the particle capacity is short, and this pays for one
      // that was dropped. The capacity is
      // `max_organisms x max_limbs x max_particles_per_limb`, so it is only
      // short when every slot already holds a full body — at which point an
      // over-charged tick is the least of it. The volume grown locks
      // `water_per_volume` of the pool into the body, and the allocation
      // never spends more than the pool can lock; the water of a dropped
      // limb is over-charged the same way, and the ledger's residual shows
      // it, because no tissue holds it and nothing counts it as leaked.
      let charged = allocate_growth(
        &self.pop,
        &self.bodies,
        &p,
        o,
        energy,
        water,
        &broken,
        &packed[base..],
        &mut out.lengthened,
        &mut grew,
      );
      energy -= charged;
      water = (water - charged * p.volume_per_energy * p.water_per_volume).max(0.0);

      let top = Vec2::new(
        packed[base + pack::LIFE_TOP_X],
        packed[base + pack::LIFE_TOP_Y],
      );
      if energy >= p.seed_threshold
        && let Some(child) = free_slots.pop()
      {
        out.seeds.push((o, child as usize, top));
        energy -= p.seed_cost;
      }

      // Transpiration leaves at the top of the plant, a whole particle at a
      // time; the fraction waits for the next tick.
      let whole = transpired.floor();
      if whole >= 1.0 {
        out.emitting.push((o, whole as u32, top));
        transpired -= whole;
      }
      self.pop.organisms.energy[o] = energy;
      self.pop.organisms.water[o] = water;
      self.pop.organisms.transpired[o] = transpired;
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

    // --- Deaths, and limbs that broke off: both leave their particles in the
    // soil where they stood. One batch, so a tick pays for one `launch_free`
    // and one pass over the deposits. Each giver's water comes back where
    // its particles stood, so the ids are kept in one run per giver. ---
    let mut death_ids: Vec<u32> = Vec::new();
    let mut givers: Vec<(std::ops::Range<usize>, f32, Vec2)> = Vec::new();
    for (o, water) in decisions.dying.iter().zip(&decisions.dying_water) {
      let start = death_ids.len();
      death_ids.extend(self.bodies.organism_particles(*o));
      givers.push((start..death_ids.len(), *water, self.bodies.anchors[*o]));
    }
    // Counted by what shed particles: a marked descendant that had not
    // grown yet is not a break, it just may not grow this tick.
    let mut shed = 0u64;
    for (k, (o, limb)) in decisions.breaking.iter().enumerate() {
      let ids = self.bodies.limb_particles_of(*o, *limb);
      if !ids.is_empty() {
        shed += 1;
      }
      // One run per organism: `breaking` is in slot order, so an organism's
      // limbs are consecutive and its run opens at the first of them.
      if k == 0 || decisions.breaking[k - 1].0 != *o {
        let water = decisions
          .shed_water
          .iter()
          .find(|(giver, _)| giver == o)
          .map_or(0.0, |(_, water)| *water);
        givers.push((
          death_ids.len()..death_ids.len(),
          water,
          self.bodies.anchors[*o],
        ));
      }
      death_ids.extend(ids);
      if let Some(run) = givers.last_mut() {
        run.0.end = death_ids.len();
      }
    }
    let mut returns: Vec<Vec2> = Vec::new();
    if !death_ids.is_empty() {
      let where_they_stood = launch_free(&self.client, &self.sph, &death_ids);
      let world = self.geom.cfg();
      let cell_size = self.geom.soil_cell_size;
      // Summed on the host, in slot order, because a float add per soil cell
      // on the device would depend on the order the units arrived in.
      for at in &where_they_stood {
        let cell = light::soil_cell_index_ref(*at, cell_size, &world);
        self.soil.cells.organic_matter[cell] += p.organic_matter_per_particle;
      }
      for (run, water, anchor) in givers {
        returns.extend(self.returned_water(&where_they_stood[run], water, anchor));
      }
    }
    for o in &decisions.dying {
      self.clear_slot(*o);
      self.record_deaths(1);
    }
    // The genome keeps the record, so the limb may grow again from nothing;
    // only the body forgets it.
    for (o, limb) in &decisions.breaking {
      self.bodies.release_limb(*o, *limb);
    }
    self.record_breaks(shed);

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
      // A seed carries no water, but the root segment it germinates with is
      // tissue all the same: the one piece of a grown body not paid for out
      // of a pool, so the ledger counts it as coming from nowhere.
      if let Some(limb) = root_limb(&self.pop, *o) {
        let record = self.pop.limb_index(*o, limb);
        let full = (self.pop.limbs.length[record] as usize)
          .min(self.body_cfg.max_particles_per_limb as usize) as f32;
        let volume = bodies::limb_volume(
          p.limb_segment_length,
          self.pop.limbs.target_thickness[record],
          full,
          full.min(1.0),
        );
        self.water_ledger.endowed += (p.water_per_volume * volume) as f64;
      }
      // A seed becomes an organism here, not when it was emitted: until it
      // lands it is a package in flight, with no body and no budget.
      self.record_births(1);
    }

    // --- One claim for every particle this tick places: the water deaths
    // and breaks give back, then the vapor plants transpire, then the seeds.
    // `claim_particles` hands back as many slots as are free, which can be
    // fewer than asked once the world is near its particle capacity, and the
    // order says who goes without: returned water that finds no slot is
    // leaked, vapor waits in its plant's accumulator, and a seed with no
    // particle is not a birth — the trailing seeds are dropped here, their
    // parents refunded, and nothing below sees them. ---
    let mut seeds = decisions.seeds;
    let emitted: usize = decisions.emitting.iter().map(|(_, n, _)| *n as usize).sum();
    let wanted = returns.len() + emitted + seeds.len();
    let ids = if wanted == 0 {
      Vec::new()
    } else {
      let mut access = self.body_access();
      bodies::claim_particles(&mut access, wanted)
    };
    let mut ids = ids.into_iter();
    let mut placement = Placement::default();
    let mut returned = 0u64;
    for at in &returns {
      match ids.next() {
        Some(id) => {
          placement.push_fluid(id, ParticleKind::Liquid, *at);
          returned += 1;
        }
        None => self.water_ledger.leaked += 1.0,
      }
    }
    let mut transpired = 0u64;
    for (o, count, at) in &decisions.emitting {
      for _ in 0..*count {
        match ids.next() {
          Some(id) => {
            placement.push_fluid(id, ParticleKind::Vapor, *at);
            transpired += 1;
          }
          None => self.pop.organisms.transpired[*o] += 1.0,
        }
      }
    }
    // A fluid particle above the fluid range is in no organism's map, so the
    // live prefix has to be told about it.
    let fluid_range = self.geom.fluid_particles;
    for id in &placement.ids {
      if *id as usize >= fluid_range {
        self.bodies.fluid_high_water = self.bodies.fluid_high_water.max(*id as usize + 1);
      }
    }
    self.counters.drinks += decisions.drinks;
    self.counters.returned += returned;
    self.counters.transpired += transpired;
    let ids: Vec<u32> = ids.collect();
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
      organisms.water[*child] = 0.0;
      organisms.transpired[*child] = 0.0;
    }

    // The organisms SoA is host-mastered except for `energy`, `water` and
    // `transpired`, which were just refreshed from the packed read, so this
    // is the whole of it going up.
    // The brain tensor is *not*: the mutation kernel below writes rows the
    // host master does not have, and a full upload would undo them.
    self.pop.upload_organisms(&self.client);

    // --- The particles themselves: water first, then the seeds ---
    for ((_, child, at), id) in seeds.iter().zip(&ids) {
      placement.ids.push(*id);
      placement.positions.push(at.x);
      placement.positions.push(at.y);
      // A seed is not in any limb, so `limb` and `index_in_limb` are read
      // by nothing; the constraint pass and the sensors both go through the
      // limb map, which has no entry for it.
      placement.push_body(*child as u32, 0, 0, crate::genome::PartType::Seed as u32);
      self.bodies.seed_particles[*child] = *id;
    }
    launch_place(&self.client, &self.sph, &placement, None);
    if !seeds.is_empty() {
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
    } else if !decisions.lengthened.is_empty() || !decisions.breaking.is_empty() {
      // A tick where every limb grew inside the segment it was already in,
      // or where one only broke off: no particle to claim, but the map and
      // `grown` both moved and the constraint pass reads them.
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

  /// Give an organism slot back: nothing alive, no body, no anchor, no
  /// water.
  fn clear_slot(&mut self, organism: usize) {
    self.bodies.release(organism);
    self.bodies.anchors[organism] = Vec2::ZERO;
    let organisms = &mut self.pop.organisms;
    organisms.alive[organism] = 0;
    organisms.energy[organism] = 0.0;
    organisms.water[organism] = 0.0;
    organisms.transpired[organism] = 0.0;
    organisms.stage[organism] = STAGE_SEED;
  }

  /// Where the water one death or break gives back goes: `floor(water)`
  /// liquid particles, one at each position its particles stood at, and any
  /// surplus beside `anchor`. The fraction below one particle is dropped and
  /// counted in the ledger — the ledger's one leak (`docs/organism.md`,
  /// *Water*).
  ///
  /// The surplus is spread along a row through the anchor, a quarter of a
  /// smoothing radius apart: two liquid particles placed on one point would
  /// feel identical forces for ever, since the pressure gradient between
  /// coincident particles is zero, and never come apart.
  fn returned_water(&mut self, stood: &[Vec2], water: f32, anchor: Vec2) -> Vec<Vec2> {
    let whole = water.max(0.0).floor();
    self.water_ledger.leaked += (water.max(0.0) - whole) as f64;
    let n = whole as usize;
    let mut out: Vec<Vec2> = stood.iter().copied().take(n).collect();
    let surplus = n - out.len();
    let spacing = 0.25 * self.params.smoothing_radius;
    let bounds = self.geom.bounds;
    for k in 0..surplus {
      let x = anchor.x + (k as f32 - 0.5 * (surplus as f32 - 1.0)) * spacing;
      out.push(Vec2::new(
        x.rem_euclid(bounds.x),
        anchor.y.clamp(0.0, bounds.y),
      ));
    }
    out
  }
}

/// Grow one organism's limbs by one tick's worth of energy, and say what it
/// cost.
///
/// `docs/organism.md`, *Growth*. A limb is **growing** when it is present,
/// short of its record's length, and — unless it is the root — less far along
/// than its parent; that one rule is the whole of parents-before-children,
/// and it needs no ordering because every limb's progress is read as it stood
/// at the start of the tick. The organism spends its energy, capped at what
/// its growing limbs can absorb between them and at what its water pool can
/// lock into tissue (`water_per_volume` particles per unit of volume, so a
/// dry plant does not grow), and the split is a softmax over the limbs'
/// growth biases plus the brain's growth logits — a bias slows a limb
/// relative to its siblings rather than deleting it, and the logit is the
/// brain saying the same thing about this tick. A share above a limb's own
/// `growth_rate` is capped there and redistributed once over the limbs that
/// were not capped.
///
/// There is no reserve held back for reproduction: a seedling germinates
/// with no more than a seed costs, and a reserve of `seed_cost` would keep
/// it from ever growing the leaf it lives on (`docs/organism.md`,
/// decisions, 2026-10-11).
///
/// A share buys **volume** at `volume_per_energy`, and the limb's new length
/// is solved from the volume it now has (`bodies::grown_from_volume`): a
/// thick limb is dear per segment and every limb's later segments cost more
/// than its first, because a limb thickens as it lengthens.
///
/// What is charged is what was converted: a share the parent clamp, the
/// record's length or a second-order rate cap cut short is not spent, so no
/// energy vanishes into growth that never happened.
///
/// `broken` is the limbs [`broken_limbs`] has decided are coming off this
/// tick; none of them grows, whatever it could have absorbed. `packed` is the
/// organism's slice of the life buffer, which is where the brain's per-limb
/// growth logit rides.
///
/// Writes nothing: the new lengths go to `lengthened` and the limbs that
/// crossed a whole segment — so the map owes them a particle — to `growth`.
#[allow(clippy::too_many_arguments)]
fn allocate_growth(
  pop: &crate::genome::Population,
  bodies: &crate::bodies::BodyState,
  params: &crate::SimParams,
  organism: usize,
  energy: f32,
  water: f32,
  broken: &[usize],
  packed: &[f32],
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
      if full(*limb) == 0.0 || progress(*limb) >= 1.0 || broken.contains(limb) {
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
  // Turgor: the energy whose volume the pool can still lock up.
  let locks = params.volume_per_energy * params.water_per_volume;
  let turgor = if locks > 0.0 {
    water / locks
  } else {
    f32::INFINITY
  };
  let spend = energy.min(turgor).clamp(0.0, capacity);
  if spend <= 0.0 {
    return 0.0;
  }

  // Softmax over the biases, shifted by the largest so a big bias cannot
  // overflow the exponential. The genome's bias is what the limb is for, the
  // brain's logit what this tick is for; they add, and a limb with neither
  // sits at zero like every other.
  let bias = |limb: usize| {
    pop.limbs.growth_bias[pop.limb_index(organism, limb)] + packed[pack::logit_slot(cfg, limb)]
  };
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
    let thickness = pop.limbs.target_thickness[pop.limb_index(organism, *limb)];
    let seg = params.limb_segment_length;
    // The share buys volume, and the length follows from it: volume is
    // quadratic in the length, so the same energy buys a thin limb more
    // length than a thick one and a limb's later segments cost more than its
    // first (`bodies::limb_volume`).
    let before = bodies::limb_volume(seg, thickness, target, grown(*limb));
    let volume = before + share[i] * params.volume_per_energy;
    let mut length = bodies::grown_from_volume(seg, thickness, target, volume);
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
    // What was converted is what is charged, so a share the clamps cut short
    // is not spent: the volume actually gained, back at the same rate.
    charged +=
      (bodies::limb_volume(seg, thickness, target, length) - before) / params.volume_per_energy;
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

/// Which of one organism's limbs break this tick, in limb order.
///
/// `docs/organism.md`, *Growth*: a base joint whose worst bend over the
/// interval — `packed`, the organism's slice of the life buffer, one float
/// per limb — passed `break_angle × clamp(thickness / break_thickness_ref,
/// 0, 1)` snaps, and everything descended from that limb comes off with it.
/// The threshold scales with the limb's *live* thickness, so a limb is
/// fragile while it is young and bears the whole of `break_angle` once it has
/// thickened past the reference.
///
/// Exempt: the root limb, which has no base joint to bend — it is pinned to
/// the anchor and measured against the world frame — and any limb of part
/// type `Root`, whose failure mode is being pulled out of the soil rather
/// than snapping, which is a different mechanism for later. A root hanging
/// off a limb that broke still goes with it: what breaks is the joint above
/// it, and nothing holds it to the body any more.
///
/// A limb with no length has no bend to report and no thickness to bear one,
/// so it never appears here even though its threshold is zero.
fn broken_limbs(
  pop: &crate::genome::Population,
  bodies: &crate::bodies::BodyState,
  params: &crate::SimParams,
  organism: usize,
  packed: &[f32],
) -> Vec<usize> {
  use crate::genome::PartType;

  let cfg = bodies.cfg;
  let mp = cfg.max_particles_per_limb as usize;
  let max_limbs = pop.max_limbs;
  let grown = |limb: usize| bodies.grown[cfg.limb_index(organism, limb)];

  let mut broken = vec![false; max_limbs];
  let mut any = false;
  for limb in 0..max_limbs {
    let record = pop.limb_index(organism, limb);
    let part_type = pop.limbs.part_type[record];
    if !part_type.is_present()
      || part_type == PartType::Root
      || pop.limbs.parent[record] as usize == limb
    {
      continue;
    }
    let target = (pop.limbs.length[record] as usize).min(mp) as f32;
    let thickness = bodies::limb_thickness(pop.limbs.target_thickness[record], target, grown(limb));
    let bears = params.break_angle * (thickness / params.break_thickness_ref).clamp(0.0, 1.0);
    if packed[pack::bend_slot(limb)] > bears {
      broken[limb] = true;
      any = true;
    }
  }
  if !any {
    return Vec::new();
  }

  // Cascade to the descendants. A record's parent can sit at a higher index
  // than the record — a structural add takes the first absent record — so one
  // pass down the indices is not enough.
  for _ in 0..max_limbs {
    let mut spread = false;
    for limb in 0..max_limbs {
      let record = pop.limb_index(organism, limb);
      let parent = pop.limbs.parent[record] as usize;
      if !broken[limb]
        && pop.limbs.part_type[record].is_present()
        && parent != limb
        && parent < max_limbs
        && broken[parent]
      {
        broken[limb] = true;
        spread = true;
      }
    }
    if !spread {
      break;
    }
  }

  // Every marked limb, grown or not. A descendant that has not grown yet
  // has nothing to shed and is not counted as a break (`apply` counts what
  // actually shed particles), but it still has to be in this list: its
  // parent is about to lose its particles, and if it were allocated growth
  // this tick it would end up with a length over an empty map.
  (0..max_limbs).filter(|limb| broken[*limb]).collect()
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
  use crate::genome::{BrainShape, Genome, PartType, Population};
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
      // One unit of energy buys one unit of volume, and every limb is a unit
      // thick over half-metre segments, so a limb's volume is
      // `grown^2 / length` and the arithmetic below is readable.
      volume_per_energy: 1.0,
      limb_segment_length: 0.5,
      thickness_root: 1.0,
      thickness_stem: 1.0,
      thickness_leaf: 1.0,
      growth_rate: 0.1,
      ..SimParams::default()
    }
  }

  /// A limb's volume at a given length, from the record and the params the
  /// test set up.
  fn volume(pop: &Population, params: &SimParams, limb: usize, grown: f32) -> f32 {
    let record = pop.limb_index(0, limb);
    let target = (pop.limbs.length[record] as usize).min(params.max_particles_per_limb as usize);
    crate::bodies::limb_volume(
      params.limb_segment_length,
      pop.limbs.target_thickness[record],
      target as f32,
      grown,
    )
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

  /// Water enough that the pool never caps the growth a test is about.
  const WET: f32 = 1e9;

  /// Everything one call decides, as `(spend, lengths, requests)`, with no
  /// bend and no growth logit in the packed entry and water to spare.
  fn allocate(
    pop: &Population,
    bodies: &BodyState,
    params: &SimParams,
    energy: f32,
  ) -> (f32, Vec<(usize, f32)>, Vec<GrowRequest>) {
    allocate_with(
      pop,
      bodies,
      params,
      energy,
      &vec![0.0; pack::stride(bodies.cfg)],
    )
  }

  /// The same, over a packed entry the caller has put logits into.
  fn allocate_with(
    pop: &Population,
    bodies: &BodyState,
    params: &SimParams,
    energy: f32,
    packed: &[f32],
  ) -> (f32, Vec<(usize, f32)>, Vec<GrowRequest>) {
    let (mut lengths, mut requests) = (Vec::new(), Vec::new());
    let spend = allocate_growth(
      pop,
      bodies,
      params,
      0,
      energy,
      WET,
      &[],
      packed,
      &mut lengths,
      &mut requests,
    );
    (spend, lengths, requests)
  }

  /// The seedling grown to its records' full lengths, and a packed life
  /// entry with no bend in it.
  ///
  /// `break_thickness_ref` is this fixture's own founder thickness, so a
  /// fully grown limb bears the whole of `break_angle` and a part-grown one
  /// bears `sqrt(progress)` of it — the scaling, at numbers a test can read.
  fn grown_seedling(params: &mut SimParams) -> (Population, BodyState, Vec<f32>) {
    params.break_angle = 1.0;
    params.break_thickness_ref = 1.0;
    let (pop, mut bodies) = seedling(params);
    let mp = params.max_particles_per_limb as usize;
    for limb in 0..pop.max_limbs {
      let record = pop.limb_index(0, limb);
      if pop.limbs.part_type[record].is_present() {
        bodies.grown[limb] = (pop.limbs.length[record] as usize).min(mp) as f32;
      }
    }
    let packed = vec![0.0f32; pack::stride(bodies.cfg)];
    (pop, bodies, packed)
  }

  /// A base joint bent past what its limb's thickness bears snaps, and
  /// everything hanging off that limb comes away with it.
  #[test]
  fn a_limb_bent_past_what_it_bears_breaks_with_its_descendants() {
    let mut params = params();
    let (pop, bodies, mut packed) = grown_seedling(&mut params);

    // The stem, at its full thickness, bears the whole of `break_angle`.
    packed[pack::bend_slot(1)] = 0.99;
    assert!(broken_limbs(&pop, &bodies, &params, 0, &packed).is_empty());

    packed[pack::bend_slot(1)] = 1.01;
    assert_eq!(
      broken_limbs(&pop, &bodies, &params, 0, &packed),
      vec![1, 2],
      "the leaf hanging off the stem stayed on"
    );
  }

  /// Roots do not break: not the root limb, which has no base joint to bend,
  /// and not a limb of root type anywhere else in the body.
  #[test]
  fn a_root_does_not_break_under_a_bend_that_snaps_a_stem() {
    let mut params = params();
    let (mut pop, bodies, mut packed) = grown_seedling(&mut params);

    packed[pack::bend_slot(0)] = 10.0;
    assert!(broken_limbs(&pop, &bodies, &params, 0, &packed).is_empty());

    // The same bend on a root-typed limb that *does* have a base joint.
    pop.limbs.part_type[1] = PartType::Root;
    packed[pack::bend_slot(1)] = 10.0;
    assert!(
      broken_limbs(&pop, &bodies, &params, 0, &packed).is_empty(),
      "a root limb snapped"
    );

    // And the stem back as a stem breaks under it, so the bend was real.
    pop.limbs.part_type[1] = PartType::Stem;
    assert_eq!(broken_limbs(&pop, &bodies, &params, 0, &packed), vec![1, 2]);
  }

  /// The threshold scales with the limb's *live* thickness, so a young limb
  /// snaps at a bend a grown one shrugs off — and a descendant that has not
  /// grown yet comes away with its parent, so that it is not grown this tick
  /// over a parent that is about to lose its particles.
  #[test]
  fn a_half_thick_limb_breaks_at_half_the_angle() {
    let mut params = params();
    let (pop, mut bodies, mut packed) = grown_seedling(&mut params);
    // A quarter of the way along its three segments is half its thickness,
    // because thickness goes as `sqrt(progress)`.
    bodies.grown[1] = 0.75;
    bodies.grown[2] = 0.0;

    packed[pack::bend_slot(1)] = 0.49;
    assert!(broken_limbs(&pop, &bodies, &params, 0, &packed).is_empty());

    packed[pack::bend_slot(1)] = 0.51;
    assert_eq!(
      broken_limbs(&pop, &bodies, &params, 0, &packed),
      vec![1, 2],
      "the ungrown leaf did not come away with its stem"
    );
  }

  /// A broken limb is not allocated any of the tick's growth: it is coming
  /// off, and next tick it starts again from nothing.
  #[test]
  fn a_breaking_limb_does_not_grow_on_its_way_out() {
    let params = params();
    let (pop, mut bodies) = seedling(&params);
    bodies.grown[0] = 1.0;

    let (mut lengths, mut requests) = (Vec::new(), Vec::new());
    let spend = allocate_growth(
      &pop,
      &bodies,
      &params,
      0,
      99.0,
      WET,
      &[1],
      &vec![0.0; pack::stride(bodies.cfg)],
      &mut lengths,
      &mut requests,
    );
    assert!(
      lengths.iter().all(|(record, _)| *record != 1),
      "the stem grew while breaking: {lengths:?}"
    );
    // Only the root is left growing, so the spend is one limb's rate.
    assert!((spend - params.growth_rate).abs() < 1e-6, "{spend}");
  }

  /// Growth is funded by whatever energy there is and capped by what the
  /// growing limbs can absorb, so it cannot run away; nothing is held back
  /// for reproduction.
  #[test]
  fn the_spend_sits_between_zero_and_the_limbs_rates() {
    let params = params();
    let (pop, mut bodies) = seedling(&params);
    bodies.grown[0] = 1.0; // the root, half its length

    // No energy, or a debt: nothing grows at all.
    assert_eq!(allocate(&pop, &bodies, &params, -0.5).0, 0.0);
    assert!(allocate(&pop, &bodies, &params, 0.0).1.is_empty());

    // A little: all of it is spent. The root and the stem are growing (the
    // stem is behind the root), the leaf is not (it is level with the stem).
    let (spend, lengths, _) = allocate(&pop, &bodies, &params, 0.05);
    assert!((spend - 0.05).abs() < 1e-6, "{spend}");
    assert_eq!(lengths.len(), 2);

    // A lot: capped at the two growing limbs' rates.
    let (spend, _, _) = allocate(&pop, &bodies, &params, 99.0);
    assert!((spend - 2.0 * params.growth_rate).abs() < 1e-6, "{spend}");
  }

  /// Turgor: the volume a tick grows locks `water_per_volume` of the pool
  /// into tissue, so the spend stops where the pool would run dry, and a dry
  /// plant does not grow at all.
  #[test]
  fn a_dry_plant_does_not_grow_and_a_damp_one_grows_what_its_pool_locks() {
    let params = SimParams {
      water_per_volume: 10.0,
      ..params()
    };
    let (pop, mut bodies) = seedling(&params);
    bodies.grown[0] = 1.0;
    let grow = |water: f32| {
      let (mut lengths, mut requests) = (Vec::new(), Vec::new());
      let spend = allocate_growth(
        &pop,
        &bodies,
        &params,
        0,
        99.0,
        water,
        &[],
        &vec![0.0; pack::stride(bodies.cfg)],
        &mut lengths,
        &mut requests,
      );
      (spend, lengths)
    };

    let (spend, lengths) = grow(0.0);
    assert_eq!(spend, 0.0);
    assert!(lengths.is_empty(), "a dry plant grew: {lengths:?}");

    // 0.5 particles of water lock 0.05 of volume, which one unit of energy
    // per unit of volume makes 0.05 of energy — under the limbs' 0.2.
    let (spend, _) = grow(0.5);
    assert!((spend - 0.05).abs() < 1e-6, "{spend}");
    let locked = spend * params.volume_per_energy * params.water_per_volume;
    assert!(locked <= 0.5 + 1e-6, "the pool went negative: {locked}");
  }

  /// The split is a softmax over the biases, and a limb that cannot take its
  /// share hands the excess back to the others.
  #[test]
  fn a_capped_share_is_redistributed_over_the_limbs_with_room() {
    let params = params();
    let (mut pop, mut bodies) = seedling(&params);
    // Three quarters of the way along, so the stem's own clamp against it is
    // not what this measures.
    bodies.grown[0] = 1.5;
    // The root is capped low and biased high: its share overflows and the
    // stem takes the remainder.
    pop.limbs.growth_rate[0] = 0.01;
    pop.limbs.growth_bias[0] = 3.0;
    pop.limbs.growth_rate[1] = 1.0;

    let (spend, lengths, _) = allocate(&pop, &bodies, &params, 0.5);
    assert!((spend - 0.5).abs() < 1e-6, "{spend}");
    // A share buys volume, so that is where the split shows: the root took
    // its cap and the stem the rest.
    let root = lengths.iter().find(|(r, _)| *r == 0).unwrap().1;
    let stem = lengths.iter().find(|(r, _)| *r == 1).unwrap().1;
    let root_gain = volume(&pop, &params, 0, root) - volume(&pop, &params, 0, 1.5);
    let stem_gain = volume(&pop, &params, 1, stem);
    assert!((root_gain - 0.01).abs() < 1e-6, "the root took {root_gain}");
    assert!((stem_gain - 0.49).abs() < 1e-5, "the stem took {stem_gain}");
  }

  /// The brain's growth logit rides in the packed entry and enters the
  /// softmax beside the genome's bias: a positive logit on one limb moves the
  /// tick's growth towards it, and an equal logit on every limb moves
  /// nothing, because a softmax only sees differences.
  #[test]
  fn a_positive_growth_logit_shifts_the_allocation_towards_its_limb() {
    let params = params();
    let (pop, mut bodies) = seedling(&params);
    // The root three quarters along, so the root and the stem are both
    // growing and neither is capped by its rate at this spend.
    bodies.grown[0] = 1.5;
    let width = pack::stride(bodies.cfg);
    let volume_of = |lengths: &[(usize, f32)], limb: usize, from: f32| {
      let to = lengths.iter().find(|(r, _)| *r == limb).unwrap().1;
      volume(&pop, &params, limb, to) - volume(&pop, &params, limb, from)
    };

    let (flat_spend, flat, _) = allocate(&pop, &bodies, &params, 0.1);
    let (root_flat, stem_flat) = (volume_of(&flat, 0, 1.5), volume_of(&flat, 1, 0.0));
    assert!(
      (root_flat - stem_flat).abs() < 1e-6,
      "two limbs at one bias split unevenly: {root_flat} against {stem_flat}"
    );

    let mut packed = vec![0.0f32; width];
    packed[pack::logit_slot(bodies.cfg, 1)] = 2.0;
    let (spend, biased, _) = allocate_with(&pop, &bodies, &params, 0.1, &packed);
    let (root, stem) = (volume_of(&biased, 0, 1.5), volume_of(&biased, 1, 0.0));
    assert!(
      stem > stem_flat,
      "the logit bought the stem nothing: {stem}"
    );
    assert!(root < root_flat, "the root kept its share: {root}");
    // e^2 : 1 over the two growing limbs, and the spend itself is unchanged —
    // the brain splits the growth, it does not add to it.
    assert!(
      (stem / (root + stem) - (2.0f32.exp() / (1.0 + 2.0f32.exp()))).abs() < 1e-4,
      "the split is not the softmax: {root} against {stem}"
    );
    assert!((spend - 0.1).abs() < 1e-6, "{spend}");

    // The same logit everywhere is no logit at all.
    let mut flat_logits = vec![0.0f32; width];
    for limb in 0..pop.max_limbs {
      flat_logits[pack::logit_slot(bodies.cfg, limb)] = -3.0;
    }
    let (same_spend, same, _) = allocate_with(&pop, &bodies, &params, 0.1, &flat_logits);
    assert_eq!(same, flat);
    assert_eq!(same_spend, flat_spend);
  }

  /// A child ends the tick no further along than its parent, whatever it was
  /// allocated — the whole of parents-before-children.
  #[test]
  fn a_child_never_outgrows_its_parent() {
    let mut params = params();
    params.volume_per_energy = 100.0;
    let (mut pop, mut bodies) = seedling(&params);
    // The root a quarter of the way along, the stem behind it, and the stem
    // biased hard enough to take almost everything.
    bodies.grown[0] = 0.5;
    pop.limbs.growth_bias[1] = 5.0;

    let (charged, lengths, _) = allocate(&pop, &bodies, &params, 1.0);
    let stem = lengths.iter().find(|(r, _)| *r == 1).unwrap().1;
    // The root's progress is 0.25, so the stem stops at a quarter of its own
    // three segments.
    assert!((stem - 0.75).abs() < 1e-5, "the stem reached {stem}");
    // And what the clamp cut off was not paid for: the charge is exactly
    // the volume the two limbs gained, at the conversion rate.
    let root = lengths.iter().find(|(r, _)| *r == 0).unwrap().1;
    let converted = (volume(&pop, &params, 1, stem) - volume(&pop, &params, 1, 0.0)
      + volume(&pop, &params, 0, root)
      - volume(&pop, &params, 0, 0.5))
      / params.volume_per_energy;
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
    let params = params();
    let (mut pop, mut bodies) = seedling(&params);
    // The root alone, so the whole spend lands on one limb.
    pop.limbs.part_type[1] = crate::genome::PartType::Absent;
    pop.limbs.part_type[2] = crate::genome::PartType::Absent;
    bodies.grown[0] = 0.5;
    bodies.limb_particles[0] = 7; // the root holds its first particle

    // A 0.05 share of volume, which is not a whole segment's worth: still
    // inside the first one, so nothing is owed.
    let (_, lengths, requests) = allocate(&pop, &bodies, &params, 0.05);
    assert_eq!(lengths.len(), 1);
    let grown = lengths[0].1;
    assert!((0.5..1.0).contains(&grown), "{grown}");
    let gained = volume(&pop, &params, 0, grown) - volume(&pop, &params, 0, 0.5);
    assert!((gained - 0.05).abs() < 1e-6, "{gained}");
    assert!(requests.is_empty());

    // Past 1: the second particle is owed.
    bodies.grown[0] = 0.95;
    let (_, _, requests) = allocate(&pop, &bodies, &params, 0.1);
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
