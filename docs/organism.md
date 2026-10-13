# Organism design

The settled design for organisms: one representation for plants and
creatures, a fixed-shape genome, a brain over body-part tokens, bodies made
of particles in the fluid's particle system, and selection by ecology rather
than a fitness function. **Status: milestone 1 is built and has been run —
the genome, the particle bodies, the brain's forward pass, energy with a
life cycle that turns the population over on its own, and the
soil-specialization experiment with its three controls, which can be run and
scored from `scripts/`.** The open work is
in `TODO.md`; this file is the spec builders build from. When
implementation diverges from it, update this file in the same commit.

## Why this shape

The project's goal is the highest degree of interesting emergent behavior
reachable: speciation, coevolution, and organisms that reshape their world
(burrowing, seed planting, soil preference). Three requirements follow.

- **Niche construction.** An organism's actions must leave persistent marks
  on the world that other organisms then live with. Terrain, water, and
  other organisms are all state that organisms can act on through one
  physical system.
- **Continuity between kinds.** Plants and animals are regions of one
  genotype space, not two systems. Speciation and coevolution show up where
  no category boundary was drawn by hand.
- **Evaluations per second.** Evolution finds things at the rate of
  organism-lifetimes per second. Every design choice below is weighed
  against throughput before fidelity. Real biology is inspiration and
  vocabulary only; nothing here has to be biologically faithful.

## Genome

Two sections.

**Discrete: the body program.** An ordered list of limb records. Each record
has a part type, a length in particles, a parent limb index, a child slot
index, and a small per-limb identity vector (a few floats that belong to the
record, so identical limbs can be told apart by the brain). Structural
mutations add, remove, or retype a record. Species are defined by distance
in this section.

**Continuous: the brain.** A single tensor of fixed shape for every organism
in the population regardless of body: the latent trunk weights, the input
and output cross-attention weights, and the per-part-type embedding table.
The shape never changes, so every organism is one row of one population
tensor. Species means, lineage deltas, and genome distance are plain tensor
operations across the whole population, and the batched forward pass is one
kernel launch.

A structural mutation needs no new weights: the brain already knows how to
run any limb of a known type, so a new limb is functional at birth.

## Body

A limb is a chain of particles in the fluid's particle system, held by
position-based constraints: a distance constraint to the previous particle
and an angle constraint at the limb's base joint. The constraint rest length
is capped at about the SPH smoothing radius, so water cannot pass between
two body particles; a limb longer than that is several particles. Bodies
are points only. This is the standard SPH treatment of solids (boundary
particles contribute density and push through pressure), and it keeps every
neighbor kernel handling one geometry type.

Consequences:

- Bodies feel water and soil through the coupling the fluid already has, and
  water feels bodies. A plant bends in flow; a body displaces water.
- Cost per organism is limbs times length, at roughly the cost of a fluid
  particle each. Compact bodies are cheap, long slender ones expensive, and
  evolution feels that pressure. The fluid's neighbor-gather kernels set
  this cost, which is why profiling SPH is on the critical path again.
- A limb's orientation is the direction from its parent's last particle to
  its own; no separate rotation state.
- Body particles carry quadratic drag (`body_drag`, applied implicitly) and
  a speed cap (`body_speed_max`) in the constraint pass's velocity update:
  the only damping a body has in air, and the solver's stability guarantee
  (decisions, 2026-10-12). Seeds in flight are not in any limb and get
  neither.
- Segment (capsule) collision is the upgrade if evolved bodies keep wanting
  long thin limbs or chunky collisions block fine manipulation. It changes
  the discretization, not the genome, since a limb is already a type and a
  length either way.

Part types at the first milestone: **root** (pinned in soil, drinks the
liquid particles around it), **stem** (structure), **leaf** (gains energy
from light, paid for in water), **seed** (a reproduction particle). Later types are additions,
not new systems: an **actuated limb** whose base joint takes its target angle
from the brain outright, rather than the bounded nudge every limb's joint
already takes (*Growth*), a **digger** that decrements soil density in its
cell, a **mouth**, and **sensors**. Burrowing, erosion, and deposition all write
soil density through one channel.

## Brain

A Perceiver-IO-style network over limb tokens, per organism.

- **Tokens.** One per limb: the part-type embedding, a spatial embedding
  (position of the limb relative to the root, in the body frame), a rotation
  embedding (angle relative to the parent limb), depth and child-slot index,
  the per-limb identity vector, and the limb's live sensor readings (light,
  water, soil, contact, energy as applicable).
- **Latents.** A fixed set of latent vectors that persist across ticks with a
  gated update, each scaled to unit RMS on its way back into the persistent
  state. They are the organism's working memory. One pass per tick. The
  normalization carries no parameters and is not optional: without it the
  tick's three residual adds compound across ticks and the latents overflow
  (decisions, 2026-09-19).
- **Input cross-attention** from latents to tokens, a fixed-size **trunk**
  over the latents, and **output cross-attention** from tokens to latents.
- **Output heads per limb:** actuator commands (a nudge on the base joint's
  target angle, later more), and a **growth head**, one logit that biases how
  much of the tick's growth this limb gets. Development and control share the
  trunk.
- **Held in reserve**, added only when evolved bodies plateau in a way that
  looks like the missing piece: an attention bias by tree distance (one
  scalar per hop count, Graphormer-style) to restore a locality prior, and a
  per-limb memory head that writes a small vector back onto the token.

There is no message passing along body edges. The tree structure enters only
through the embeddings and, if ever needed, the reserved distance bias.

**Size budget.** The trunk is per-organism weights read once per tick with
no reuse, so its cost is weight bytes times population and it is
bandwidth-bound (measured in `alife_cuda_2`'s `BENCH.md`: a 20k-parameter
MLP over 16k agents at fp16 is about 0.4 ms per tick). Cross-attention
projections are reused across a body's tokens and are cheap. Trunk width is
therefore a first-class throughput knob, and "beefy" means tens of thousands
of parameters, not millions. `alife_cuda_2`'s custom fp16 batched-GEMV
kernel (one block per agent, coalesced half reads, fp32 accumulate) is the
design the trunk kernel follows, rewritten in CubeCL; its block-DAG engine
assumes one MLP architecture per population and is not ported.

## World coupling and life cycle

The whole cycle runs on one cadence: every `life_interval` steps (default
10), after the brain. `crates/alife-sim/src/life/` is it.

- **Light.** A per-soil-cell occlusion grid, rebuilt each tick. One kernel,
  one unit per organism over its stem and leaf limbs, counts their
  particles per cell into a `u32` grid in fixed point: a particle adds
  `round(256 × segment_fraction(grown, k))`, so it shades as much as its
  segment has grown and a grown one adds exactly 256. A second, one unit
  per soil column, walks from the top down and multiplies the light by
  `light_attenuation` (default 0.7) once per whole 256 it has passed and,
  for a cell's remainder `r`, by `1 − (r/256)(1 − light_attenuation)` —
  linear, because a `powf` would not be bit-identical across backends —
  starting at `light_top` (1.0), and by the soil of each cell it
  passes: a cell transmits `1 − solid_fraction × light_soil_opacity`,
  floored at zero, so at the default opacity (2.0) one full cell of any
  soil type is opaque and nothing under it earns light. A leaf in the top
  soil cell still reads the light arriving at that cell. The sensors read
  the same grid, so between rebuilds a limb's light reading is up to
  `life_interval` steps stale.
- **Energy and water.** One unit per organism, serial over its particles.
  Leaves earn `light_gain` times the light in their cell times how far
  their segment has grown (`segment_fraction`, as for shade), but only as far
  as the organism's water pool pays for it, at `water_per_energy` particles
  per unit of energy; roots earn nothing — they drink, and the pool is
  what they drink into (*Water*, below). Upkeep is `upkeep_per_volume`
  times the volume of the organism's limbs, so what a body costs is how
  much of it there is rather than how many particles it is discretized
  into (*Growth*). Energy below zero is death: every particle goes back to
  the free pool and leaves `organic_matter_per_particle` in the soil cell
  it was standing in, and every drop the plant held goes back to the fluid
  as liquid.
- **Growth.** Continuous, every limb at once, the organism's growth spend
  split across its growing limbs by a softmax over per-limb biases and
  spent on volume; a child never outgrows its parent in progress. The whole
  model is the next section, *Growth*.
- **Reproduction.** Energy above `seed_threshold` is spent on a seed
  particle carrying a mutated copy of the genome. The seed is a particle
  with no limb, so no constraint holds it: it falls, water carries it, and
  dispersal is free. A seed that comes to rest (speed below
  `germinate_speed`) in soil or on the floor germinates — its particle is
  released, its root grows from where it landed, it starts on
  `seed_energy` and an empty pool, and only then does it count as a birth.
  One that drifts for `seed_lifetime` steps without landing dies
  unremarked. A seed costs its parent `seed_cost` (3.0) and germinates
  with `seed_energy` (3.0), so it carries exactly what it cost; reproduction
  is triggered by `seed_threshold` (5.0) alone, with nothing held back from
  growth for it.
- **Selection.** None explicit. Energy, shading, water, space, and other
  organisms are the fitness function; the simulation is the evaluator.

### Water

Water is a ledger. Every particle of it is in exactly one place — the fluid
(liquid or vapor), an organism's pool, an organism's tissue, or an
organism's transpiration accumulator — and it moves between them only here:

- **Drinking** (`life::drink`, on the life tick, before the energy kernel).
  One unit per liquid particle finds the nearest `Root` body particle within
  `smoothing_radius`, the lowest particle id breaking a tie, so exactly one
  organism is ever in the running. If that organism is *thirsty* — its pool
  below capacity, a per-organism flag the host uploads every tick — a
  Threefry draw keyed by the particle id, the step and stream word 3 (the
  evaporation and vapor kernels use 1 and 2) drinks the particle with
  probability `drink_rate` (0.05 per tick) times the root particle's
  `segment_fraction` (*Growth*): its slot becomes `Free`, exactly
  as a freed body slot is, and the organism's `u32` drink count goes up by
  one with an integer atomic. The free-slot scan covers the fluid range, so
  a drunk slot is the first thing the next claim hands out.
- **The pool**, `water` in the organism SoA, in particles, holds up to
  `water_capacity_per_volume` (3000) times the organism's volume — about 5.2
  particles for a grown founder. A drink is whole and the capacity a float,
  so the flag gates *starting* drinks: a root drinks only while the pool is
  below capacity, and because every liquid particle in reach rolls against
  the same flag within one tick, the pool can end a tick above capacity by
  that tick's drinks — on average `drink_rate` times the liquid particles
  in reach, a particle or so at the defaults. It is mastered like
  `energy`: the energy kernel credits the tick's drinks and spends light
  income from it, the packed readback carries it to the host, the decision
  walk charges growth to it and uploads it.
- **Light income spends it.** A tick's light income is capped at
  `water / water_per_energy` (1.0 particle per unit of energy, the
  water-use-efficiency knob), and what is earned moves `earned ×
  water_per_energy` from the pool into the organism's `transpired`
  accumulator. A plant with an empty pool earns nothing however bright its
  leaves are.
- **Transpiration** leaves at the leaf. Every tick, in slot order, each
  whole particle in a plant's accumulator becomes a `Vapor` particle at its
  top — the highest stem or leaf particle, the point a seed leaves from —
  rising at `(0, 0.5)` as freshly evaporated water does; with no free slot
  it waits in the accumulator. From there the fluid's own
  evaporation/condensation cycle carries it, and it comes down as liquid
  somewhere else.
- **Tissue.** Growing `ΔV` of volume locks `water_per_volume × ΔV` (1000 per
  unit of volume, so a whole founder body is 1.7 particles) of the pool into
  the body, and the tick's growth spend is capped so the pool cannot go
  negative: a dry plant does not grow (*Growth*).
- **Return.** A death gives back the pool, the accumulator and the tissue
  (`water_per_volume` times the body's volume); a break gives back the
  tissue of the limbs that came off. `floor` of that comes back as liquid at
  rest, one particle at each position the shed particles stood at and any
  surplus in a row through the anchor, a quarter of a smoothing radius
  apart; the fraction below one particle is dropped. That is the ledger's
  only leak, less than a particle per death or break.
- **Sources.** A founder starts with its grown body's tissue and a full
  pool; a `--load-pop` population with its tissue and the pool its file
  carries (full, for a file written before the pool existed); a germinating
  seed's first root segment comes with its tissue water. A seed itself
  carries none. Those, and drinks, are the only ways in.

A headless run prints the whole ledger at exit — endowed plus drunk against
transpired, returned, leaked, pools, tissue and pending — and it closes to
float rounding.

## Growth

Settled 2026-09-19 (decisions). Limbs grow continuously in length and
thickness, the way the C++ trees did, instead of appearing whole; the
genome still says what the body is, and the brain only biases how fast
each part gets there. Four chunks, in build order; `TODO.md` carries the
one still open.

**1. Continuous length and allocation, and 2. thickness and volume.** Both
built; the two together are the model below.

- *State per `(organism, limb)`:* `grown: f32`, the limb's current length
  in segments, `0 ≤ grown ≤ length` (the record's `length` is now the
  target in particles). Progress `f = grown / length`. Particle `k`
  (0-based) of a limb exists while `grown > k`; its rest distance to its
  predecessor — the previous particle, or the parent's last particle for
  `k = 0` — is `limb_segment_length × clamp(grown − k, 0, 1)`, so the
  last segment lengthens between particle claims and growth is visible
  frame to frame. The root's `k = 0` has no predecessor and stays pinned to
  the anchor. Germination sets the root's `grown` to 1; a founder, which is
  an initial condition rather than something the life cycle produced, starts
  at its records' full lengths, and so does a body re-grown by `--load-pop`.
  `grown` lives in `bodies::BodyState` beside the particle map, because it
  is state rather than genome, and the two are one fact: a limb holds
  `ceil(grown)` particles, so the map's leading run is what every kernel
  reads as the limb's length rather than the record's.
- *Genome per limb record*, all three mutable: `growth_bias: f32` (softmax
  logit, founders 0), `growth_rate: f32` (energy the limb can absorb
  per life tick, founders and structural adds at the `growth_rate` param)
  and `target_thickness: f32` (the thickness at full length; founders and
  structural adds at the `thickness_root` / `thickness_stem` /
  `thickness_leaf` param of the record's part type, a seed record sharing
  the leaf's). The bias and rate get a small Gaussian of their own,
  `growth_sigma`, and the thickness its own `thickness_sigma` in metres
  (a thickness is of the order of the founder values, so the growth genes'
  sigma would be a step the size of the value), as `grow_angle` gets
  `angle_sigma`; the rate is floored at zero and the
  thickness strictly above it, because volume divides by the thickness
  squared. The population file format bumps its version per gene added and
  reads an older one with the founder values.
- *Thickness is not state:* `thickness = target_thickness × sqrt(f)`, the
  C++ rule, so a limb thickens as it lengthens, and *volume* is `V = grown ×
  limb_segment_length × thickness²` — which with that rule is
  `limb_segment_length × target_thickness² × grown² / length`, quadratic in
  `grown` and therefore invertible in closed form
  (`bodies::grown_from_volume`). Thickness does not enter the SPH coupling.
- *Growing:* a present limb with `f < 1`, and — unless it is the root —
  `f < f_parent`. A child's increment is clamped so it ends the tick at
  `f ≤ f_parent`. This is the whole parent-before-child rule; there is no
  ordering.
- *Spend per tick:* `spend = clamp(min(energy, water / (volume_per_energy ×
  water_per_volume)), 0, Σ growth_rate over growing limbs)` — all the
  energy there is, capped by what the growing limbs can absorb and by what
  the water pool can lock into the new tissue (*Water*); nothing is held
  back for reproduction. The volume grown takes `water_per_volume` per unit
  of it out of the pool. Shares are `spend × softmax(growth_bias + logit)`
  over the growing limbs, `logit` being the brain's growth head for that
  limb, as it stood at the tick's own step; a share above the
  limb's `growth_rate` is capped and the excess redistributed once, by the
  same softmax, over the uncapped limbs. A share buys *volume* at
  `volume_per_energy`, and the limb's new `grown` is solved from `V + ΔV`,
  so a thick limb is dear per segment and a limb's later segments cost more
  than its first. What was converted is charged — a share cut short by the
  parent clamp or the record's length is not spent; `sprout_cost` is gone.
- *Upkeep* is `upkeep_per_volume × Σ V` over the organism's present limbs,
  in place of a per-particle charge, so a tall thin plant is cheaper than a
  tall thick one and a half-grown limb costs a quarter of what it will. A
  seed in flight is in no limb, has no volume, and costs nothing beyond
  `seed_cost` at emission. Income stays per particle: what a leaf catches
  is sampled at its particles, and what a root drinks is found from its.
  Each particle counts for `segment_fraction(grown, k)` of one — its light,
  its shade and its drinking alike — so a limb's newest particle earns,
  shades and drinks in proportion to its growing segment rather than
  wholly from the tick it is claimed.
- *Particles* are claimed when `ceil(grown)` rises, placed at the
  predecessor plus the rest offset along the limb's direction, by the
  existing growth kernel over a list of `(organism, limb, first index,
  count)` — a *suffix* of a limb's chain rather than a whole limb, which is
  one particle in an ordinary tick and a whole limb for a founder.

The renderer draws each segment at its limb's thickness, and the body discs
at the same half-width, so a joint stays round.

**3. Breaking.** Built. A base joint whose angular error against its rest
angle exceeds `break_angle × clamp(thickness / break_thickness_ref, 0, 1)`
breaks: the limb and everything descended from it leave their particles as
organic matter where they stand — the death path's per-particle deposit and
release — and their `grown` returns to 0. The records stay in the genome, so
a broken limb regrows from nothing through the ordinary allocation; it is
allocated none of the growth of the tick it comes off in.

`thickness` is the limb's *live* thickness, `target_thickness × sqrt(f)`, so
a limb is fragile while it is young and bears the whole of `break_angle`
once it has thickened past `break_thickness_ref`. The defaults are 1.2 rad
(69°) and 0.02 m: every founder limb is thicker than the reference at full
length — root 0.04, stem 0.05, leaf 0.03 — so a grown one only snaps under a
large bend, and a stem reaches full strength at 16% of its length. Roots do
not break: not the root limb, which has no base joint to bend (it is pinned
to its anchor and measured against the world frame), and not a limb of root
type anywhere else, whose failure is being pulled out rather than snapping.
A root hanging off a limb that broke still comes away with it — what broke
is the joint above it.

The error is what the constraint pass measures at each base joint before
every sweep pulls it back, against the parent's *axis*: its last whole
segment, or, for a parent with none of its own yet, its own parent's axis
turned by its grow angle, walking up to the root if need be
(`kernels::constraints::limb_axis`). The segment a parent is still growing
into is millimetres long when it has just been claimed, and its direction is
the fluid's jostling rather than the limb's; and a joint left unmeasured
while its parent's first segment grows drifts, and reads the whole drift the
moment measuring resumes (decisions, 2026-10-12). The maximum per limb over the interval rides in
the packed life buffer, one float per limb, written by the one unit that
owns the organism and so needing no atomic. The pack kernel that reads it
resets it, so each tick sees its own interval, and the decision is the
host's, in slot order, like every other life decision. A limb shed this way
counts in the `breaks` column of the metrics time series.

**4. The brain's part.** Built. `head_growth` is one logit per limb, added
to `growth_bias` before the softmax — the sprout head's "when" role, now
continuous; it rides in the packed life buffer's second per-limb slice. The
actuator head's first output `a` is wired too: the base-joint target is
`grow_angle + actuator_gain × tanh(a)`, the brain nudging
the genome's rest angle within a band `tanh` keeps it inside, so no head
value can bend a joint far enough to break the limb. The gain's default is
0 — the actuator is off until it has a rate limit and a cost (decisions,
2026-10-11); `--actuator-gain 0.3` is what it was. Both are the wire
phototropism needs: a token carries its limb's light, and a brain that
raises the logit or bends the joint toward the brighter side grows toward
the sun with no new mechanism. The second actuator output stays reserved.

## Evolution operators

- **Continuous mutation:** Gaussian perturbation of the brain tensor.
- **Structural mutation:** add, remove, or retype a limb record; perturb
  length or identity vector.
- **Adaptive mutation distribution (per-species knob, default off).** Two
  equivalent accelerations are available: shift each offspring's mutation
  mean along the lineage's recent selected deltas (the CMA-ES evolution
  path, with Adam-style per-coordinate normalization and a short horizon
  because the landscape is non-stationary), or estimate a species-level
  gradient from the population's empirical covariance weighted by who
  reproduced (a natural evolution strategy with binary fitness). Both must
  bias the mutation distribution, never replace individuals, so organisms
  stay embedded in the world. Faster hill-climbing within a niche reduces
  standing diversity, which is the raw material of speciation, so this is
  measured against lineage diversity with the knob on and off before it is
  ever the default.

## Metrics

The metrics that score this project are evolutionary, not fluid-physical:

- lineage count and phylogenetic depth over time;
- distributions of a few genome and body traits (root fraction, height, leaf
  count) over time, and whether they bimodalize;
- population, births, deaths, and energy flux.

Written by headless runs as a sampled time series to a file, with a summary
derived from the same samples at exit. Fluid metrics (density error,
divergence, settling) gate only fluid optimization and are secondary.

`--metrics <path> --metrics-every K` (default 100) is that time series:
`crates/alife-sim/src/metrics/`, one CSV row per sample, its header naming
every column. The traits are reported per soil column, because the
experiment below asks whether they split by soil — the terrain publishes its
extents as `SoilGrid::columns()`, and an organism is binned by the column its
anchor sits in: `alive`, `root_frac`, `height`, `leaf_count`, `energy_mean`
and `species` per column. A germinated plant anchored in a gap counts in
`alive` and in no column, and so does a seed in flight, which holds a slot
but has no anchor until it lands.
Species are greedy clusters: each organism joins the first cluster whose
representative it is within `species_threshold` (default 0.25) of by
`species_distance`, else starts one. Phylogenetic depth is the `generation`
field of the organism SoA: a founder is 0 and a child is its parent's plus
one. The water ledger (*Water*) rides beside the fluid counts it moves
between: `water_mean` is the mean pool over germinated plants, and
`drinks`, `transpired` and `returned` are cumulative particle counts, read
against `fluid_liquid` and `fluid_vapor`. Sampling only reads, so a run with
`--metrics` steps the same world as one without.

### The first experiment: soil specialization

Run plants in the capillary field terrain (`--terrain-mode 2`), where sand,
silt, and clay columns stand side by side with gaps between them, and watch
whether trait distributions split by column. Skepticism is warranted that
soil alone drives a split, and the controls must separate causes:

- **Isolation control.** The gaps between columns are spatial barriers, so
  lineages diverge by drift alone (allopatric divergence without adaptation).
  Run the same geometry with identical soil in every column; any split that
  appears there is not soil specialization.
- **Position control.** Columns differ in light, distance from the water
  pool, and edge effects. Permute which soil type sits in which column
  across seeds; specialization must follow the soil, not the position.
- **Transplant test.** Take a lineage from one column, seed it into another,
  and compare its energy trajectory against the resident lineage. A true
  specialist does worse away from home.

**The terrain.** `--terrain-mode 2` (`TerrainMode::CapillaryField`,
`crates/alife-sim/src/soil.rs`) is mode 1 with a ceiling: the same pool below
20% of the world height, the same six columns at the same x extents, the same
gaps, but the soil stops at `--column-top` of the world height and everything
above is air. Mode 1 fills the columns to the top of the world and is
therefore uninhabitable (decisions, 2026-09-19); it is also frozen rather
than fixed — `perf.md`'s measurements were taken on it. How far above the pool the columns end decides whether anything lives
on them at all — see the `--column-top` decision entry.

**The controls**, both mode 2 only; the validator rejects them on any other
mode, and `SoilGrid::columns()` is what the metrics bin by, so a control that
moves soil around moves the metrics with it.

- `--soil-permutation <seed>` (default 0 = identity, the absence of the
  control) deals the six compositions out to the six positions by a
  Fisher–Yates shuffle over a SplitMix64 stream seeded with `<seed>` and
  nothing else. A column's *label* travels with its composition, so the
  per-column metrics still call the sand column `sand` wherever it stands,
  and the CSV's column order is still ascending x.
- `--uniform-soil sand|silt|clay` (default `none`) fills every column with
  one pure composition, leaving only the gaps to separate them. With no
  composition left to name a column after, the labels become `col0..col5` by
  position.
- The transplant is `--save-pop`, `--load-pop` and `--transplant <from>:<to>`
  (by column label, repeatable, every move applied at once from the
  membership before any of them, so `sand:clay` with `clay:sand` swaps the
  two). A transplanted organism keeps its relative x within the column and
  re-anchors on the destination's soil surface
  (`bodies::transplant_anchors`). The population file's format is documented
  above the writer in `crates/alife-sim/src/popdump.rs`: it carries the
  genomes, the lineage fields, the energies, the latent state and the
  anchors, and a load re-grows the bodies from those and drops seeds in
  flight.

**Running it.** `scripts/soil_experiment.sh [N]` runs the eleven headless
runs — main, isolation and position controls over three seeds, plus the
transplant and its no-transplant twin — into `runs/soil/`, and
`scripts/soil_score.py` scores them.

## Milestones

1. **Plants.** Anchored organisms with root, stem, leaf, and seed types; the
   growth head biases growth; energy, death, reproduction; evolutionary
   metrics; the soil-specialization experiment with its controls. This
   exercises every piece above except the actuated limb type.
2. **Mobile creatures.** Add the actuated limb, a mouth, and contact and
   proximity sensors. Locomotion must be evolved, not given. If populations
   never learn to move, the escape hatch is a thrust part type that pushes
   directly when in contact with ground or water; do not build it before a
   run shows the stall.
3. **Soil as actionable state.** The digger type, so terrain is written by
   creatures as well as by erosion.

## Implementation layout (milestone 1)

What the chunks in `TODO.md` share, pinned so they can be built in parallel.
Sizes are sim params with these defaults; a flag changes them.

**Particle system.** `ParticleKind` gains `Body` and `Free`. A `Free` slot
is unallocated: every kernel skips it, its position is parked outside the
grid. Capacity is fixed at start: fluid particles plus `max_organisms ×
max_limbs × max_particles_per_limb` (defaults 1024 × 16 × 8); the comptime
particle count is that capacity, but the per-particle kernels are launched
only over the live prefix — the fluid plus the high-water mark of claimed
body slots — because slots are claimed ascending and everything above the
mark is `Free`. Per particle: `organism` (u32, `u32::MAX` for none), `limb`
(u8), `index_in_limb` (u8) and `part_type` (u8, the owning limb's type, so
the renderer and the energy pass need no genome lookup). Body particles are
SPH boundary particles: they carry mass and contribute density to their
neighbors, and they integrate with the same forces as liquid (gravity,
pressure, viscosity, soil friction). After integration a **constraint
pass** projects them (`kernels::constraints`), one unit per organism, serial
over that organism's particles, `constraint_iterations` Gauss-Seidel sweeps
(default 4). Per sweep, in order: the root's first particle pinned to its
anchor; distance at `limb_segment_length` (default 0.15, validated ≤ the
grid cell size) times how far the limb has grown into that segment
(*Growth*), which is 1 for every segment but a growing limb's last;
the angle at each limb's base joint against the parent
segment's direction (target `grow_angle + actuator_gain × tanh(a)`, the limb
record's rest angle nudged by the brain's actuator output, stiffness
`joint_stiffness`), whose error before
the projection is kept per limb as a running maximum for the life tick to
break on (*Growth*); bend, keeping a
limb's consecutive segments in line (`bend_stiffness`), or a limb is a rope;
and the world bounds, x wrapping and y clamping. Velocity is recomputed from
the projected displacement, position-based-dynamics style. Slot allocation
is deterministic: births claim `Free` particle slots and free organism slots
in order via a prefix scan over the free flags, never via atomics. The
particle scan covers every slot, the fluid range included, because a
liquid particle a root drank is a `Free` slot like any other (*Water*); a
fluid particle the life tick places above the fluid range is in no
organism's map, so `BodyState::fluid_high_water` keeps the live prefix
covering it.

A limb's chain, for both the distance and the angle constraints, is its
parent's last particle followed by its own: the base joint sits on the
parent, which is what gives a one-particle limb an orientation. The root
limb has no such particle, so its chain is its own and its first segment is
`p0 → p1`, in the world frame. Differences between two body particles are
taken as the minimum image across the world's x seam, so a body standing on
the seam is not torn apart by its own distance constraints.

Which particle a projection moves: a distance constraint moves both ends
symmetrically, half the error each; an angle constraint moves only the
child-side particle, because the parent side is either already projected
this sweep or the pinned root.

`bodies::BodyState` holds what the pass walks — the `(organism, limb,
index) → particle slot` map, each organism's anchor, and the per-limb
geometry the brain will read — beside the population tensors.

**Genome buffers**, population SoA with capacity `max_organisms`:

- Discrete, `[max_organisms × max_limbs]`: `part_type` (u8, 0 = absent;
  1 root, 2 stem, 3 leaf, 4 seed, 5–7 reserved), `length` (u8, particles),
  `parent` (u8 limb index; the root limb is its own parent), `child_slot`
  (u8), `grow_angle` (f32, the rest angle at the base joint, measured
  against the parent segment's direction — the root, being its own parent,
  measures against the world frame instead, 0 at +x and counter-clockwise),
  `identity` (`[f32; 4]`), `growth_bias`, `growth_rate` and
  `target_thickness` (f32, the softmax logit, the per-tick energy cap and
  the thickness at full length of *Growth*). A field appended
  here appends an array to the population file, which is what lets an older
  one still load (`popdump.rs`).
- Continuous, `[max_organisms × BrainShape::param_count()]` f32 on the
  host as the master copy, uploaded per organism on birth; the fp16 device
  shadow is the brain chunk's concern.
- Lineage: `parent_id`, `birth_step`, `lineage_id` (root ancestor),
  `generation` (hops from the founder; mutation never touches it), plus
  the runtime `alive` flag, `energy`, `stage`, `water` and `transpired`
  (the pool and the accumulator of *Water*), and the latent state
  `[max_organisms × n_latents × d_latent]`, which is state, not genome.
  `alive` means the slot is occupied — it is what the slot allocator scans
  — and `stage` says by what: 0 a seed in flight, 1 a germinated plant. A
  seed holds a slot and a genome but has no body, no energy budget and no
  place in the population counts until it lands.
- Mutation is two kernels over newborn organisms, Threefry-keyed by
  `(child slot, step, stream)` and countered by the draw's index within its
  stream, so what a child gets depends on nothing but its slot and the
  step. One is a draw per brain parameter over `newborns × param_count`;
  the other is one unit per newborn, serial over that organism's limb
  records — one unit for both would put 18,659 serial iterations on a
  single thread. Both copy the parent's section before perturbing it.
  Gaussian perturbation of the brain tensor (`mutation_sigma`), and with
  probability `structural_rate` one structural edit chosen uniformly: add a
  limb (random present parent, first absent record, random type 1–4, random
  length, child slot = the parent's existing child count, a uniform grow
  angle and a fresh sigma-1 identity), remove a limb (a random present
  non-root limb, marked absent and cascaded to its descendants), or retype
  (a random present non-root limb, to a type 2–4). The root is never
  removed or retyped. Identity and grow angle get their own small Gaussian,
  on present records only. Species distance is over the discrete section:
  the number of differing part types (absent counts as a type) plus the
  mean identity L2 over the limbs present in both, the sum divided by
  `max_limbs`, computed on the host for metrics.

**Brain shape** (`BrainShape`, owned by the genome chunk, consumed by the
brain chunk). Defaults: `d_token` 32, `d_latent` 32, `n_latents` 8,
`trunk_hidden` 64, `n_types` 8, identity 4, sensors 5 (light, water,
soil solid fraction, contact, energy). Token features are the
concatenation of spatial (2), rotation as cos/sin (2), depth (1), child
slot (1), identity (4), sensors (5) = 15, projected to `d_token` and added
to the part-type embedding. Parameter slices, in this order, each a named
range so both chunks index the same tensor:

| slice | shape |
|---|---|
| `type_embed` | `n_types × d_token` |
| `tok_proj`, `tok_bias` | `15 × d_token`, `d_token` |
| `in_q`, `in_k`, `in_v`, `in_o` | `d_latent × d_latent`, `d_token × d_latent`, `d_token × d_latent`, `d_latent × d_latent` |
| `self_q`, `self_k`, `self_v`, `self_o` | four `d_latent × d_latent` |
| `mlp_w1`, `mlp_b1`, `mlp_w2`, `mlp_b2` | `d_latent × trunk_hidden`, `trunk_hidden`, `trunk_hidden × d_latent`, `d_latent` |
| `gate_w`, `gate_b` | `d_latent × d_latent`, `d_latent` |
| `out_q`, `out_k`, `out_v`, `out_o` | `d_token × d_latent`, `d_latent × d_latent`, `d_latent × d_latent`, `d_latent × d_token` |
| `head_growth`, `growth_b` | `d_token × 1`, `1` (the growth logit) |
| `head_actuator`, `actuator_b` | `d_token × 2`, `2` (joint nudge, reserved) |
| `latent_init` | `n_latents × d_latent` |

18,659 parameters at the defaults. Initialization is Gaussian with sigma
`1/sqrt(fan_in)` per slice, where a weight slice is stored `[input dim ×
output dim]` so its fan-in is its row count; biases are zero and
`latent_init` is sigma 1. One tick of the brain: tokens from limb
geometry and sensors; input cross-attention latents→tokens; latent
self-attention; MLP; gated update of the persistent latents, scaled to unit
RMS; output cross-attention tokens→latents; heads per limb. Single-head
attention, fp32 accumulate, one pass per tick. The trunk's activation is
SiLU. Token features are built inline in the token kernel rather than
materialized, and the heads are `[max_organisms × max_limbs × 3]` — the
growth logit, then the two actuator outputs — zeroed for an absent limb or a
free slot.

The pass is **22 launches** (`brain::forward::LAUNCHES`), because kernels
stay barrier-free and every cross-unit reduction is therefore a launch
boundary. Measured cost and the fp16-versus-fp32 comparison are in
`perf.md`; at the defaults it is about 6% of the step and entirely
launch-bound, so fp32 storage is the default and `--brain-fp16` is a flag.

**Step order** once organisms exist: grid build (all non-`Free`) → density
→ evap probability → accel → evaporate → move liquid → move vapor →
constraint pass → limb geometry → sense (token features per limb) → brain
forward → the life tick, every `life_interval` steps. The constraint pass
publishes limb geometry (root-relative position, segment angle, depth) for
the token features; the life tick runs on the heads the forward pass just
wrote, so applying a head and deciding the life cycle are one step. The
constraint pass is the exception, because it runs before the brain: the
actuator output it nudges a base joint by is the one the *previous* step's
forward pass wrote, and zeros before the first — a step of latency, like the
sensors' own.

**The life tick's shape.** Light, drinking and energy are kernels; growth,
placement, freeing and mutation are kernels; the *decisions* between them
are a serial walk over the organism slots on the host, in slot order, and
therefore deterministic. It costs an upload of the thirst flags and one
`read` per tick of a packed `[max_organisms × (9 + 2 × max_limbs)]` float
buffer — energies, water pools, transpired water and the tick's drinks
(reset by the pack kernel that reads them), seed positions and speeds,
body tops, then two per-limb slices: the worst each
base joint was bent over the interval, accumulated by the constraint pass and
reset by the pack kernel that reads it, and the brain's growth logit, copied
out of the heads buffer and not reset, because the brain rewrites it every
step — plus the free-slot scan's reads when
something is claimed and one limb-record download when something was born.
The organism SoA is host-mastered except `energy`, `water` and
`transpired`, which the energy kernel writes and the walk adjusts; the
returned water, the transpired vapor and the seeds are claimed in one scan
and placed in one launch. The brain tensor and the limb records are device-mastered for a
newborn, because that is where the mutation kernels wrote them, so a birth
never uploads the whole population. The growth decision is part of that
walk: per organism it reads the limbs' progress, splits the tick's spend
(*Growth*), writes the new `grown` values and asks `bodies::grow_limbs` for
the particles the limbs that crossed a whole segment now owe. `grown` and
the map go up together, so a tick where a limb only lengthened still
uploads them.

## Decisions & dead ends

- 2026-09-18 — **Single organism representation** for plants and creatures.
  Special-casing draws category lines by hand, which is what earlier sims
  did; intermediate forms must be reachable for speciation to be visible.
- 2026-09-18 — **Fixed-shape continuous genome** over variable-shape. A
  variable brain shape forces bucketing by architecture for the batched
  forward pass and makes cross-species genome operations ill-defined.
- 2026-09-18 — **Attention over limb tokens, no message passing.** A GNN
  pass would add only a locality prior and per-node state, both of which
  are cheaper as a reserved attention bias and a memory head.
- 2026-09-18 — **Bodies are particles, not rigid segments.** One geometry
  type in every neighbor kernel and free coupling to water and soil;
  capsule collision is the upgrade path, not the start.
- 2026-09-18 — **Plants first.** The locomotion stall is the one
  project-killing failure mode, and anchored organisms do not have it.
- 2026-09-18 — **Substrate: Rust + CubeCL, pinned to an exact
  pre-release.** CPU and Radeon support are musts, Intel a bonus, Mac
  possible if the project is ever gamified. Kokkos and AdaptiveCpp reach
  AMD only through ROCm (short consumer-card list, no AMD on Windows) and
  never reach Metal; wgpu with hand-written WGSL reaches every GPU but has
  no native CPU path (software Vulkan only) and no generics. CubeCL is one
  Rust kernel language over CPU (LLVM JIT), CUDA, HIP, Vulkan, Metal, and
  WebGPU, with tensor cores on the native backends if the brain ever
  becomes GEMM-shaped. It is alpha with breaking changes between minor
  versions; the mitigation is an exact version pin bumped deliberately.
  The C++/CUDA tree was the reference until parity, then deleted (tag
  `cpp-reference`).
- 2026-09-18 — **Spike outcome (the deleted `crates/spike`, commit
  2a56cb0).** The
  CubeCL wgpu runtime adopts an existing `wgpu::Device` via `WgpuSetup` +
  `init_device`, and a kernel's buffer binds in a render pass with no
  copy; egui-wgpu 0.36 and cubecl-wgpu 0.11.0-pre.3 share one `wgpu 30`.
  Two rules for the renderer: a CubeCL handle is a slice of a pooled
  buffer (bind with its offset, never as the entire buffer), and CubeCL
  buffers lack `VERTEX` usage, so draw from storage buffers indexed by
  vertex id. **CPU kernels are not debuggable**: the LLVM JIT emits no
  symbols or line tables, so `gdb` never sees the kernel. The debugging
  path is a plain-Rust reference implementation per kernel plus
  `CUBECL_DEBUG_PLIRON` IR dumps. **The pin must cover transitive deps**:
  the published crate does not resolve on its own (`pliron` 0.17 pulls an
  incompatible `pliron-derive` 0.18), so `Cargo.lock` is committed and
  authoritative.
- 2026-09-18 — **Fluid solver swap (FLIP/PIC) is on hold.** Fluid fidelity
  is not on the path to emergence. SPH kernel optimization stays relevant
  because bodies ride on the same per-particle cost.
- 2026-09-19 — **The port buys reproducibility with two divergences from
  the C++, and they are not up for trade.** The neighbour grid is built by
  a counting sort rather than one atomic slot grab per particle, and
  `calculate_accel` writes a second velocity buffer that is swapped in
  rather than writing `sph.vel` while its own viscosity term reads it. Both
  are why a same-seed Rust run is bit-identical and a same-seed C++ run is
  not. They cost ~9% of the step (`perf.md`); an optimisation that gives
  either back is not an optimisation. The C++'s `sph.vel` read-write hazard
  is a genuine race, not just a determinism nuisance.
- 2026-09-19 — **Kernels stay barrier-free and shared-memory-free until
  something needs them.** The grid build's prefix scan started as one cube
  with two `sync_cube()` barriers, which cost the CPU runtime 280 ms per
  step; three barrier-free launches cost the GPU backends a few µs and made
  the CPU runtime usable. A cube-wide barrier is not a portable primitive
  across this runtime set at this maturity.
- 2026-09-19 — **A kernel that calls `ln` or `cos` is not bit-identical
  across backends.** The mutation kernels' Box–Muller draw matches its
  plain-Rust twin exactly on the CPU runtime, but CUDA lands 2.4e-7 and
  wgpu 4.8e-7 away from it — about one ulp at these magnitudes, from the
  backends' own transcendental implementations. Integer decisions taken
  from the same RNG words (which structural edit, which limb) still match
  exactly everywhere, and the fluid's bit-reproducibility is untouched
  because no fluid kernel calls a transcendental. Rule for later kernels:
  exact within one runtime, tolerance across them, and keep anything that
  must agree across backends on the integer side of the draw.
- 2026-09-19 — **Restructuring a gather kernel needs a dump diff, not just
  the determinism test.** Two measured facts from the cleanup pass, both
  recorded next to the code they constrain in `kernels/grid.rs`: flattening
  the 3×3 neighbour loop to a single `0..9` loop cost wgpu's
  `calculate_accel` 14% (the nested loops stay); and applying the seam
  x-shift unconditionally (`pos + 0.0`) changed the CUDA backend's float
  contraction, moving 4157 of 51200 densities by up to 4 ulp after one
  step and diverging visibly by 50. Same-seed reproducibility on one
  backend cannot see either; only a byte-for-byte diff against the
  previous binary's dump can.
- 2026-09-19 — **Mode 1 is frozen; the experiment's terrain is mode 2.** The
  fix for the entry below is a new `TerrainMode::CapillaryField` rather than a
  change to `capillary_test`, because mode 1 is what every
  `resources/parity/*.bin` was generated from and `tests/parity.rs` measured
  this port against the C++ binary through it (both since deleted with the
  C++ tree). Mode 2 reuses
  `capillary_columns` for the x layout, so `SoilGrid::columns()` and every
  per-column metric work unchanged, and stops the soil at `--column-top`.
- 2026-09-19 — **`--column-top` is 0.25, not the 0.55 the brief specified,
  and above ~0.28 nothing lives.** Measured, `--terrain-mode 2 --founders 60
  --seed 42`, 4000 steps on CUDA, one run per value (the pool top is 0.2):
  0.25 leaves 120 alive and 57 lineages with 30 births against 32 deaths;
  0.27 leaves 44 alive; 0.28 leaves 29; 0.30 and 0.55 leave nothing at all —
  60 deaths, no births, and at 0.55 every founder is dead inside the first
  100 steps. A founder's six particles cost 0.018 a step and its one leaf in
  open light earns 0.012, so it lives on what its two root particles draw,
  and that is only the water the column's capillary suction has lifted to
  them. A column standing 5.6 m above the pool lifts nothing that high
  whatever it is made of, so at 0.55 every column is equally dry and the soil
  difference the experiment is about cannot express itself. At 0.25 — 0.8 m
  of column above the pool, 12 m of air above that — it plainly can: after
  4000 steps mean energy is 1.54 in the clay column against 0.18 in the sand
  one, which is the ordering of the capillary constants.
- 2026-09-19 — **Founders are spaced along the soil, not along the world.**
  `--founders N` put them at even x across the world, so on a capillary
  terrain about one in eight landed in a gap, where `soil_surface` falls back
  to the world floor — the bottom of the pool, with unlimited water. Those
  few were the only survivors of the first runs and would have been a tenth
  of the experiment's sample. They are now spread over the union of
  `SoilGrid::columns()`; a terrain whose single column spans the world is
  bit-identical to what it was.
- 2026-09-19 — **`--terrain-mode 1` has no room for a plant.** The
  capillary test fills every soil column from 20% height to the very top of
  the world and leaves the gaps between columns empty, so a founder
  anchored on its column's surface stands at the ceiling with nowhere to
  grow, and one in a gap anchors on the floor. Measured: at `--founders 64
  --terrain-mode 1 --iterations 500`, 114 of 384 body particles sit exactly
  at `y = bounds.y` and the worst segment is 73% short of its rest length;
  the same run on the noise terrain is 0.08%. Terrain mode 2 is the answer,
  above; mode 1 keeps this property and is kept for the `perf.md`
  baselines measured on it.
- 2026-09-19 — **The brain's latents need a normalization the slice table
  does not have.** One tick makes three residual adds onto the latent stream
  — the input cross-attention, the self-attention and the MLP — and the
  spec normalizes none of them, so with weights at `1/sqrt(fan_in)` each one
  multiplies the latent norm by about `sqrt(2)`. Measured on the default
  world at `--founders 256`: the largest latent grows ~2.6× per step,
  reaches 4e18 by step 40 and overflows to infinity by step 45, taking every
  head with it. A Perceiver-IO normalizes each block's *input*; scaling the
  recurrent state to unit RMS instead bounds the same thing in one launch
  rather than four, and being parameter-free it leaves the 18,890-parameter
  slice table alone — a learned LayerNorm per block would have added six
  slices and changed every genome. With it the largest latent sits at 4.83
  for all 500 steps. If the brain ever plateaus in a way that looks like a
  missing pre-norm, the pre-norm variant is the thing to try, and it costs
  parameters.
- 2026-09-19 — **fp16 brain weights are built and not the default.** The
  spec asks for fp16 storage with fp32 accumulate, on the reasoning that the
  trunk is bandwidth-bound. At the shipped size it is not: 256 organisms ×
  18,890 parameters is 19 MB per tick, ~13 µs of bandwidth spread over 22
  launches, so every launch is dominated by its own dispatch. Measured,
  fp16 is within noise on CUDA (44.9 µs against 45.7 µs of real GPU time)
  and ~4% *slower* on wgpu, while widening the gap to the plain-Rust
  reference from 3e-7 to 3e-4. `--brain-fp16` keeps the path alive for when
  the trunk is wide enough to be bandwidth-bound; `perf.md` has the numbers.
- 2026-09-19 — **A body's velocity is the integrated one plus the
  projection's displacement**, not the projection's displacement alone.
  `move_particles` has already integrated and possibly bounced the particle
  by the time the constraint pass runs, so `(ppos_projected -
  ppos_before_the_step) / dt` — the position-based-dynamics velocity —
  equals `vel + (ppos_projected - ppos_after_move) / dt`. Taking only the
  second term would throw a body's inertia away every step.
- 2026-09-19 — **The growth head decides *how fast* a genome's limb records
  grow, not *what* they are.** The spec said the head "decides whether to
  grow a child limb of which type", and the type logits existed for one
  chunk; they were read as a gate only — argmax against index 0, "none" —
  and the limb that grew was the next ungrown child record already in the
  genome, by child slot. Continuous growth deleted the gate and the type
  logits with it: `head_growth` is one logit per limb, added to the record's
  own `growth_bias` before the softmax that splits the tick's growth, so the
  brain sets a limb's tempo relative to its siblings and nothing
  else. Letting the head choose the type would make the body a product of
  the brain's runtime state rather than of the genome: the part type would
  stop being heritable, two siblings with one genome could grow different
  bodies, and `species_distance` — which is defined over the discrete
  section — would no longer describe anything an offspring inherits. That
  is Lamarckian development, and it takes the structural mutation operators
  out of the loop they are there to close. If evolved bodies ever plateau in
  a way that looks like "the genome cannot express this limb", the thing to
  try is a richer discrete section, not a brain that overrides it.
- 2026-09-19 — **An integer atomic is not the atomic the spec rules out.**
  The light grid's occluder count is an `atomicAdd` of 1 into a `u32` per
  soil cell. Integer addition is associative and exact, so the arrival
  order cannot change the result, and the run stays bit-reproducible. What
  "no atomics that decide order" rules out is an atomic whose *return
  value* is used — a slot grab — or a float accumulation, where the
  summation order is the result. Same rule applies to the organic matter a
  death leaves: that one is a float sum per cell, so it is done on the host
  in slot order instead.
- 2026-09-19 — **`upkeep_per_particle` is 0.003, not the 0.001 the
  derivation started from, and `germinate_speed` 0.5, not 0.05.** Measured
  at `--founders 64 --iterations 3000 --seed 42` on the noise terrain: at
  0.001 a founder's six particles cost 0.006 a step against 0.012 of light
  in the open, so open light alone pays for a whole body, nothing ever
  starves, and the run records 3 deaths against 99 births with every
  organism slot full by step 400 — a population with no selection in it. At
  0.003 the same run records 78 deaths against 68 births, ends at 93 of 256
  slots, and 45 of the 64 founder lineages are already extinct. 0.05 for
  the germination speed is below the fluid's own mean speed, so seeds
  almost never settled and held their slots until `seed_lifetime`; 0.5 is
  "has come to rest relative to the water around it". The other constants
  are as specified. `perf.md` has the runs.
- 2026-09-19 — **A run left to itself fills every organism slot after a few
  thousand steps, and that is `max_organisms`, not the energy balance.**
  At `--founders 64` the population sits between 40 and 110 for the first
  ~4000 steps, then evolution finds a body whose gain outruns upkeep, mean
  energy climbs without bound and the 256 slots fill by ~5000. The ceiling
  that should bind is light and space; at this body size, 256 plants in a
  32 m world do not shade each other enough to be that ceiling. Raising
  `max_organisms` moves the wall rather than removing it, so the real
  answer is bodies big enough to compete for light — which is the
  soil-specialization experiment's terrain work, not a constant.
- 2026-09-19 — **The GUI runs the sim on the wgpu runtime, always.** The
  renderer binds CubeCL's own buffers; a sim on the CUDA or CPU runtime
  would have to copy every buffer through the host each frame, which is the
  CUDA-GL interop the port set out to delete. `--runtime` is a headless
  flag.
- 2026-09-19 — **Growth is continuous, allocated by softmax, and children
  never outgrow parents.** Whole-limb sprouting was dropped after the first
  look at plants in the window: a limb appearing at full length in one tick
  let a child exist before its parent had any length, made growth order an
  accident of record indices, and never looked like growing. The C++ trees'
  continuous length and thickness is what looked organic, so that model
  returns on top of the particle bodies (the *Growth* section). The choices
  that could be re-litigated: softmax over per-limb biases rather than a
  fixed order, because a bias is only meaningful relative to siblings and a
  softmax slows a limb rather than deleting it; the parent rule is `child
  progress ≤ parent progress`, not a cap on the sum of children, because
  the sum expresses load, which breaking-by-thickness expresses better and
  which would punish bushy plans for being bushy; upkeep is by **volume**
  (`length × thickness²`), not by thickness or particle count, so a tall
  thin plant is cheaper than a tall thick one; the growth budget is the C++
  rule, a per-limb `growth_rate` summed and capped at `energy − seed_cost`,
  so growth never starves reproduction and each limb has its own tempo;
  the brain biases the split and not the total, and later nudges the joint
  angle — the genome stays the sole heritable body plan. Breaking scales
  with thickness so a plant that grows leaves faster than trunk snaps in
  the first current; roots are exempt because their failure is being
  pulled out, which is a different mechanism for later. One energy pool per
  organism stays; the old per-node give-take diffusion is not coming back,
  the softmax does that job.
- 2026-09-19 — **Continuous growth makes the energy balance the binding
  constraint, and at the shipped constants a seedling has no surplus to grow
  on.** Growth spends `energy − seed_cost`, so a plant grows only above its
  reproduction reserve, and a germinated seedling starts at `seed_energy`
  0.5 against a `seed_cost` of 1.0 — below the floor, with one root particle
  earning about what it costs in upkeep. Measured, `--founders 64 --seed 42`,
  3000 steps on CUDA: the founders (which start fully grown) carry the run to
  step ~800, after which the population is their offspring and the mean body
  drifts to root-only — `root_frac` 0.33 → 0.76, `leaf_count` 1.0 → 0.33,
  height 0.45 m → 0.12 m — while alive, births and deaths stay in the old
  range (71 alive, 52 births, 64 deaths, 18 lineages; 154 alive and
  generation 6 at `--seed-energy 1.5`). `--length-per-energy` 2 through 16
  changes none of it, which is what proves the conversion rate is not what
  binds: the surplus is. So the thing to change when seedlings should reach a
  leaf is the energy balance — what a root earns, what a seed germinates
  with, or whether the reserve is `seed_cost` — not the growth constants.
  `TODO.md` carries it next to the `--column-top` band it is the other half
  of.
- 2026-09-19 — **A founder starts fully grown; a seedling starts as one root
  segment.** `--founders N` is the run's initial condition rather than
  something the life cycle produced, so a founder is its genome laid out
  whole, as it always was, and `--load-pop` re-grows a saved body the same
  way; germination is the spec's `grown = 1` on the root. Founders as
  one-segment roots too would starve the run at the shipped constants (the
  entry above) and would make `--founders` a different experiment from the
  one every measurement in `perf.md` was taken with.
- 2026-09-19 — **The volume constants are the per-particle ones, re-expressed
  for one founder.** A founder at full length is 2 root segments at
  thickness 0.04, 3 stem at 0.05 and 1 leaf at 0.03 over
  `limb_segment_length` 0.15: volume 0.00174. `upkeep_per_volume` 10 makes
  that 0.0174 a step against the 0.018 the six particles cost at
  `upkeep_per_particle` 0.003, and `volume_per_energy` 0.0006 makes the whole
  body cost 2.9 energy to grow against the 3.0 it cost at
  `length_per_energy` 2.0 — so the 64-founder run starts from where it was
  and what changed is the *shape* of the two costs, not their scale. The
  thicknesses themselves are the only genuinely new numbers: a stem thicker
  than a root and a root thicker than a leaf, at a third of a segment's
  length, which is what makes a plant read as a plant in the window. What
  the change does move, deliberately, is that a growing limb is cheaper than
  a grown one (volume goes as `grown²`) and that a seedling's one root
  segment, half way along a two-segment root, costs 0.0012 a step where one
  particle cost 0.003.
- 2026-10-11 — **The C++ L-system trees' joint physics is the reference
  for a torque-actuated joint (milestone 2's actuated limb), and nothing
  else of them carries over.** The trees never ran in the shipped C++
  binary — `main.cu` pushed only the fluid-soil screen, the tree screen's
  line commented out — so their behavior was never seen and nothing live
  was lost when the tree went (tag `cpp-reference`, `src/systems/trees.cu`
  and `tree_types.cuh`). Per node, against its parent: torque
  `rot_delta × min_thickness⁴ × TORQUE_PER_RAD` (100), with `min_thickness`
  the thinner of node and parent — the fourth power is a beam's second
  moment of area, so stiffness falls off fast toward the tips; mass
  `length × π × thickness² × MASS_PER_CUBIC_CM`, where `MASS_PER_ENERGY` 1
  and `CUBIC_CM_PER_ENERGY` 1000 give 0.001 per cubic unit; moment of
  inertia `(1/3) m L²`, a rod about its end; angular acceleration
  `torque / I` clamped to ±10 (named a rad/s cap but applied to the
  acceleration); angular velocity damped ×0.96 per step, independent of
  `dt`, then integrated; a zero-length or massless node does not
  accelerate. A clamp of `rot_delta` to ±π/2 and a linear-in-thickness
  torque were both present and commented out. Child target thickness
  tapered ×0.95 per level. What the particle bodies use instead is a
  stiff projected target angle (`project_constraints`); a torque joint is
  the alternative when a limb must push against the world with a bounded
  force.
- 2026-10-11 — **The actuator is off by default (`actuator_gain` 0) until
  it is a torque with a rate limit and a cost.** The untrained recurrent
  brain emits a new head value every step, so a joint target can jump by
  up to `2 × actuator_gain` in one 1/600 s step for free, and the
  position-based velocity rule turns the projection's displacement into
  inertia that the fluid coupling then carries. Measured on the noise
  terrain, `--founders 20 --max-organisms 1024 --seed 3`, a dump at step
  11000 on CUDA: body-particle speeds at gain 0.3 are p90 0.38, p99 87 and
  max 630 m/s against a liquid p99 of 0.38; at gain 0 they are p90 0.23,
  p99 0.83 and max 0.2 m/s for limb particles (the run's 4.6 is a seed in
  flight), and the run sheds 489 limbs against 2491. The fastest bodies are
  founder-shaped at ordinary thicknesses, every one with a second root-typed
  limb — a base joint the brain nudges, standing in soil where only the
  projection acts on it, and exempt from breaking, so nothing ever ends it.
  Breaking cannot punish any of this, because the bend is measured against
  the nudged target. Milestone 2's actuated limb is the place to turn the
  gain back on, built on the torque joint of the entry above with a cost
  per radian; `--actuator-gain 0.3` reproduces the old behaviour.
- 2026-10-11 — **Water is a ledger, and it is what makes the population
  density-dependent.** Roots read water from the fluid and never removed
  it, so nothing a plant did used anything up: measured at `--founders 20
  --max-organisms 1024`, 15000 steps, eight seeds, two went extinct by step
  3400 and six filled the cap between steps 3100 and 9700, nothing in
  between, root fraction 0.59–0.94 at the cap and mean energy climbing
  without bound — a root-only plant on a wet spot reproduced forever without
  a leaf. The choices (*Water*): **roots drink whole particles, and the
  liquid particle decides** — each one picks its single nearest root, so no
  two units ever contend for a slot, nothing needs a float atomic or a slot
  grab, and the SPH kernels are untouched; **the credit is an integer
  count**, exact in any arrival order (the 2026-09-19 entry on integer
  atomics), turned into a float pool by the one unit that owns the
  organism. **Transpired water leaves at the leaf, as vapor, as the leaf
  earns** — not at the root and not the moment it is drunk — because
  transpiration tracks carbon gain rather than work, and water a plant
  holds without spending it is its turgor, which is what lets it grow
  (tissue water) and what it gives back when it dies. **Roots no longer
  earn energy**: wetness paid for upkeep was the loophole the root-only
  plants lived in, and with a pool the root's whole value is the water a
  leaf needs. **The growth reserve is gone**: growth spent `energy −
  seed_cost`, the C++ trees' "growth never starves reproduction", and a
  seedling germinates with no more than a seed costs (0.5 against 1.0 at
  the old defaults), so it could never grow the leaf it now has to have; reproduction is triggered by
  `seed_threshold` alone, and a seed costs exactly what it carries (3.0).
  **A death floors its water to whole particles** and drops the fraction:
  the ledger's one leak, under a particle per death or break, against the
  alternative of a fractional particle or a per-cell float sum. At the
  `light_gain` it shipped with, 0.012, every seed was extinct by step 600
  (`perf.md`), because a founder-shaped body's one leaf earns 0.012 a step
  against 0.0174 of upkeep — the root term had been carrying it — and
  `water_per_energy` from 0.25 to 4 does not move that; the entry below
  sets the energy scale.
- 2026-10-11 — **`light_gain` is 0.025, because that is where the water
  ledger holds a population between extinction and the cap on every
  seed.** Measured, `--founders 20 --max-organisms 1024`, 15000 steps on
  CUDA, eight seeds at each of three gains (`perf.md` has the table). At
  0.02 no seed goes extinct or hits the cap but the populations are small
  — 9 to 89 plants, 43 to 202 organisms with seeds in flight — pools sit
  at 1.5 to 4.7 particles, so water is not what limits them, and the
  bodies are the leafiest of the three (leaf count 0.5–0.9, height
  0.19–0.40 m). At 0.025 every seed holds 79 to 304 plants (182 to 827
  organisms) and spends 9,400 to 12,800 of the 15,000 steps between a
  tenth and nine tenths of the cap, with 5 to 14 lineages standing at the
  end; mean pools are 0.3 to 2.2 particles, so water is binding, which is
  the density dependence the ledger was built for. At 0.03 the plants are
  131 to 378, the bodies go root-heavy (root fraction 0.73–0.85, height
  about 0.1 m) and up to half the fluid is in the air as vapor at once
  (27,496 liquid against 23,367 vapor on seed 4). 0.025 is the middle: the
  first gain at which the cap is never the ceiling and water is. What the
  ledger does not yet do is also in the table: two thirds of a saturated
  `alive` is seeds in flight, and the water-limited bodies are short and
  root-heavy, which is what a water-limited plant should be but is also
  what the old loophole produced, so the body shape is not yet evidence of
  anything.
- 2026-10-12 — **The defaults are the run the ledger was calibrated on:
  `--founders 20` and `max_organisms` 1024.** A bare `alife` or `alife
  --headless` evolves without flags to remember; `--founders 0` is the
  fluid-only world. The slots are free: 3000 steps at `--founders 20
  --seed 1` on CUDA cost 1.639 ms a step at 256 slots and 1.651 at 1024,
  because the per-particle kernels run over the claimed prefix and the
  per-organism passes are launch-bound either way (`perf.md`).
- 2026-10-12 — **A limb's axis is its last whole segment, and a limb with
  none points where its joint aims it.** Nine in ten breaks were a
  measurement artefact, and it is why every run grew grass. A child's base
  joint was measured against its parent's last segment, which while the
  parent grows is the segment it is growing into: millimetres long when
  just claimed, with a direction that is the fluid's jostling. Of 177
  breaks logged over 6000 steps at seed 1, 130 had the parent's last
  segment under a tenth grown and 109 read a bend over 3 rad; one had a
  fully grown parent and a real fold; 135 were leaves. So a seedling shed
  its leaf each time its stem claimed a particle, and a limb added by
  mutation was shed the same way — at `structural_rate` 0.5 a run broke
  2526 limbs in 15000 steps against 498 at the default. Reading only whole
  segments cut that to 222, and the remainder was the rule's own doing: a
  stem at 0.955 of its first segment has no whole segment, its leaf's
  joint went unmeasured and unprojected, drifted 1.5 rad, and was swung
  back in one sweep when the stem claimed its next particle. Hence the
  walk: a limb with no whole segment takes its parent's axis turned by its
  grow angle, which is where its own joint is aiming it anyway, so no
  joint is ever free. Breaks over 15000 steps at seed 1: 498 before, 49
  after, and across seeds 1–4 a run with breaking on now matches one with
  `--break-angle 100` on plants, births, height and leaf count, so what is
  left is real folds. The bodies are still short (height 0.12–0.26 m, root
  fraction 0.54–0.79 at step 15000) — that is the economy and four
  generations, not the measurement. Ruled out on the way, by probes that
  did not move the breaks per birth: SPH pressure (near-pressure ×0.1,
  pressure ×0.25), the solver (1 or 16 iterations, stiffness 0.1–1.0),
  gravity, and young limbs' own thinness (`break_thickness_ref` 0.002).
- 2026-10-12 — **Body particles have quadratic drag and a speed cap,
  because the position-based solver had no damping and ran away.** A
  337k-step window run (defaults, random seed) dumped with 792 organisms
  at one every 4 cm had 19 with a particle over 5 m/s, the worst a
  six-particle leaf whose speed climbed along the chain to 1424 m/s at
  the tip with segments stretched to 1.6 m against 0.15, and the actuator
  was off. The same shape stood still a few metres away; neither wetness
  (the world was dry: 2445 liquid against 34312 vapor and 11261 drunk)
  nor nearness to other bodies separated the fast from the quiet. What
  distinguishes a runaway is only being stretched: past what four
  half-step corrections pull back, each step's correction adds more
  velocity than it removes, and viscosity acts only against liquid
  neighbours. Drag is quadratic because every limb is past Reynolds a
  thousand whenever it moves, and implicit (`v / (1 + k v dt)`) so no
  speed overshoots to a sign flip; at the default 1/m a limb swaying at
  0.1 m/s loses a hundred-thousandth of its speed a step and one at 100
  m/s a sixth. Drag alone balances a runaway's injection at a few hundred
  m/s, so the cap (5 m/s) is the guarantee and drag is the physics that
  keeps bodies away from it. Seven of the 19 were pinned single-particle
  roots, whose 5–12 m/s is the solver's residual and never moved them;
  it only showed because the window drew `pos`, a `dt_predict` ahead
  along it, which is why the window now draws `ppos`.
- 2026-10-12 — **The angle constraints' stiffness is 0.9, because under
  0.8 a long chain never settles.** With drag and the cap in, a
  300k-step run at seed 1 still had 48 of its 77 bodies of eleven or more
  particles oscillating at the cap and breaks rising to 0.4 per plant per
  thousand steps. Probed by loading that run's saved population and
  stepping 2000 steps with one thing changed: near-pressure off, pressure
  ×0.09, 16 sweeps, drag ×10, gravity off and viscosity off each left
  135–205 of 230 large bodies at the cap and 5821 breaks at the defaults;
  joint stiffness 1 alone 163, bend stiffness 1 alone 141, and both at
  0.6, 0.7, 0.8, 0.9, 1.0 gave 125, 100, 0, 0, 0 fast bodies and 1638,
  147, 155, 108, 123 breaks. The under-relaxed angle projections — 0.5
  and 0.3 per sweep, four sweeps — are a Gauss-Seidel limit cycle on a
  chain of ten or more particles, not a physical compliance: every other
  knob leaves it alone, and the founders' three short limbs never showed
  it. 0.9 is the stable value with the fewest breaks. A limb that is
  meant to bend in flow by a set amount needs compliance that does not
  depend on the sweep count, which is XPBD's Lagrange multiplier, not a
  stiffness under 0.8.
