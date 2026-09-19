# Organism design

The settled design for organisms: one representation for plants and
creatures, a fixed-shape genome, a brain over body-part tokens, bodies made
of particles in the fluid's particle system, and selection by ecology rather
than a fitness function. **Status: milestone 1 runs — the genome, the
particle bodies, the brain's forward pass, and energy with a life cycle
that turns the population over on its own. What is left of the milestone
is the soil-specialization experiment and its controls.** The open work is
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
- Segment (capsule) collision is the upgrade if evolved bodies keep wanting
  long thin limbs or chunky collisions block fine manipulation. It changes
  the discretization, not the genome, since a limb is already a type and a
  length either way.

Part types at the first milestone: **root** (pinned in soil, draws water
from the cell's saturation), **stem** (structure), **leaf** (gains energy
from light), **seed** (a reproduction particle). Later types are additions,
not new systems: an **actuated limb** whose base joint takes a target angle
from the brain, a **digger** that decrements soil density in its cell, a
**mouth**, and **sensors**. Burrowing, erosion, and deposition all write
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
- **Output heads per limb:** actuator commands (target joint angle, later
  more), and a **sprout head** that decides whether to grow a child limb of
  which type. Development and control share the trunk.
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

- **Light.** A per-soil-cell occlusion grid, rebuilt each tick. One kernel
  counts the stem and leaf particles per cell into a `u32` grid; a second,
  one unit per soil column, walks from the top down and multiplies the
  light by `light_attenuation` (default 0.7) once per particle it has
  passed, starting at `light_top` (1.0). The sensors read the same grid, so
  between rebuilds a limb's light reading is up to `life_interval` steps
  stale.
- **Energy.** One unit per organism, serial over its particles. Leaves earn
  `light_gain` times the light in their cell; roots earn `water_gain` times
  `max(0, water − 1)`, where `water` is the brain sensor's own quantity
  (`density / target_density`, which reads exactly 1 in air because of the
  soil's solid-density offset, so the `− 1` is the wetness). Every particle
  costs `upkeep_per_particle`. Energy below zero is death: every particle
  goes back to the free pool and leaves `organic_matter_per_particle` in
  the soil cell it was standing in.
- **Growth.** A plant with `energy ≥ sprout_cost` whose sprout head does
  not say "none" grows one ungrown child record of its genome, lowest child
  slot first, and is charged. At most one limb per organism per tick.
- **Reproduction.** Energy above `seed_threshold` is spent on a seed
  particle carrying a mutated copy of the genome. The seed is a particle
  with no limb, so no constraint holds it: it falls, water carries it, and
  dispersal is free. A seed that comes to rest (speed below
  `germinate_speed`) in soil or on the floor germinates — its particle is
  released, its root grows from where it landed, it starts on
  `seed_energy`, and only then does it count as a birth. One that drifts
  for `seed_lifetime` steps without landing dies unremarked.
- **Selection.** None explicit. Energy, shading, water, space, and other
  organisms are the fitness function; the simulation is the evaluator.

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
anchor sits in, so one anchored in a gap counts in `alive` and in no column.
Species are greedy clusters: each organism joins the first cluster whose
representative it is within `species_threshold` (default 0.25) of by
`species_distance`, else starts one. Phylogenetic depth is the `generation`
field of the organism SoA: a founder is 0 and a child is its parent's plus
one. Sampling only reads, so a run with `--metrics` steps the same world as
one without.

### The first experiment: soil specialization

Run plants in the capillary test terrain (`--terrain-mode 1`), where sand,
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

## Milestones

1. **Plants.** Anchored organisms with root, stem, leaf, and seed types; the
   sprout head drives growth; energy, death, reproduction; evolutionary
   metrics; the soil-specialization experiment with its controls. This
   exercises every piece above except actuators.
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
max_limbs × max_particles_per_limb` (defaults 256 × 16 × 8); the comptime
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
grid cell size); the angle at each limb's base joint against the parent
segment's direction (target from the limb record for plants, from the brain
for actuated limbs later, stiffness `joint_stiffness`); bend, keeping a
limb's consecutive segments in line (`bend_stiffness`), or a limb is a rope;
and the world bounds, x wrapping and y clamping. Velocity is recomputed from
the projected displacement, position-based-dynamics style. Slot allocation
is deterministic: births claim `Free` particle slots and free organism slots
in order via a prefix scan over the free flags, never via atomics.

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
  `identity` (`[f32; 4]`).
- Continuous, `[max_organisms × BrainShape::param_count()]` f32 on the
  host as the master copy, uploaded per organism on birth; the fp16 device
  shadow is the brain chunk's concern.
- Lineage: `parent_id`, `birth_step`, `lineage_id` (root ancestor),
  `generation` (hops from the founder; mutation never touches it), plus
  the runtime `alive` flag, `energy`, `stage`, and the latent state
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
  records — one unit for both would put 18,890 serial iterations on a
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
| `head_sprout`, `sprout_b` | `d_token × n_types`, `n_types` (logits over child type, 0 = none) |
| `head_actuator`, `actuator_b` | `d_token × 2`, `2` (target angle, reserved) |
| `latent_init` | `n_latents × d_latent` |

18,890 parameters at the defaults. Initialization is Gaussian with sigma
`1/sqrt(fan_in)` per slice, where a weight slice is stored `[input dim ×
output dim]` so its fan-in is its row count; biases are zero and
`latent_init` is sigma 1. One tick of the brain: tokens from limb
geometry and sensors; input cross-attention latents→tokens; latent
self-attention; MLP; gated update of the persistent latents, scaled to unit
RMS; output cross-attention tokens→latents; heads per limb. Single-head
attention, fp32 accumulate, one pass per tick. The trunk's activation is
SiLU. Token features are built inline in the token kernel rather than
materialized, and the heads are `[max_organisms × max_limbs × (n_types +
2)]` — sprout logits then actuator outputs — zeroed for an absent limb or a
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
the token features; the life tick reads the sprout head the forward pass
just wrote, so applying a head and deciding the life cycle are one step.

**The life tick's shape.** Light and energy are kernels; growth, placement,
freeing and mutation are kernels; the *decisions* between them are a serial
walk over the organism slots on the host, in slot order, and therefore
deterministic. It costs one `read` per tick of a packed
`[max_organisms × (6 + max_limbs)]` float buffer — energies, seed positions
and speeds, body tops, and the sprout head's argmax per limb, reduced on
the device rather than shipped — plus the free-slot scan's reads when
something is claimed and one limb-record download when something was born.
The organism SoA is host-mastered except `energy`, which the energy kernel
writes; the brain tensor and the limb records are device-mastered for a
newborn, because that is where the mutation kernels wrote them, so a birth
never uploads the whole population.

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
  The C++/CUDA tree is the reference until parity, then deleted.
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
- 2026-09-19 — **`--terrain-mode 1` has no room for a plant.** The
  capillary test fills every soil column from 20% height to the very top of
  the world and leaves the gaps between columns empty, so a founder
  anchored on its column's surface stands at the ceiling with nowhere to
  grow, and one in a gap anchors on the floor. Measured: at `--founders 64
  --terrain-mode 1 --iterations 500`, 114 of 384 body particles sit exactly
  at `y = bounds.y` and the worst segment is 73% short of its rest length;
  the same run on the noise terrain is 0.08%. The soil-specialization
  experiment needs `capillary_test` to leave headroom above the columns
  before plants can live in it — that is the experiment chunk's to fix, not
  the bodies chunk's, because it changes the terrain every existing parity
  reference was generated from.
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
- 2026-09-19 — **The sprout head decides *when* a genome's limb records
  grow, not *what* they are.** The spec says the head "decides whether to
  grow a child limb of which type", and the type logits exist; they are
  read as a gate only — argmax against index 0, "none" — and the limb that
  grows is the next ungrown child record already in the genome, by child
  slot. Letting the head choose the type would make the body a product of
  the brain's runtime state rather than of the genome: the part type would
  stop being heritable, two siblings with one genome could grow different
  bodies, and `species_distance` — which is defined over the discrete
  section — would no longer describe anything an offspring inherits. That
  is Lamarckian development, and it takes the structural mutation operators
  out of the loop they are there to close. The type logits stay in the
  head's shape and stay informational; if evolved bodies ever plateau in a
  way that looks like "the genome cannot express this limb", the thing to
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
