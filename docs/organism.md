# Organism design

The settled design for organisms: one representation for plants and
creatures, a fixed-shape genome, a brain over body-part tokens, bodies made
of particles in the fluid's particle system, and selection by ecology rather
than a fitness function. **Status: specified, not yet implemented.** The
open work is in `TODO.md`; this file is the spec builders build from. When
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
  gated update. They are the organism's working memory. One pass per tick.
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

- **Energy.** Leaf particles gain energy from light, computed by a
  per-column occlusion scan from the top of the world so plants shade each
  other. Root particles draw water from their soil cell's saturation. Every
  particle costs energy per tick. Energy below zero is death; the body's
  particles become organic matter in the soil.
- **Reproduction.** Energy above a threshold is spent on a seed particle
  carrying a mutated copy of the genome. The seed is a particle, so it
  falls, water carries it, and dispersal is free. A seed that lands in soil
  pins its root and starts growing.
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
particle count is that capacity. Per particle: `organism` (u32,
`u32::MAX` for none), `limb` (u8), `index_in_limb` (u8). Body particles are
SPH boundary particles: they carry mass and contribute density to their
neighbors, and they integrate with the same forces as liquid (gravity,
pressure, viscosity, soil friction). After integration a **constraint
pass** projects them, one unit per organism, serial over that organism's
particles, a fixed number of Gauss-Seidel iterations (default 4): distance
to the previous particle in the limb (rest length ≤ cell size), the angle
at each limb's base joint against the parent segment's direction (target
from the limb record for plants, from the brain for actuated limbs later),
and the root's first particle pinned to its soil cell. Velocity is
recomputed from the projected displacement, position-based-dynamics style.
Slot allocation is deterministic: births claim `Free` particle slots and
free organism slots in order via a prefix scan over the free flags, never
via atomics.

**Genome buffers**, population SoA with capacity `max_organisms`:

- Discrete, `[max_organisms × max_limbs]`: `part_type` (u8, 0 = absent;
  1 root, 2 stem, 3 leaf, 4 seed, 5–7 reserved), `length` (u8, particles),
  `parent` (u8 limb index; the root limb is its own parent), `child_slot`
  (u8), `grow_angle` (f32, the rest angle at the base joint), `identity`
  (`[f32; 4]`).
- Continuous, `[max_organisms × BrainShape::param_count()]` f32 on the
  host as the master copy, uploaded per organism on birth; the fp16 device
  shadow is the brain chunk's concern.
- Lineage: `parent_id`, `birth_step`, `lineage_id` (root ancestor), plus
  the runtime `alive` flag, `energy`, and the latent state
  `[max_organisms × n_latents × d_latent]`, which is state, not genome.
- Mutation is a kernel over newborn organisms, Threefry-keyed by child slot
  and step: Gaussian perturbation of the brain tensor (`mutation_sigma`),
  and with probability `structural_rate` one structural edit: add a limb
  (random present parent, first absent record, random type and length,
  child slot = next free), remove a limb (mark absent, cascade to its
  children), or retype; identity and grow angle get their own small
  Gaussian. Species distance is over the discrete section: number of
  differing part types plus normalized identity distance, computed on the
  host for metrics.

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

About 18.9k parameters at the defaults. Initialization is Gaussian with
sigma `1/sqrt(fan_in)` per slice. One tick of the brain: tokens from limb
geometry and sensors; input cross-attention latents→tokens; latent
self-attention; MLP; gated update of the persistent latents; output
cross-attention tokens→latents; heads per limb. Single-head attention,
fp32 accumulate, one pass per tick.

**Step order** once organisms exist: grid build (all non-`Free`) → density
→ evap probability → accel → evaporate → move liquid → move vapor →
sense (token features per limb) → brain forward → apply heads (sprout,
actuator targets) → constraint pass → energy and life cycle. The bodies
chunk owns the constraint pass and publishes limb geometry (root-relative
position, segment angle, depth) for the token features; the brain chunk
owns sense → forward → apply.

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
- 2026-09-19 — **The GUI runs the sim on the wgpu runtime, always.** The
  renderer binds CubeCL's own buffers; a sim on the CUDA or CPU runtime
  would have to copy every buffer through the host each frame, which is the
  CUDA-GL interop the port set out to delete. `--runtime` is a headless
  flag.
