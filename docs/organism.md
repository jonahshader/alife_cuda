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
starting point for the trunk; its block-DAG engine assumes one MLP
architecture per population and is not ported as-is.

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
- 2026-09-18 — **Fluid solver swap (FLIP/PIC) is on hold.** Fluid fidelity
  is not on the path to emergence. SPH kernel optimization stays relevant
  because bodies ride on the same per-particle cost.
