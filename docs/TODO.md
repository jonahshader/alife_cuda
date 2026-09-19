# TODO

Outstanding work. Completed items get deleted — `git log` is the record of
what shipped.

## Organisms (the critical path — spec in `organism.md`)

Milestone 1, plants. Each bullet is one delegation-cycle chunk; build from
the spec, don't re-derive it.

- **Genome + population tensors.** Discrete limb records and the
  fixed-shape continuous brain tensor as SoA population buffers; Gaussian
  and structural mutation operators; species distance over the discrete
  section.
- **Particle bodies.** Limbs as particle chains in the fluid's particle
  system with distance and base-joint angle constraints (rest length capped
  at the smoothing radius); root pinning in soil; part types root, stem,
  leaf, seed. Particles need a per-particle organism id and limb index.
- **Brain forward pass.** Perceiver-IO over limb tokens: embeddings
  (type, spatial, rotation, depth, child slot, identity), persistent gated
  latents, input and output cross-attention, fixed trunk, sprout head. One
  batched pass per tick across the population; fp16 weights, fp32
  accumulate, starting from `alife_cuda_2`'s custom batched GEMV kernel.
- **Energy and life cycle.** Per-column light occlusion scan, leaf energy
  gain, root water draw from soil saturation, per-particle upkeep, death to
  soil organic matter, seed particle emission with a mutated genome,
  germination on landing in soil.
- **Evolutionary metrics.** Headless `--metrics <path>` writes a sampled
  time series (population, births, deaths, energy flux, lineage count,
  phylogenetic depth, trait distributions per soil type) plus a summary at
  exit derived from the same samples. This is the scoring path for every
  later agent-driven run.
- **Soil-specialization experiment** in `--terrain-mode 1`, with the three
  controls in the spec (identical-soil isolation control, soil-position
  permutation, transplant test). The capillary test reset must publish its
  column extents so per-column metrics can be computed.

## Inspection & verification

- **Fluid metrics** (density error vs `target_density`, max velocity
  divergence, kinetic-energy settling, capillary rise per column) are
  secondary to the evolutionary metrics above and gate only fluid solver
  work; add them to the same `--metrics` time series when that work starts.
- **Test harness.** None exists. Pick one (Catch2 is what `mg-rl-rewrite`
  uses) and start with behavior-property tests over headless runs rather than
  goldens pinned to epsilon.
- **Frame cap / `--fps` flag** for the GUI, plus a `--shot <png>` that
  renders one frame offscreen and exits — the minimum for an agent to look
  at the sim without parking a window.

## Code health

- Remaining ports from `alife_cuda_2`: CPU/GPU duality (Device enum + OpenMP
  fallbacks) for debuggability; determinism infrastructure (counter-based
  RNG is in for the fluid via Random123, fixed-point energy and determinism
  tests are not); `CopyLevel` enum for partial state copies
  (FULL/NORMAL/RENDER).
- Redesign `World` as a top-level composition of all systems (soil + fluid +
  organisms).
- `--extended-lambda` and `--expt-relaxed-constexpr` are set in
  `CMakeLists.txt` but nothing uses device lambdas or device-side
  `constexpr`; drop them.
- `FetchContent_Populate` is deprecated (CMake 4 warns) for glad,
  artery_font, lodepng, FastNoiseLite, imgui; convert to
  `FetchContent_MakeAvailable`.
- `soil_render.cu` re-registers/unregisters its GL buffer every frame; the
  other renderers register once.
- `rect_tex_renderer.cu` creates a `cudaTextureObject_t` that no kernel
  samples; `curand_kernel.h` is included in three files with every use
  commented out. Remove both.
- `screens/tree_test.cu` and `systems/trees.cu` carry large commented-out
  blocks; delete or restore.

## Fluid simulation

- **Solver choice — on hold** (`organism.md`, decisions). FLIP/PIC is not on
  the path to emergence; revisit only if fluid behavior itself blocks an
  organism milestone.
- **Profile the SPH step.** Organism bodies are particles in this system,
  so the per-particle cost of the three neighbor-gather kernels (98% of the
  ~0.45 ms step for 51k particles, `docs/perf.md`) is now the cost of
  bodies too. Start with `ncu` on
  `calculate_accel`: occupancy, achieved bandwidth, and whether
  `-rdc=true` is blocking inlining of the `__device__` helpers in the
  neighbor loop. The `TimingProfiler` synchronizes after every section,
  so it hides any launch overlap; use `nsys` for the timeline.
- Autotune kernel launch parameters (block size, particles per cell).
- Spatial adaptive resolution — fewer particles where velocity is low.
- Time-smoothed velocity (EMA) for stable erosion calculations.
- Erosion and deposition: pickup probability from velocity vs friction
  threshold (Hjulstrom-inspired); carried material type and amount per
  particle; drop probability inverse to pickup (hysteresis); continuous soil
  density change on pickup/drop; use the smoothed velocity to avoid
  flicker.

## Organisms, later milestones (`organism.md`)

- **Milestone 2, mobile creatures.** Actuated limb type (base joint takes a
  target angle from the brain), mouth, contact and proximity sensors.
  Locomotion is evolved, not given; the thrust-part escape hatch is built
  only after a run shows the stall.
- **Milestone 3, soil as actionable state.** Digger part type writing soil
  density through the same channel as erosion and deposition.
- **Adaptive mutation distribution** (lineage momentum or species-level ES,
  per-species knob, default off) measured against lineage diversity before
  it is ever on by default.
- **Reserved brain features**, added only on a plateau that looks like the
  missing piece: tree-distance attention bias, per-limb memory head.
- The existing L-system trees (`systems/trees.cu`, `tree_types.cuh`) are
  superseded by the plant milestone; delete or absorb once particle-body
  plants render.

## Tooling

- **Rust port — revisit, not now** (decided 2026-09-18). The codebase is a
  good fit (25 flat kernels, no shared memory / CUB / cuBLAS, thrust as
  containers only) but NVIDIA's `cuda-oxide` is alpha: pinned nightly,
  custom LLVM 21, git-install, Linux only. Re-evaluate when it reaches
  crates.io on stable Rust. If curious sooner, the cheap probe is porting
  `particle_fluid2.cu` alone as a standalone experiment. GL interop would
  need two hand-declared FFI functions (`cuGraphicsGLRegisterBuffer` /
  `RegisterImage`); no crate exposes them.
