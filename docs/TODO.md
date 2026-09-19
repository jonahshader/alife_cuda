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
- **Frame cap / `--fps` flag** for the GUI, plus a `--shot <png>` that
  renders one frame offscreen and exits — the minimum for an agent to look
  at the sim without parking a window. `egui_wgpu::capture` is the offscreen
  path.

## Code health (C++ tree — reference only; fix nothing in place)

Everything here is either already handled in the port or dies with the C++
tree. Listed so a future session recognises them rather than re-deriving
them.

- `World` as a top-level composition of all systems (soil + fluid +
  organisms) with one init and one step shared by headless and GUI. The
  Rust `Sim` is that for fluid + soil; organisms join it.
- `--extended-lambda` and `--expt-relaxed-constexpr` are set in
  `CMakeLists.txt` but nothing uses device lambdas or device-side
  `constexpr`. `FetchContent_Populate` is deprecated (CMake 4 warns) for
  glad, artery_font, lodepng, FastNoiseLite, imgui.
- `soil_render.cu` re-registers/unregisters its GL buffer every frame;
  `rect_tex_renderer.cu` creates a `cudaTextureObject_t` no kernel samples;
  `curand_kernel.h` is included in three files with every use commented out;
  `screens/tree_test.cu` and `systems/trees.cu` carry large commented-out
  blocks.
- `--write-config` emits repeated `[fluid]`/`[world]` section headers, which
  its own loader rejects, and prints floats with `%f`, so the `dt` it writes
  reads back 2e-4 off 1/600. The Rust writer groups sections and prints
  round-trippable literals.

## Fluid simulation

- **Solver choice — on hold** (`organism.md`, decisions). FLIP/PIC is not on
  the path to emergence; revisit only if fluid behavior itself blocks an
  organism milestone.
- **Profile the SPH step.** Organism bodies are particles in this system,
  so the per-particle cost of the three neighbour-gather kernels (93% of the
  ~0.49 ms step for 51k particles, `docs/perf.md`) is now the cost of
  bodies too. Start with `ncu` on `calculate_accel`: occupancy, achieved
  bandwidth, and what CubeCL's generated CUDA does with the neighbour loop —
  it is 3% slower than the hand-written `__device__` version, and
  `calculate_evap_prob` is 9%. The per-kernel timing synchronises after
  every launch, so it hides any launch overlap; use `nsys` for the timeline.
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

## Substrate: Rust + CubeCL (decided 2026-09-18, see `organism.md` decisions)

The fluid and soil are ported and at parity (`perf.md`). What is left:

- **Delete the C++ tree.** It is reference-only now. It still owns the
  L-system trees, which the plant milestone supersedes, so it goes when
  particle-body plants render — together with `src/`, `CMakeLists.txt`, the
  SDL/ImGui/GL dependencies and the parity references that check against it.
- **The GUI has never been looked at.** It builds and its shaders validate,
  but nobody has opened the window. Check it before trusting it: soil
  colors, particle size and the evap debug ramp against the C++ screen, and
  whether pan/zoom feel right.
- **Interaction is not ported.** The C++ screen's mouse grab/repel
  (`attract_fluid`, the `+`/`-`/`[`/`]` keys) and the density-grid overlay
  (`calculate_fluid_density_grid`) have no Rust equivalent yet.
- **Soil has no update step.** `update_soil` is empty, as `update_soil_cuda`
  was. `calculate_soil_saturation` was defined and never launched in the
  C++, so it did not come along; the `saturation` field is still in the SoA
  and still written by nothing.
- The port's own open questions live with the code: `sort_cells` is 9.7 µs
  of the 29.6 µs grid build and is one thread per cell; the prefix scan's
  middle stage is one unit doing 256 serial adds.
