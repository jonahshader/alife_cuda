# TODO

Outstanding work. Completed items get deleted — `git log` is the record of
what shipped.

## Organisms (the critical path — spec in `organism.md`)

Milestone 1, plants. Each bullet is one delegation-cycle chunk; build from
the spec, don't re-derive it.

- **Soil-specialization experiment** in `--terrain-mode 1`, with the three
  controls in the spec (identical-soil isolation control, soil-position
  permutation, transplant test). It must leave air above the columns: today
  they run to the top of the world, so a plant anchored on a column's
  surface stands at the ceiling (`organism.md`, decisions). Changing the
  terrain regenerates every parity reference, so it belongs to this chunk
  rather than an earlier one.

Left open by the bodies chunk:

- `project_constraints` is one unit per organism and so one cube at the
  default `max_organisms` of 256: 0.099 ms of the 1.378 ms step at
  `--founders 64` with the life cycle running (`perf.md`), on one SM.
  Splitting a unit per limb needs the sweeps to become separate launches,
  which is only worth it if organism counts stay this low.

Left open by the metrics chunk:

- **Per-column traits are binned by anchor**, so an organism anchored in the
  gap between two columns counts in `alive` and in no column. That is right
  while plants are anchored; a mobile creature needs binning by where it
  currently is.
- **`alive` counts seeds in flight**, because a seed holds an organism slot
  (`organism.md`, *Genome buffers*). The per-column figures do not: a seed
  has no anchor yet. Splitting the two in the time series is a column, not
  a design question — do it when a run turns on how many of the population
  are in the air.

Left open by the life-cycle chunk:

- **A run left to itself fills every organism slot after a few thousand
  steps** (`organism.md`, decisions, 2026-09-19). The binding constraint is
  `max_organisms`, not light or space, because plants at the founder body
  size do not shade each other enough in a 32 m world. The
  soil-specialization terrain work is where that gets tested properly.
- **The `stage` field is not in the dump**, because the population tensors
  are not: `--load` resumes the fluid and reserves body capacity on top
  (`crates/alife-sim/README.md`). A run cannot be checkpointed and resumed
  with its organisms until the dump carries them.
- **`soil.organic_matter` is host-side only.** A death deposits into the
  host `SoilGrid` and the device mirror is not refreshed, because no kernel
  reads the field. Whoever first has a kernel read it — erosion, or a root
  that prefers rich soil — owns uploading it, or moving the master to the
  device.
- **The free-slot scan reads the whole `free_ids` buffer back** (336 KB at
  the defaults) for the handful of ids a tick claims. `read_free_slots`
  has no way to ask for a prefix; a handle slice or a device-side count
  would cut it to nothing. It is two reads on a tick that claims anything,
  measured inside the 0.066 ms/step the whole life tick costs
  (`perf.md`), so it is not urgent.

Left open by the brain chunk:

- **The actuator head is still read by nothing.** `brain::forward` writes
  the sprout logits and then two actuator outputs; the life tick reads the
  sprout logits' argmax and ignores the rest, because no part type takes a
  target angle yet. Milestone 2's actuated limb is what consumes them.
- **`contact` is the spec's `solid fraction > 0.5`, and pure clay sits
  exactly on the threshold** (`1 - CLAY_POROSITY` is 0.50, against sand's
  0.62 and silt's 0.55), so a limb buried in undiluted clay reads no
  contact. Deciding whether contact should instead come from soil presence
  — which has no such edge, but cannot tell one soil from another — belongs
  with whoever first uses the channel.
- **The `water` sensor carries the soil's solid-density offset**, because
  `sph.density` does (`kernels::density` adds `solid_density_at_pos`, which
  is `target_density` in air). The reading therefore sits near 1 in free air
  rather than near 0. It is a usable signal as it stands; subtracting the
  offset would mean sampling the soil twice per limb. The energy kernel
  takes the same quantity and subtracts 1 from it, so the two agree about
  what "wet" means and moving one moves the other.
- **22 launches, ~6% of the step, and entirely launch-bound** (`perf.md`).
  Merging them buys back microseconds against a step the three neighbour
  kernels dominate, so it is not worth doing until organism counts or brain
  width grow by an order of magnitude. If they do, the first merges are the
  scores kernels into their attends, at 11x the arithmetic.
- **fp16 weight storage is built and off.** `--brain-fp16 1` works on all
  three runtimes and is within noise of fp32 on CUDA and ~4% slower on wgpu
  (`perf.md`), because the pass is launch-bound rather than
  bandwidth-bound; it costs three orders of magnitude of accuracy against
  the reference. Revisit when the trunk is wide enough to be bandwidth-bound.

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

The fluid and soil are ported and at parity (`perf.md`) on the CPU and CUDA
runtimes; on wgpu one field, `evap_prob`, lands at 3.2× the C++ run-to-run
floor instead of 1× (float contraction in the shader compiler; nothing
downstream diverges, `REVIEW.md`). What is left:

- **Delete the C++ tree.** It is reference-only now. It still owns the
  L-system trees, which the plant milestone supersedes, so it goes when
  particle-body plants render — together with `src/`, `CMakeLists.txt`, the
  SDL/ImGui/GL dependencies and the parity references that check against it.
- **Segfault at exit when CUDA is present but unusable.**
  `CUDA_VISIBLE_DEVICES="" ./target/release/alife --headless --iterations 1`
  runs on wgpu and then dies with SIGSEGV in teardown (exit 139,
  reproduced 2026-09-19): the auto-selection probe creates a CUDA client
  that panics, and its runner thread is torn down badly. A machine with a
  CUDA library but no usable device is a real target. Probe the driver
  through `cudarc` (init + device count) before ever creating a CubeCL
  CUDA client, so no failed client exists to unwind.
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
