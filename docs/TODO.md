# TODO

Outstanding work. Completed items get deleted — `git log` is the record of
what shipped.

## Organisms (the critical path — spec in `organism.md`)

Milestone 1, plants. Each bullet is one delegation-cycle chunk; build from
the spec, don't re-derive it.

Left open by the soil-specialization experiment:

- **A plant lives only where liquid reaches its roots, which on terrain
  mode 2 was a 0.8 m band above the pool.** `--column-top` 0.25 worked and
  0.30 was total extinction when roots earned energy from wetness
  (`organism.md`, decisions, 2026-09-19); a root now drinks only the liquid
  particles within a smoothing radius of it, which is the same question —
  whether capillary suction lifts water to the column's surface — and has
  not been re-measured. Every column the experiment can use is therefore
  short. Widening the band — deeper roots, a wetter world, the transpired
  vapor that now rains back down — is what a terrain with real vertical
  structure needs, and it is a change to the water and energy balance
  rather than to the terrain.
- **Once every organism slot is full, `energy_mean` stops meaning
  fitness.** A plant over `seed_threshold` with no free slot to seed into
  keeps its energy, so a saturated column's mean energy measures how long
  it has been saturated. The per-column traits are unaffected; the energy
  comparison in `soil_score.py` is the one to distrust on a saturated run.
- **The transplant's destination column mixes.** Over the N/4 window new
  seeds germinate in the column the moved organisms were put in, so its
  `energy_mean` is the incomers plus whatever germinated beside them. An
  honest "moved versus resident" needs the metrics to split a column by
  lineage of origin, which is a column in the time series, not a design
  question.
- **`--load-pop` starts the fluid from scratch at step 0.** The saved
  step count and seed are provenance; a population comes back into a
  freshly initialised world rather than the one it left. Resuming both at
  once needs `--dump` and `--save-pop` to agree about a run, which nothing
  needs yet.

Left open by the bodies chunk:

- `project_constraints` is one unit per organism and so one cube per 256
  slots: 0.099 ms of the 1.378 ms step at `--founders 64 --max-organisms
  256` with the life cycle running (`perf.md`), on one SM.
  Splitting a unit per limb needs the sweeps to become separate launches,
  which is only worth it if organism counts stay this low.

Left open by the metrics chunk:

- **Per-column traits are binned by anchor**, so an organism anchored in the
  gap between two columns counts in `alive` and in no column. That is right
  while plants are anchored; a mobile creature needs binning by where it
  currently is.
- **`alive` counts seeds in flight**, because a seed holds an organism slot
  (`organism.md`, *Genome buffers*), and the per-column figures do not.
  Splitting the two in the time series is a column, not a design question —
  do it when a run turns on how many of the population are in the air. The
  experiment's runs already do: a saturated `alive` is mostly seeds.

Left open by the life-cycle chunk:

- **A population now settles between extinction and the cap, and what it
  looks like there is unexamined.** At `light_gain` 0.025 every seed holds
  79 to 304 plants for most of 15000 steps with water binding
  (`organism.md`, decisions, 2026-10-11; `perf.md` has the sweep). Three
  things the table shows and nobody has looked at: two thirds of `alive`
  is seeds in flight, which hold organism slots for up to `seed_lifetime`
  steps; the water-limited bodies are short and root-heavy (root fraction
  0.6–0.85, height 0.1–0.2 m), which is right for a water-limited plant but
  is also the shape the old loophole produced, so it is not yet evidence of
  adaptation; and up to half the fluid is in the air as vapor at once, so
  the condensation cycle's return rate is part of the carrying capacity and
  has not been tuned as one. The window is where to look first.
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
  offset would mean sampling the soil twice per limb.
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

## Fluid simulation

- **Solver choice — on hold** (`organism.md`, decisions). FLIP/PIC is not on
  the path to emergence; revisit only if fluid behavior itself blocks an
  organism milestone.
- **The pressure equation describes a gas.** `pressure_mult * (density -
  target_density)` (`kernels/accel.rs`) is linear in density, which is a
  gas's equation of state rather than a liquid's. A stiffer one (Tait) is the obvious candidate, and belongs with the
  solver question above rather than before it.
- **Profile the SPH step.** Organism bodies are particles in this system,
  so the per-particle cost of the three neighbour-gather kernels (93% of the
  ~0.49 ms step for 51k particles, `docs/perf.md`) is now the cost of
  bodies too. Start with `ncu` on `calculate_accel`: occupancy, achieved
  bandwidth, and what CubeCL's generated CUDA does with the neighbour loop —
  it measured 3% slower than the deleted C++ tree's hand-written
  `__device__` version, and `calculate_evap_prob` 9% (`perf.md`). The per-kernel timing synchronises after
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
  only after a run shows the stall. The actuator is **off by default**
  until then (`organism.md`, decisions, 2026-10-11): the instantaneous
  target nudge drives body particles to hundreds of m/s, so the actuated
  limb needs a rate limit and an energy cost per radian — the torque joint
  the C++ trees had is the reference.
- **Milestone 3, soil as actionable state.** Digger part type writing soil
  density through the same channel as erosion and deposition.
- **Adaptive mutation distribution** (lineage momentum or species-level ES,
  per-species knob, default off) measured against lineage diversity before
  it is ever on by default.
- **Reserved brain features**, added only on a plateau that looks like the
  missing piece: tree-distance attention bias, per-limb memory head.

## Substrate: Rust + CubeCL (decided 2026-09-18, see `organism.md` decisions)

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
  colors, particle size and the evap debug ramp, whether pan (middle drag)
  and zoom feel right, the mouse tool (left attracts, right repels,
  `+`/`-`/`[`/`]`), the density-grid overlay, the profiler window, the fps
  readout and Esc to quit (`REVIEW.md`).
- **Soil has no update step.** `update_soil` is empty, as `update_soil_cuda`
  was. `calculate_soil_saturation` was defined and never launched in the
  C++, so it did not come along; the `saturation` field is still in the SoA
  and still written by nothing.
- The port's own open questions live with the code: `sort_cells` is 9.7 µs
  of the 29.6 µs grid build and is one thread per cell; the prefix scan's
  middle stage is one unit doing 256 serial adds.
