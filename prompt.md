# Handoff — tuning the evolution loop, interactively with Jonah (2026-10-11)

Next session is an **interactive investigation**, not a delegation cycle:
Jonah and Claude watch, measure and tune the evolution loop that exists now
until it behaves like an ecology. Work one question at a time with a short
recommendation (root `CLAUDE.md`, *Design discussions*). Jonah runs the
window; Claude measures headless and reads the code. Don't build new
features unless a tuning question turns out to need one — if it does,
settle the design in chat first.

## Where things stand

The C++ tree is gone (tag `cpp-reference`); the Rust sim on `main` is the
whole project. This session also added the GUI's mouse water tool (LMB
pull, RMB push, `+`/`-` radius, `[`/`]` strength, **middle-drag pans**),
a density-grid overlay (View), a live profiler window, an fps readout and
Esc to quit — all untested by eye, so say if anything looks wrong.

Read first: `docs/organism.md` (*Growth*, *Energy*, the decisions log),
`crates/alife-sim/src/life/mod.rs` (`life_tick` → `decide` → `apply`) and
`life/energy.rs`. The parameters are the `organism.*` and `life.*` lines in
`crates/alife-sim/src/params.rs`; each is a CLI flag and a TOML key.

## What Jonah saw (`alife --founders 20 --max-organisms 10000`, GUI)

1. **The population grows exponentially until it hits the cap.** Nothing
   limits it but `max_organisms` — not light, not water, not space.
2. **Some plants spin around wildly.**
3. **Plants stay where they started and spread very slowly.**

None of this has been investigated yet. Likely starting points, unverified:

- (1) is the energy balance the 2026-09-19 session flagged: under volume
  pricing a seedling's first segment is cheap, so nothing starves. Knobs:
  `light_gain`, `water_gain`, `upkeep_per_volume`, `volume_per_energy`,
  `seed_cost` / `seed_energy` / `seed_threshold`, and whether the growth
  reserve should be `seed_cost`. Light competition (`light_*`) is the
  mechanism that *should* cap density — check whether shading actually
  bites. `docs/TODO.md`'s first growth bullet and its `--column-top`
  bullet are the same question.
- (2) may be `actuator_gain` (0.3 rad by default): an untrained brain
  wanders every joint, and the 2026-09-19 sweep (`docs/perf.md`) saw limb
  breaks double at 0.3 and the tallest, best-fed bodies at 0.1. It may also
  be a constraint-solver instability (`kernels/constraints.rs`) — tell the
  two apart before tuning.
- (3) is seed dispersal: how far a seed drifts before it lands slow enough
  to germinate (`germinate_speed`, `seed_lifetime`, and how seeds are
  launched in `life/mod.rs`).

## How to investigate

- **Headless with metrics is the measuring tool**: `./target/release/alife
  --headless --founders 20 --iterations N --seed S --metrics run.csv`
  writes the time series (alive, births, deaths, breaks, height, energy,
  species …) and prints a summary (`docs/organism.md`, *Metrics*). Compare
  settings by seed-matched runs, a few seeds each — one run is an anecdote.
- **The window is Jonah's**: never start a GUI run unless he asks
  (root `CLAUDE.md`). When he reports what he sees, turn it into a
  headless measurement that captures it.
- Record each settled tuning call as a dated entry in `organism.md`'s
  *Decisions & dead ends* with the numbers that justified it, and change
  the default in `params.rs` in the same commit. Runs worth keeping go in
  `docs/perf.md`.
- Ask Jonah how many cores builds may use; last session he capped it at
  four (`taskset -c 0-3 cargo … -j4` — `-j` alone does not cap the test
  harness or the CubeCL CPU backend).

## Parked, not for this session unless it comes up

- **Tool palette** for the GUI: a default Select tool (hover stats, click to
  inspect, right-click menu) with the water grabber as a tool you switch
  to, so its circle isn't always drawn. Agreed in principle, not built.
  An inspector would help tuning — raise it if watching single plants
  becomes the bottleneck.
- **No test compares backends with each other** since the parity test went;
  a CPU/CUDA/wgpu aggregate comparison would close it.
- `docs/REVIEW.md` still has Jonah's older review items.
