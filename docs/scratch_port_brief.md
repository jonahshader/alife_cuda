# Scratch: the fluid + soil port brief (working tier)

The brief given to the port builder on 2026-09-18, kept here so a later
session can re-issue or resume it. Durable content lives in `TODO.md`
(*Substrate* chunk), `organism.md` (decisions), `perf.md` (numbers) and the
crate READMEs; delete this file once the port has landed on `main` and been
reviewed.

Reference dumps for the parity protocol are committed under
`resources/parity/` (C++ binary at `2cb23ab`, `--terrain-mode 1 --seed 42`):
`ref1.bin` (1 step), `ref2a.bin` and `ref2b.bin` (2 steps, two runs; their
difference is the run-to-run floor). A builder on a machine without CUDA
uses these instead of building the C++.

---

Read the root `CLAUDE.md` first, then `docs/TODO.md` (section "Substrate:
Rust + CubeCL" is your chunk; "Code health" lists what to carry vs. drop),
the "Decisions & dead ends" section of `docs/organism.md` (substrate
decision and spike outcome), `docs/perf.md` (toolchain traps and the
numbers to beat), and `crates/spike/README.md` +
`crates/spike/src/wgpu_setup.rs` (proven CubeCL/wgpu patterns: device
adoption, pooled-buffer offsets, no VERTEX usage). Build from those; do not
re-derive the design or re-litigate the substrate.

## Goal

Port the C++/CUDA fluid + soil simulation to Rust with CubeCL kernels, to
parity with the C++ binary, as a headless-first application with an egui
GUI. The L-system trees do NOT come along. The C++ tree stays untouched as
the reference.

Source of truth for behavior: `src/systems/particle_fluid2.cu`
(soil-coupled path only: the second `update_fluid(state, soil)` and the
kernel variants it launches; the soil-free duplicates are dead),
`particle_fluid2.cuh`, `soil.cu`/`soil.cuh` (including
`reset_soil_capillary_test` for `--terrain-mode 1` and the noise-based
default terrain), `src/config/sim_params.h` (every param, its default, TOML
key, CLI flag, help), `src/main.cu` (headless loop, `--dump` format
documented above its writer, the profiler printout). Read all of it before
designing.

## Design rules (settled; follow them)

- **Kernels faithful first, host idiomatic from day one.** Kernel math and
  loop structure line-for-line where CubeCL allows, same constants, same
  kernel ordering per step. Host side: modules, ownership, `Result`, enums
  (particle kind is an enum, not a magic byte), iterators. No
  transliterated C++ on the host.
- **Every kernel gets a plain-Rust reference implementation** and a test
  that runs both on a fixed small input and compares. This is the only
  kernel-debugging path (the LLVM JIT emits no symbols). The reference
  lives next to the kernel.
- **Deterministic grid build.** Replace the atomic `populate_grid_indices`
  with a stable sort-by-cell (counting sort or radix; CubeCL may have a
  sort in `cubecl-std`; if not, write a counting sort: cell histogram →
  exclusive scan → stable scatter). Same-seed runs must then be
  bit-reproducible on a given runtime; add a test that runs 20 steps twice
  and asserts bitwise-equal dumps. This is a deliberate difference from
  the C++.
- **Layout.** Cargo workspace already exists at the root (`Cargo.toml`,
  `Cargo.lock` is authoritative; do NOT regenerate it — add deps with
  `cargo add` and keep every `cubecl*` at `=0.11.0-pre.3`, `wgpu` stays
  30.x). Create `crates/alife-sim` (library: state structs, kernels,
  references, step functions, params, dump/load — no windowing, no egui)
  and `crates/alife` (binary: CLI, headless loop, GUI). Delete
  `crates/spike` in your LAST commit, after its wgpu-setup code has been
  absorbed into `alife` (carry its comments).
- **SoA state via one field list.** One declaration per SoA type generates
  the host struct, the device buffers, the kernel argument struct, and the
  dump/load field order, mirroring `soa_helper.h`'s `DEFINE_STRUCTS`. A
  `macro_rules!` is fine; a proc-macro crate is fine if `macro_rules!`
  gets ugly. Field order must match `FOR_SPH` in `particle_fluid2.cuh` so
  the C++ dump loads directly.
- **Sim params via one declaration each**, mirroring `sim_params.h`'s
  X-macro: a single struct with `#[derive(clap::Parser,
  serde::Deserialize, serde::Serialize)]` (or one attribute macro) so each
  field yields the CLI flag, the TOML key, the default and the help text.
  Precedence CLI > TOML > default, `--write-config` emits defaults,
  `config.toml` read from CWD, same flag names as the C++ (`--help` of
  both should list the same sim flags).
- **Runtime selection at startup**: `--runtime cpu|cuda|wgpu` (default:
  cuda if available, else wgpu, else cpu), plus wgpu adapter selection as
  the spike showed. The CUDA runtime on the dev box needs
  `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64` (perf.md explains).
- **RNG**: counter-based Threefry4x32 as a `#[cube]` function (20 rounds,
  Random123 constants); bit-exact match with Random123 is nice-to-have,
  not required.
- **Noise terrain**: use the `fastnoise-lite` crate (a port of
  FastNoiseLite). Check whether it reproduces the C++ terrain for the same
  seed/params; report either way.
- **Timing**: per-kernel timing in headless mode printed at exit like the
  C++ profiler (use CubeCL's profiling API if the pinned version has one
  usable per launch, else wall time around `client.sync()` per kernel as
  the spike did, and say which).
- Libraries: `eframe`/`egui`/`egui-wgpu` (versions that resolve against
  the locked `wgpu 30`), `clap` (derive), `serde` + `toml`, `glam`,
  `tracing`, `bytemuck`, `anyhow`/`thiserror`. Anything else, your call;
  list it in the report.
- `cargo +1.98.1 fmt` and `cargo +1.98.1 clippy --all-targets` clean
  before each commit.

## Parity protocol (the acceptance test)

The C++ binary is not bit-reproducible run to run (atomic grid order), so
parity is: (a) after ONE step from an identical initial state, per-particle
differences are at the C++ run-to-run floor; (b) over 50 steps, aggregates
match to ~6 significant figures.

1. Reference data is in `resources/parity/` (see the header of this file).
   On a CUDA machine you may also regenerate it: `cmake -S . -B build
   -DCMAKE_BUILD_TYPE=Release && cmake --build build -j16`, then
   `./build/alife_cuda --headless --terrain-mode 1 --seed 42 --iterations 1
   --dump ref1.bin`, and `--iterations 2` twice for `ref2a`/`ref2b`.
2. Add `--load <dump>` to the Rust binary: initialize particles from a C++
   `--dump` file (same field order). Soil for parity runs comes from
   `--terrain-mode 1` (deterministic, no noise) unless the noise port
   matches, in which case also run mode 0.
3. Rust: `--load ref1.bin --terrain-mode 1 --iterations 1 --dump
   rust2.bin`. Assert max-abs-diff(rust2, ref2a) per field ≤ ~3×
   max-abs-diff(ref2a, ref2b). Do this on every runtime available (cpu,
   cuda, wgpu).
4. Aggregates over 50 steps (mean/max density, mean near-density, mean
   speed, mean y, vapor count) Rust vs C++ at seed 42, mode 1, each from 3
   runs; report side by side. (C++ aggregates need a CUDA machine; if none,
   report the Rust side and mark the C++ column as pending.)
5. Make the 1-step check a `cargo test` that skips cleanly if the
   reference files are absent, and document how to regenerate them in the
   crate README.

## Commit plan (atomic, by explicit path, no push, in this order so an interruption leaves a useful tree)

1. workspace: `alife-sim` + `alife` crates scaffold, params struct, SoA
   macro, dump/load, CPU-runtime headless loop that steps a trivial kernel.
2. grid build (sort-based) + reference + test.
3. density kernel + reference + test.
4. accel (pressure/viscosity/capillary/soil friction) + reference + test.
5. evap prob, evaporate, move, vapor move + references + tests.
6. soil: init (both terrain modes), update, saturation coupling.
7. headless parity: `--load`, timing printout, the parity test, numbers
   recorded in `docs/perf.md` (new dated subsection, exact commands,
   commit hashes) and the aggregates table.
8. GUI: eframe window, particle renderer reading CubeCL buffers as storage
   (vertex_index-indexed), soil renderer, an egui panel with the same
   controls the ImGui `FluidSoil` screen has (read
   `src/screens/fluid_soil.cu` and `src/graphics/*.cu`). Build it; DO NOT
   run it (no window may open; the user checks visuals).
9. delete `crates/spike`; update docs: root `CLAUDE.md` *Build, test & run*
   (Rust commands become primary, C++ marked reference-only),
   `docs/CLAUDE.md` index if you add a doc, `docs/TODO.md` (delete
   completed bullets, add anything you had to leave), `docs/perf.md`.

If you are interrupted, the next session resumes from your commits; keep
the tree building at every commit.

## Environment

Dev box: 16 cores / 76 GB / RTX 5090. Foreground builds only, 600000 ms
timeout. Rust toolchain `+1.98.1`. `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`
for the CUDA runtime. No window ever. Scratch files in your worktree's
`target/` or `/tmp`; clean up `/tmp` at the end. Never drop an existing
comment silently: the C++ comments that explain why (kernel choices, TODOs
about gas-vs-liquid kernels, "scale with mass?") move to the Rust.

## Report shape

1. Commits (hash + one line each).
2. Parity: the per-field 1-step numbers vs the floor on each runtime; the
   50-step aggregates table; the noise-terrain match result.
3. Timing per kernel per runtime vs `docs/perf.md`'s C++ numbers.
4. Decisions the brief left open and what you chose (sort algorithm,
   timing method, any library).
5. What is not done or deviates from the brief, and why.
6. Pre-existing issues found in the C++ while porting, unfixed.
