# TODO

Outstanding work. Completed items get deleted — `git log` is the record of
what shipped.

## Inspection & verification (gates autonomous optimization work)

- **Simulation metrics.** Before any agent-driven fluid optimization, the
  project needs a numeric answer to "is this aspect good": candidates are
  mass/volume conservation over N steps, density error vs `target_density`,
  incompressibility (max velocity divergence), kinetic-energy settling time
  for a resting column, per-step wall time per particle, and capillary rise
  height per soil column in `--terrain-mode 1`. Design with Jonah first (one
  fork at a time); the output should be a headless mode that prints or
  writes them so a run can be scored without a window.
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
- Redesign `World` as a top-level composition of all systems (trees + soil +
  fluid + creatures).
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

- **Solver choice.** The current solver is SPH. FLIP/PIC or whatever is
  current SOTA for this scale is the intended direction for the autonomous
  optimization effort, but it is gated on the metrics above — no solver
  swap until a run can be scored.
- **Profile the SPH step.** The three neighbor-gather kernels are 98% of
  the ~0.45 ms step for 51k particles (`docs/perf.md`). Start with `ncu` on
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

## Plant system

- Couple trees with terrain + fluid: roots grow downward through soil,
  branching like the existing L-system; roots anchor plants and absorb
  water/nutrients; roots impede flow and water feeds root growth.

## Creatures

- Policy network engine (port from `alife_cuda_2`): block-based
  architecture (Linear, Activation, Add, Mul, OutputAct), JSON-defined archs,
  per-creature weights with strided batched GEMV (cuBLAS), mutation.
- Agent system: sensory inputs (eyes, proximity, internal state), motor
  outputs (movement, eating, reproduction), energy/metabolism, reproduction
  with mutation.

## Tooling

- **Rust port — revisit, not now** (decided 2026-09-18). The codebase is a
  good fit (25 flat kernels, no shared memory / CUB / cuBLAS, thrust as
  containers only) but NVIDIA's `cuda-oxide` is alpha: pinned nightly,
  custom LLVM 21, git-install, Linux only. Re-evaluate when it reaches
  crates.io on stable Rust. If curious sooner, the cheap probe is porting
  `particle_fluid2.cu` alone as a standalone experiment. GL interop would
  need two hand-declared FFI functions (`cuGraphicsGLRegisterBuffer` /
  `RegisterImage`); no crate exposes them.
