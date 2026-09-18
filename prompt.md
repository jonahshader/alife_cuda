# Handoff — workflow scaffolding is in; next is the metrics design (2026-09-18)

First session on this repo since 2026-04-04. It revived the build, committed
the water-cycle work that had sat uncommitted since April, decided against a
Rust rewrite for now, and ported the working conventions from
`mg-rl-rewrite` (root `CLAUDE.md`, `docs/CLAUDE.md`, `TODO.md`, `REVIEW.md`,
`perf.md`, the Sonnet-pinned review workflows, `worktree.baseRef: head`).

## State of `main`

- Builds and runs on the dev box (RTX 5090, CUDA 13.3, CMake 4.2). The fixes
  that made that true: glm 1.0.1, `CMAKE_CUDA_ARCHITECTURES` defaulting to
  `native`, SDL libsamplerate off.
- The particle fluid has a liquid/vapor state per particle with
  evaporation from the density-gradient surface signal and altitude-weighted
  condensation, rolled with Random123 Threefry. Five tunables in
  `sim_params.h`; a "Debug Evap Colors" toggle in the Particle Fluid panel.
- Capillary constants were retuned hard (sand 5 / silt 100 / clay 8000) and
  `--terrain-mode 1` now has a shared water pool under separated soil
  columns. Jonah has looked at it running but has not signed off on the
  numbers as final.

## Next

1. **Design the simulation-metrics / inspection path with Jonah** — the
   first item in `docs/TODO.md`. Nothing agent-driven on the fluid should
   start until a headless run can be scored. Work it one fork at a time in
   chat; then it is a delegation-cycle chunk.
2. After that: profile SPH with the numbers, then the FLIP/PIC question.

## Things this session learned

- A five-month gap moved CMake to 4.x (rejects sub-3.5 minimums in fetched
  deps) and CUDA to 13.3 on a Blackwell GPU (driver refuses to JIT PTX from
  a newer toolkit, so the binary must carry a native cubin). Stale
  `CMakeCache.txt` remembered dev headers that were no longer installed;
  wiping the cache was faster than fixing probes one by one.
- Headless works from any CWD; the GUI needs `resources/` as CWD. There is
  no frame cap flag yet (`TODO.md`).
- **A masked failure looked like a pass.** `check_cuda` used to print and
  continue, so an sm_75 build whose every kernel failed to launch exited 0
  and printed timings for kernels that never ran; this session briefly
  reported that as a clean smoke test. `check_cuda` is fatal now.
- **The headless profiler printout was mislabeled by 1000×** (it
  multiplied millisecond durations by 1000 and still printed "ms"), which
  this session briefly read as a 0.46 s step and wrote into the docs before
  Nsight showed the real 0.45 ms. Fixed; when a number is surprising,
  confirm it with `nsys` before writing it anywhere (`docs/perf.md`).

## Environment (machine-specific)

Dev box: 16 cores, 76 GB RAM. A full rebuild at `-j16` is fine for this
project (`docs/perf.md`). Build dir is `build/` (Release). Nothing is
pushed from this session unless Jonah asks.
