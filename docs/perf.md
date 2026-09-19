# Performance baselines

Measured numbers per machine. Every entry records the exact command and the
commit it was taken at, so a later run is comparable. A number without a
command is not a baseline.

## How to measure

```
cd build && ./alife_cuda --headless --iterations 100
```

prints per-kernel timings in milliseconds from `TimingProfiler` on exit.
Each section is bracketed by a CUDA event pair and synchronized, so trivial
kernels report a floor of a few microseconds and the run is serialized; for
launch overlap or sub-kernel detail use Nsight:

```
nsys profile -o /tmp/hl --stats=true ./alife_cuda --headless --iterations 20
nsys stats --report cuda_gpu_kern_sum /tmp/hl.nsys-rep
```

The two agree to within a few percent for the neighbor kernels. A run that
finishes did execute its kernels: `check_cuda` is fatal, so a failed launch
exits non-zero instead of printing plausible numbers for kernels that never
ran.

## dev box — RTX 5090 (sm_120) / 16-core / 76 GB

Toolchain (2026-09-18): CUDA 13.3 toolkit, driver 595.91.07, CMake 4.2,
GCC host. **The driver reports CUDA 13.2 and rejects PTX from the 13.3
toolchain.** The C++ tree never notices because nvcc emits SASS ahead of
time; anything that JITs from PTX (the CubeCL CUDA runtime does, via nvrtc)
must load the 13.2 libraries: `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`.
Otherwise the failure is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

Rust tree: CubeCL 0.11.0-pre.3 needs rustc ≥ 1.95; the default stable here
is 1.93.1, so a named toolchain is installed (`rustup toolchain install
1.98.1`) and builds use `cargo +1.98.1`. The CPU runtime downloads a
prebuilt LLVM 22.1.4 (36 MB, cached under `~/.cache/tracel`) and ignores
the system LLVM. Clean release build of the whole workspace with the bundle
cached: **1 m 32 s at `-j16`**.

**Build ceiling: `-j16` is fine for this project** — a full rebuild
including SDL2 and the CUDA translation units completed at `-j16` with
headroom (2026-09-18). Unlike `mg-rl-rewrite`, no OOM has been observed
here; re-measure if translation units grow.

### Headless baseline, default config (2026-09-18, commit after 4454b4a)

World 32×16 m, smoothing radius 0.2 → 160×80 cells × 4 particles per cell =
**51,200 particles**. Nsight kernel summary over 20 steps, GPU otherwise
idle, clocks at ~2.9 GHz under load:

| Kernel                       | avg (µs) |
|------------------------------|---------:|
| `calculate_accel`            |      234 |
| `calculate_evap_prob`        |      107 |
| `calculate_particle_density` |      105 |
| `populate_grid_indices`      |      2.4 |
| `move_particles`             |      1.2 |
| `move_vapor_particles`       |      1.2 |
| `evaporate_particles`        |      1.1 |
| `reset_particles_per_cell`   |      0.5 |

About **0.45 ms of GPU time per simulation step**, 98% of it in the three
neighbor-gather kernels. The GUI reports ~1.95 ms per frame including
rendering and the density-grid pass.

### Neighbor kernels after the grid-tiling fix (2026-09-18)

Same machine, same default config as the table above. Commits: `6a2e26c`
(before), `2233423` (grid tiles the width exactly, seam widening gone),
`f668795` (`viscosity_kernel` takes `dst2`).

```
./build/alife_cuda --headless --iterations 200
nsys profile -o /tmp/hl --stats=false --force-overwrite=true \
  ./build/alife_cuda --headless --iterations 20
nsys stats --report cuda_gpu_kern_sum /tmp/hl.nsys-rep
```

Nsight kernel averages over 20 steps, each measurement repeated (µs):

| Kernel                       | 6a2e26c     | 2233423            | f668795            |
|------------------------------|------------:|-------------------:|-------------------:|
| `calculate_accel`            | 235.2 238.9 | 227.2 230.3 229.2  | 230.1 227.4 226.3  |
| `calculate_evap_prob`        | 108.8 108.4 | 103.7 104.0        | 103.6 104.2        |
| `calculate_particle_density` | 107.6 106.7 | 103.3 103.2        | 103.4 103.8        |

`TimingProfiler` averages over 200 steps (ms) agree on the grid fix —
`calculate_accel` 0.233 → 0.229–0.231, `calculate_evap_prob` 0.111 → 0.102,
`calculate_particle_density` 0.109 → 0.103 — so dropping the fourth seam
column is worth **~3–5%** on each neighbor kernel. The viscosity change is
**within noise**: nsys sees no difference and the event profiler reads ~1.5%
*slower* on `calculate_accel` (0.231 → 0.234), i.e. the duplicate `length()`
was already being folded into the one inside `density_kernel_gradient`.

Sim behavior is unchanged by both: at `--seed 42`, 50 steps, mean and max
density, mean speed and vapor count match before and after to 7 significant
figures. Per-particle trajectories do **not** match, and cannot: the same
binary run twice at one seed already diverges the same way (max abs density
diff 0.03–2.2 at 50 steps, 1.2e-4 after 1 step), because `atomicAdd` in
`populate_grid_indices` decides the per-cell particle order and with it the
float summation order.

### CubeCL spike, grid build + density on every runtime (2026-09-18, commit 2a56cb0)

```
cargo +1.98.1 build --release -j16
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/spike all
```

Same 51,200 particles and 160×80 grid, soil-free density kernel (not
like-for-like with the table above, which is the soil-coupled one). Wall
time around reset + populate + density + sync, after a warm-up launch,
averaged over 100 launches:

| Runtime                          | per launch | max abs diff vs plain-Rust reference |
|----------------------------------|-----------:|-------------------------------------:|
| `cubecl-cuda`                    |   0.068 ms | 9e-5 |
| wgpu / Vulkan on the 5090        |   0.079 ms | 6e-5 |
| `cubecl-cpu` (LLVM JIT, 16 threads) | 2.24 ms | 6e-5 |
| wgpu / Vulkan on llvmpipe        |   2.98 ms | 6e-5 |

Per-cell counts and particle-id sets are bit-exact everywhere; the float
differences are summation order from the atomic grid build. The native CPU
runtime beats llvmpipe and beats a single-threaded Rust loop 5.6×, so it is
the GPU-less path, not a fallback. First-launch JIT cost: CUDA ~200–350 ms,
wgpu and CPU ~10–50 ms, cached across runs.

### Rust port, every runtime (2026-09-19, commit a993941)

The port reached parity here; these replace the spike's numbers above, which
measured a soil-free kernel on a throwaway harness.

```
cargo +1.98.1 build --release -j16
# per-kernel timings, printed on exit; the first step is a warm-up and is not
# counted, because every kernel is JIT-compiled on its first launch
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 \
  ./target/release/alife --headless --seed 42 --runtime cuda --iterations 201
./target/release/alife --headless --seed 42 --runtime wgpu --iterations 201
./target/release/alife --headless --seed 42 --runtime cpu  --iterations 21
```

Default config: world 32×16 m, smoothing radius 0.2, 160×80 cells, 51,200
particles, noise terrain. Averages in ms per launch.

| Kernel                       | C++ event profiler | Rust CUDA | Rust wgpu | Rust CPU |
|------------------------------|-------------------:|----------:|----------:|---------:|
| `calculate_accel`            |              0.227 |     0.245 |     0.231 |    4.660 |
| `calculate_evap_prob`        |              0.105 |     0.123 |     0.107 |    2.054 |
| `calculate_particle_density` |              0.105 |     0.121 |     0.105 |    2.183 |
| grid build, all of it        |              0.031 |     0.087 |     0.080 |    0.460 |
| `evaporate_particles`        |              0.006 |     0.013 |     0.003 |    0.104 |
| `move_particles`             |              0.005 |     0.012 |     0.003 |    0.128 |
| `move_vapor_particles`       |              0.005 |     0.013 |     0.003 |    0.083 |
| **per step**                 |          **0.485** | **0.615** | **0.533** | **9.67** |

The three columns are **not** measured the same way and only the wgpu one is
device time. `ComputeClient::profile` reports device timestamps on wgpu and
falls back to wall time around a sync on CUDA and CPU, which the printout
names. Wall time carries a ~11 µs floor per launch, so the CUDA column
overstates every cheap kernel; the C++ event profiler has its own ~5 µs
floor. Use Nsight for a like-for-like CUDA comparison:

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 nsys profile -o /tmp/hl \
  --stats=false --force-overwrite=true \
  ./target/release/alife --headless --seed 42 --runtime cuda --iterations 21
nsys stats --report cuda_gpu_kern_sum /tmp/hl.nsys-rep
```

Nsight averages over 21 steps (µs), against the C++ numbers in the table
above:

| Kernel                       | C++ (f668795) | Rust CUDA |
|------------------------------|--------------:|----------:|
| `calculate_accel`            |   226.3–230.1 |     235.6 |
| `calculate_evap_prob`        |   103.6–104.2 |     112.9 |
| `calculate_particle_density` |   103.4–103.8 |     110.9 |
| grid build, all of it        |           2.9 |      29.6 |
| **per step**                 |       **~450** |  **~492** |

About **9% more GPU time per step**, and the neighbour kernels are 3–9%
slower each. The grid build costs 10× what the C++ one does — 29.6 µs
against 2.9 µs — which is what determinism is worth here: a counting sort
with a prefix scan and a per-cell sort, instead of one atomic per particle.
It is 6% of the step.

The scan is three launches with no cube barrier. It started as one cube with
two `sync_cube()` barriers, which cost the **CPU runtime 280 ms per step** —
a barrier at that cube size is ~140 ms there. Rewriting it barrier-free took
the CPU runtime from 292.7 ms to 9.7 ms per step and cost the GPU backends
2.5 µs (CUDA) and 39 µs (wgpu) on the scan itself.

### Rust port vs the C++, same state (2026-09-19, commit f288d7a)

The C++ is not bit-reproducible run to run — `populate_grid_indices`'s
atomics fix each cell's summation order, and `calculate_accel` reads
`sph.vel` while writing it — so parity is stated against that binary's own
floor. `crates/alife-sim/README.md` has the commands; `tests/parity.rs`
runs the check.

One step from `ref1.bin`, max absolute difference per field against a C++
run of the same two steps, as a multiple of the spread across three C++
runs:

| Field          | C++ floor | Rust CUDA | Rust wgpu | Rust CPU |
|----------------|----------:|----------:|----------:|---------:|
| `pos`          |  3.815e-6 |     1.00× |     1.00× |    1.00× |
| `ppos`         |  3.815e-6 |     1.00× |     1.00× |    1.00× |
| `vel`          |  7.641e-5 |     1.00× |     1.00× |    1.00× |
| `density`      |  4.944e-3 |     1.00× |     1.00× |    1.00× |
| `near_density` |  1.837e-2 |     1.01× |     1.00× |    1.01× |
| `evap_prob`    |  3.278e-5 |     0.99× |     3.16× |    1.00× |
| `state`        |         0 |    exact |     exact |    exact |

`evap_prob` sums gradient terms that nearly cancel, so its absolute error is
not bounded by the floor the way the others are; wgpu's shader compiler
contracts floats differently and lands at 3.2×. Terrain mode 0 measures the
same, within a C++ floor that itself moves by 5× between pairs of runs.

Aggregates 50 steps in, all four binaries resumed from `ref1.bin`, three
runs each. Every Rust run is bit-identical to its own repeats, so one row per
backend:

| Run       | mean density | max density | mean near-density | mean speed | mean y    | vapor |
|-----------|-------------:|------------:|------------------:|-----------:|----------:|------:|
| C++ ×3    |    385.15814 |   531.84436 |         444.77954 | 0.91266389 | 3.9665165 |     4 |
|           |    385.15814 |   531.84442 |         444.77960 | 0.91266386 | 3.9665160 |     4 |
|           |    385.15814 |   531.84448 |         444.77954 | 0.91266382 | 3.9665160 |     4 |
| Rust CUDA |    385.15814 |   531.84436 |         444.77954 | 0.91266396 | 3.9665165 |     4 |
| Rust wgpu |    385.15814 |   531.84436 |         444.77954 | 0.91266396 | 3.9665160 |     4 |
| Rust CPU  |    385.15814 |   531.84558 |         444.77954 | 0.91266404 | 3.9665160 |     4 |

Seven to eight significant figures on every aggregate, and the vapor count
matches exactly — the same particles evaporate, so the Threefry stream lines
up with Random123's.

The noise terrain reproduces the C++ **bit for bit**: the `mt19937_64` draws,
the FastNoiseLite samples, the column heights and the softmax composition all
match, which `soil::tests::noise_terrain_matches_the_cpp` pins against values
printed by a probe built on the C++ tree's own headers.

