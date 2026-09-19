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
