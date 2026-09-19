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
