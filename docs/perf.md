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

Toolchain (2026-09-18): CUDA 13.3, driver 595.91.07, CMake 4.2, GCC host.

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
