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

### Particle bodies (2026-09-19, commit after 392d460)

Organisms add `max_organisms × max_limbs × max_particles_per_limb` = 32,768
body slots on top of the 51,200 fluid particles, and two passes per step.
Default config, CUDA, the per-kernel printout (wall time around a sync, so
every cheap kernel carries its ~11 µs floor):

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 \
  ./target/release/alife --headless --seed 42 --runtime cuda \
  --iterations 501 --founders <N>
```

| Kernel                       | main (1eca794) | `--founders 0` | `--founders 64` |
|------------------------------|---------------:|---------------:|----------------:|
| `calculate_accel`            |          0.260 |          0.261 |           0.264 |
| `calculate_evap_prob`        |          0.129 |          0.130 |           0.131 |
| `calculate_particle_density` |          0.126 |          0.127 |           0.128 |
| grid build, all of it        |          0.089 |          0.088 |           0.088 |
| `evaporate_particles`        |          0.013 |          0.012 |           0.013 |
| `move_particles`             |          0.012 |          0.012 |           0.012 |
| `move_vapor_particles`       |          0.012 |          0.012 |           0.012 |
| `project_constraints`        |              — |              — |           0.109 |
| `write_limb_geometry`        |              — |              — |           0.015 |
| **per step**                 |      **0.641** |      **0.642** |       **0.772** |

`--founders 0` is main's cost because the per-particle kernels are launched
over the live prefix — the fluid plus the high-water mark of claimed body
slots — not over the whole capacity; launching over the capacity cost the
three neighbour kernels 6–9% each and the step 5.8%. The organism passes are
skipped outright when nothing is alive, so they do not appear at all.

`project_constraints` is one unit per organism over `max_organisms` = 256
units, so it is a single cube on one SM regardless of how many organisms are
alive: 0.109 ms whether 64 or 256 slots are filled. It is 14% of the step at
`--founders 64` and is the obvious thing to split if organism counts grow.

Byte-for-byte fluid parity at `--founders 0` against main, checked by
dumping both binaries and comparing the fluid prefix field by field: CUDA
and wgpu at 50 steps, the CPU runtime at 20, terrain modes 0 and 1, all
identical.

### The metrics sampler (2026-09-19, commit after d466745)

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/alife --headless \
  --seed 42 --runtime cuda --founders 64 --terrain-mode 1 --iterations 1000 \
  [--metrics /tmp/m.csv]
```

Process wall time, three runs each: **3.80 / 3.84 / 3.85 s** without
`--metrics` against **3.835 / 3.835 / 3.837 s** with it, at the default
`--metrics-every 100`. The sampler's own readback — five particle fields
(`state`, `density`, `organism`, `part_type`, `ppos`) over the live prefix —
is **0.25 ms per sample**, 2.5 ms across the ten samples of that run, so at
K=100 it is under 0.1% of the step budget and the difference above is noise.
Dumps at `--founders 0` and `--founders 16` are byte-identical with and
without the flag: the sampler only reads.


### The brain forward pass (2026-09-19, commit after 0463de4)

One brain tick per step: sense, tokens, and the 22 launches of
`brain::forward`. Same default world as above, 83,968 particle slots.

```
cargo +1.98.1 build --release -j16
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 \
  ./target/release/alife --headless --seed 42 --runtime cuda \
  --iterations 501 --founders <N> [--brain-fp16 1]
./target/release/alife --headless --seed 42 --runtime wgpu \
  --iterations 501 --founders <N> [--brain-fp16 1]
```

Per launch, µs, averaged over 500 steps. **Only the wgpu columns are device
time.** The CUDA columns are wall clock around a synchronise, which carries
a ~11 µs floor, and every brain launch is under that floor — the CUDA column
measures the profiler, not the kernel. Nsight is below.

| launch                | CUDA 64 | CUDA 256 | wgpu 64 | wgpu 256 |
|-----------------------|--------:|---------:|--------:|---------:|
| `brain_sense`         |    13.6 |     13.8 |     4.2 |      4.3 |
| `brain_tokens`        |    14.1 |     14.2 |     4.2 |      4.3 |
| `brain_in_q`          |    12.5 |     13.1 |     8.3 |      8.4 |
| `brain_in_kv`         |    12.8 |     14.5 |     8.3 |     12.5 |
| `brain_in_scores`     |    13.1 |     13.0 |     4.1 |      4.3 |
| `brain_in_attend`     |    12.9 |     13.1 |     4.7 |      5.0 |
| `brain_in_out`        |    12.1 |     12.9 |     8.3 |      8.4 |
| `brain_self_q`        |    11.6 |     12.6 |     8.3 |      8.4 |
| `brain_self_kv`       |    11.6 |     13.2 |     8.3 |      8.4 |
| `brain_self_scores`   |    12.1 |     12.4 |     4.2 |      4.3 |
| `brain_self_attend`   |    11.0 |     11.1 |     2.8 |      3.1 |
| `brain_self_out`      |    11.8 |     12.8 |     8.3 |      8.4 |
| `brain_mlp1`          |    12.0 |     13.2 |     4.2 |      5.1 |
| `brain_mlp2`          |    13.1 |     13.3 |    12.6 |     13.0 |
| `brain_gate`          |    12.7 |     13.1 |     4.2 |      4.3 |
| `brain_latent_norm`   |    11.0 |     11.2 |     3.0 |      3.4 |
| `brain_out_q`         |    11.9 |     13.3 |     8.3 |      8.4 |
| `brain_out_kv`        |    11.8 |     13.2 |     8.3 |      8.4 |
| `brain_out_scores`    |    12.2 |     12.6 |     4.2 |      4.3 |
| `brain_out_attend`    |    11.0 |     11.4 |     3.0 |      3.5 |
| `brain_out_out`       |    11.9 |     13.3 |     8.3 |      8.4 |
| `brain_heads`         |    12.4 |     12.5 |     4.3 |      4.6 |
| **brain, all of it**  | **269** |  **284** | **134** |  **143** |
| **per step (ms)**     |   1.033 |    1.085 |   0.833 |    0.859 |

Nsight over 21 steps is the real CUDA number, and it is an order of
magnitude below what the event profiler reports:

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 nsys profile -o /tmp/hl \
  --stats=false --force-overwrite=true \
  ./target/release/alife --headless --seed 42 --runtime cuda \
  --iterations 21 --founders 256
nsys stats --force-export=true --report cuda_gpu_kern_sum /tmp/hl.nsys-rep
```

Avg ns per launch, and the per-step total for the launches each kernel
serves:

| kernel              | launches/step | 64 (ns) | 256 (ns) | 256 fp16 (ns) |
|---------------------|--------------:|--------:|---------:|--------------:|
| `gemv`              |            11 |   1,595 |    2,409 |         2,339 |
| `attn_scores`       |             3 |   2,112 |    2,182 |         2,187 |
| `attn_attend`       |             3 |   1,252 |    1,502 |         1,501 |
| `write_sensors`     |             1 |   1,855 |    1,902 |         1,873 |
| `write_tokens`      |             1 |   1,588 |    1,626 |         1,624 |
| `gate_update`       |             1 |   1,608 |    1,905 |         1,928 |
| `latent_norm`       |             1 |   1,061 |    1,265 |         1,257 |
| `write_heads`       |             1 |   1,396 |    1,446 |         1,463 |
| **brain per step**  |        **22** |**35.1 µs**|**45.7 µs**|  **44.9 µs** |

Against the fluid in the same runs (`calculate_accel` 238/248 µs,
`calculate_evap_prob` 112/114, `calculate_particle_density` 110/112,
`project_constraints` 88/124, grid build 29.6, the rest ~5), the step is
**618 µs at `--founders 64` and 679 µs at 256**, and the brain is **5.7% and
6.7%** of it. The pass is launch-bound, not arithmetic- or bandwidth-bound:
at 256 organisms it reads 19 MB of weights per tick, about 13 µs of
bandwidth on this card, spread over 22 dispatches.

**fp16 weights are not a win at this size.** `--brain-fp16 1` halves the
weight bytes and changes nothing measurable: in real GPU time on CUDA it is
44.9 µs against 45.7 µs at `--founders 256`, inside run-to-run spread; on
wgpu's device timestamps it is 4.3–4.5% *slower* (149 µs against 143 at 256,
140 against 134 at 64), the f16-to-f32 conversion costing more than the
halved load saves. Accuracy against the plain-Rust reference goes from 3e-7
to 3e-4. All three runtimes have f16 buffer storage at this pin, so the path
works; fp32 is the default and `--brain-fp16` keeps it available for when
the trunk is wide enough to be bandwidth-bound.

Kernel against `brain::forward::forward_population_ref`, after 30 steps of a
real run, as a relative deviation against the largest value in the field
(`cargo test -p alife-sim --test brain`):

| weights | CPU (latents / heads) | CUDA | wgpu |
|---------|----------------------:|-----:|-----:|
| fp32    | 0 / 0 (exact) | 2.4e-7 / 1.5e-7 | 2.9e-7 / 2.3e-7 |
| fp16    | 2.0e-4 / 3.0e-4 | 2.0e-4 / 3.0e-4 | 2.0e-4 / 3.0e-4 |

Byte-for-byte fluid parity at `--founders 0` against the pre-brain build
(`a8cb8df`), whole dumps compared with `cmp`: CUDA and wgpu at 50 steps, the
CPU runtime at 20, terrain modes 0 and 1, all six identical.

`--founders 256 --iterations 500` on CUDA: no non-finite latent or head
anywhere, and the largest latent sits at 4.83 for the whole run. Before
`latent_norm` existed the same run passed 4e18 by step 40 and overflowed by
step 45 (`organism.md`, decisions).

### Energy and the life cycle (2026-09-19, commit after c5d9f00)

Two kernels and one packed readback every `life_interval` = 10 steps, plus
whatever that tick decides to grow, place, free or mutate.

```
cargo +1.98.1 build --release -j16
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 \
  ./target/release/alife --headless --runtime cuda --seed 42 \
  --iterations 3001 --founders <N> [--life-interval 100000]
```

Per-kernel averages over 3000 steps (the event profiler, which on CUDA is
wall time around a synchronise and so carries a ~11 µs floor per launch;
the two life kernels run 300 times, not 3000, and their `max` is the
first-launch JIT):

| kernel                       | `--founders 0` | 64, life off | 64, life on |
|------------------------------|---------------:|-------------:|------------:|
| `calculate_accel`            |          0.341 |            — |       0.345 |
| `calculate_evap_prob`        |          0.167 |            — |       0.167 |
| `calculate_particle_density` |          0.164 |            — |       0.165 |
| `project_constraints`        |              — |            — |       0.099 |
| `write_limb_geometry`        |              — |            — |       0.015 |
| brain, all 22 launches       |              — |            — |       0.276 |
| `light_grid` (per tick)      |              — |            — |       0.040 |
| `organism_energy` (per tick) |              — |            — |       0.015 |
| **profiler per step**        |      **0.814** |    **1.209** |   **1.378** |
| **wall clock per step**      |      **1.216** |    **1.970** |   **2.036** |

The profiler serialises every launch, so the wall-clock row is the honest
per-step figure and the profiler row is the sum of the serialised parts.
Between the two `--founders 64` runs — same startup, same JIT, one with
`--life-interval 100000` so the tick never fires — the whole life cycle
costs **0.066 ms per step**, about 3% of the step: 0.0055 ms of it is the
two kernels amortised over the interval, and the rest is the packed
readback, the free-slot scans on a tick that claims anything, and the limb
download on a tick with a birth. Note that the two runs do not simulate the
same world — the one with the life cycle on grows, seeds and kills bodies —
so this is the cost of having a life cycle, not of one fixed workload.

### The population, 3000 steps (2026-09-19, commit after c5d9f00)

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/alife --headless \
  --runtime <cuda|wgpu|cpu> --seed 42 --terrain-mode 0 --founders 64 \
  --iterations <3000|500> --metrics m.csv --dump d.bin
```

| runtime | steps | alive | births | deaths | lineages | generation max | species |
|---------|------:|------:|-------:|-------:|---------:|---------------:|--------:|
| cuda    |  3000 |    44 |     48 |     90 |       17 |              3 |       2 |
| wgpu    |  3000 |    35 |     46 |     93 |       17 |              1 |       1 |
| cpu     |   500 |    39 |      0 |     27 |       37 |              1 |       1 |
| cpu     |  1500 |    53 |     25 |     61 |       24 |              1 |       1 |

Started from 64 founders and 64 lineages, so a third of the lineages are
extinct by step 3000 and the population is well under the 256 organism
slots. The three runtimes diverge from each other — the brain's
transcendentals are one ulp apart across backends (`organism.md`,
decisions), and one sprout decision landing differently changes a body —
but each is **bit-identical to its own repeat**: two runs per runtime,
dumps compared with `cmp`, identical, and the metrics CSVs identical once
the wall-clock column is dropped. No field of any CSV is NaN or infinite.
The CPU runtime at 500 steps has no births yet because the first seed lands
around step 600; 1500 steps is where its cycle is visible.

`--founders 0` dumps are byte-identical to the pre-life-cycle build
(`c9bae4a`), whole files compared with `cmp`: CUDA and wgpu at 50 steps,
the CPU runtime at 20, terrain modes 0 and 1, all six identical.

The energy constants are `organism.md`'s decision entry; the run that
chose them is the 0.001-versus-0.003 pair of `upkeep_per_particle`, 3
deaths and every slot full against 78 deaths and 93 of 256 slots.

### The soil-specialization experiment (2026-09-19, commit after 4b19bcb)

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./scripts/soil_experiment.sh 60000
```

Eleven runs on CUDA at `--terrain-mode 2 --founders 60 --max-organisms 1024
--metrics-every 100`: nine of 60,000 steps and two of 15,000. Wall time per
run, as the script prints it:

| run | steps | s |
|---|---:|---:|
| `main_1` | 60,000 | 103 |
| `isolation_1` | 60,000 | 114 |
| `permuted_1` | 60,000 | 105 |
| `main_2` | 60,000 | 88 |
| `isolation_2` | 60,000 | 69 |
| `permuted_2` | 60,000 | 109 |
| `main_3` | 60,000 | 116 |
| `isolation_3` | 60,000 | 116 |
| `permuted_3` | 60,000 | 84 |
| `transplant_3` | 15,000 | 54 |
| `resident_3` | 15,000 | 54 |
| **whole script** | | **1012** |

1.4–1.9 ms a step, against the 2.0 ms of the `--founders 64
--max-organisms 256` life-cycle run above: four times the organism slots
costs well under twice the step, because the per-particle kernels are
launched over the claimed body slots rather than over the capacity and the
per-organism passes are a single cube either way. The spread between runs is
the population they carry — the two that went extinct early are the two
fastest — not anything about the configuration.

A saved population is `max_organisms x param_count` fp32 and nothing else of
consequence: **78 MB** at these settings, written in well under a second.

What the runs found is in `REVIEW.md`.
