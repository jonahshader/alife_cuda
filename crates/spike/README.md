# CubeCL feasibility spike

Throwaway. It exists to answer the four unknowns in `docs/TODO.md` §
*Substrate: Rust + CubeCL* and is not part of the port; delete it once the
answers are recorded in `docs/perf.md` and `docs/organism.md`.

One kernel — a port of `reset_particles_per_cell`, `populate_grid_indices` and
the soil-free `calculate_particle_density` from `src/systems/particle_fluid2.cu`
— runs unchanged on the CPU, CUDA and wgpu/Vulkan runtimes, on the NVIDIA
adapter and on llvmpipe. Every run is checked against a plain-Rust reference
(`src/reference.rs`) over the same fixed-seed input.

Problem size matches the headless baseline in `docs/perf.md`: 51,200 particles,
160×80 cells, smoothing radius 0.2, 32 slots per cell. Nothing opens a window.

## Running it

Requires **Rust ≥ 1.95** — see answer 0. Add `-j16` to the build; `docs/perf.md`
has the measured ceiling.

```
cargo +1.98.1 build --release -j16

# needs CUDA 13.2's nvrtc, see answer 3
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/spike all

# individually
./target/release/spike adapters            # enumerate Vulkan adapters
./target/release/spike cpu                 # cubecl-cpu (LLVM JIT)
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/spike cuda
./target/release/spike wgpu                # wgpu on a device WE created
./target/release/spike wgpu-cubecl-device  # wgpu on WgpuDevice::DiscreteGpu(0)
./target/release/spike wgpu-cpu-device     # wgpu on WgpuDevice::Cpu -> llvmpipe
./target/release/spike llvmpipe            # wgpu on an explicitly picked llvmpipe adapter
./target/release/spike share               # the device/buffer sharing proof
./target/release/spike gdb-probe           # one CPU launch, then a breakpoint target
```

## Answers

### 0. Rust version (not one of the four, but blocking)

**Rust 1.93.1 cannot build CubeCL 0.11.0-pre.3.** `cubecl-zspace` (pulled in by
`cubecl-macros`) declares `rust-version = 1.95`, and cargo refuses. Installed
1.98.1 alongside and used `cargo +1.98.1` throughout; the default toolchain was
left at 1.93.1.

### 1. Device and buffer sharing with rendering — **yes**

`spike share` creates a `wgpu::Instance`/`Adapter`/`Device`/`Queue` itself
(headless, `InstanceDescriptor::new_without_display_handle`), packs them into
`cubecl_wgpu::WgpuSetup` and calls `cubecl_wgpu::init_device(setup, opts)`,
which returns a `WgpuDevice::Existing(id)` whose client runs on *our* device.
`ComputeClient::get_resource(handle)` then hands back a `WgpuResource` with a
public `buffer: wgpu::Buffer` plus `offset`/`size`, and
`as_wgpu_bind_resource()` builds the `BindingResource` for it.

Observed output (RTX 5090, Vulkan), all three using only our own device, queue
and encoders, with no CubeCL readback:

```
CubeCL density handle -> wgpu::Buffer { size 204800 B, offset 870400 } (usages BufferUsages(COPY_SRC | COPY_DST | STORAGE | INDIRECT))
wgpu copy_buffer_to_buffer + map_async: max|diff| = 6.104e-5
our own compute pass reading it as storage: max|diff| = 1.221e-4
our own offscreen render pass, fragment reads src[7]: 127.60047 (want 127.60048)
```

The render-pass line is a real `begin_render_pass` into a 1×1 R32Float
attachment whose fragment shader reads the CubeCL buffer as
`var<storage, read>`; the texel comes back equal to the kernel's output. So a
renderer can draw straight from CubeCL's buffers with no host round trip.

Two things the port has to respect:

- **A handle is a slice of a pooled buffer, not a buffer.** The offset above is
  870400, not 0. Bindings must use `offset`/`size` (or
  `as_wgpu_bind_resource()`), never `as_entire_binding()`.
- **No `VERTEX` usage.** `cubecl-wgpu` allocates with
  `STORAGE | COPY_SRC | COPY_DST | INDIRECT`
  (`cubecl-wgpu/src/compute/mem_manager.rs`), so a CubeCL buffer *cannot* be
  bound with `set_vertex_buffer`. Read it as a storage buffer in the vertex
  shader and index by `@builtin(vertex_index)` instead — which is what the
  particle renderer wants anyway.

`egui-wgpu` was added as a dependency purely to check version unification: it
does not conflict. `cubecl-wgpu 0.11.0-pre.3` wants `wgpu ^30.0.0`,
`egui-wgpu 0.36.2` wants `wgpu ^30.0`, and the lock resolves a **single**
`wgpu 30.0.1`, which is the precondition for passing a `wgpu::Device` between
them. (`egui-wgpu 0.34` would have pulled a second, incompatible `wgpu 29`.)

**Adapter selection is possible both ways.** Through CubeCL's own API:
`WgpuDevice::{DiscreteGpu(i), IntegratedGpu(i), VirtualGpu(i), Cpu}` plus the
`CUBECL_WGPU_DEFAULT_DEVICE` environment variable, and `init_setup::<Vulkan>`
to pin the graphics API. For anything finer — picking by adapter name, or
reusing a renderer's adapter — build the `WgpuSetup` yourself and use
`init_device`, as `spike llvmpipe` does.

### 2. CPU runtime build requirements — **no system LLVM at all**

`cubecl-cpu` → `cubecl-llvm` → `llvm-sys 221.1.0` built with
`no-llvm-linking` + `disable-alltargets-init`, so `llvm-config` is never
consulted. The build dependency `tracel-llvm-bundler 22.1.4-6` downloads a
prebuilt, self-contained **LLVM 22.1.4** and emits the link configuration.

- Downloaded archive: `~/.cache/tracel/tracel-llvm-22.1.4-6-linux-x64.tar.xz`,
  **36 MB**; unpacked to `~/.local/share/tracel/tracel-llvm-22.1.4-6`,
  **241 MB**. Both paths are derived from `$HOME`, not from XDG variables.
- **No `LLVM_SYS_*_PREFIX`, no `llvm-config`, no distro LLVM package.** This
  box's `/usr/lib/llvm-20`, `/usr/lib/llvm-21` and the `llvm-config` 21.1.8 on
  `PATH` are all ignored.
- The only environment variable that matters is `CUBECL_DEBUG_PLIRON` — set at
  *build* time it enables the `pliron-dump` feature, and at run time it names a
  directory for the IR dumps (see answer 4).
- Clean `cargo build --release -j16` of the whole workspace (all three runtimes
  + wgpu + egui-wgpu), with the LLVM bundle already cached: **1 min 32 s**.
  A cold machine adds the 36 MB download. Resulting binary: **109 MB**, LLVM
  statically linked.

**A laptop with only a distro LLVM package builds this fine** — better than
fine, since the distro package is irrelevant. What it needs is ~300 MB of disk
and a network fetch on first build. There is no source build of LLVM.

**But there is a dependency-resolution bug that has to be pinned around.**
`cubecl-llvm` requires `pliron 0.17` and `pliron-llvm 0.17`; `pliron 0.17.0`
declares `pliron-derive = "0"`, so cargo picks `pliron-derive 0.18.0`, and
`pliron 0.17.0` then fails to compile with ~55 errors
(`cannot find trait TypeImplsInterface in module ::pliron::type`, etc.).
`pliron-llvm 0.17.0` separately declares `pliron = "0"` and drags in a second
`pliron 0.18.0`. Fixed in the committed `Cargo.lock` by:

```
cargo update pliron@0.18.0 --precise 0.17.0
cargo update pliron-derive --precise 0.17.0
```

A fresh `cargo generate-lockfile` reintroduces the breakage, which is why
`Cargo.lock` is committed.

### 3. CUDA runtime on this box — **yes, with one environment variable**

Out of the box it fails:

```
Unable to load the PTX: DriverError(CUDA_ERROR_UNSUPPORTED_PTX_VERSION,
"the provided PTX was compiled with an unsupported toolchain.")
```

`cubecl-cuda` compiles with nvrtc (`--gpu-architecture=sm_120a`) and hands the
resulting **PTX** to the driver to JIT. `/usr/local/cuda` is 13.3 but the
driver (595.91.07) reports `CUDA Version: 13.2`, so nvrtc 13.3's PTX ISA is
newer than the driver accepts. Pointing cudarc's dynamic loader at the 13.2
toolkit, which is also installed, fixes it:

```
LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/spike cuda
```

`CUDA_PATH` is not needed (cudarc uses `fallback-dynamic-loading` and resolves
`libnvrtc.so.13` through the loader path). No other variable was required.

This is not a problem the C++ tree has: nvcc AOT-compiles to SASS for the
native arch, so nothing is JIT'd from PTX. The port either keeps the toolkit
and driver in step or ships a pinned nvrtc.

### 4. Debuggability on the CPU runtime — **no**

`cubecl-llvm` JITs with `LLVMLLJIT::new_with_default_builder()` and installs no
ORC debugger-support plugin, and the pliron → LLVM IR conversion emits no debug
metadata at all (`grep -c '!dbg\|DILocation\|DISubprogram' llvm.ll` → 0).

Measured, not inferred. `spike gdb-probe` runs one launch on the CPU runtime
and then calls `spike_after_jit()`, so the breakpoint fires after every kernel
has been compiled *and executed*:

```
$ gdb -batch -ex "set breakpoint pending on" \
      -ex "break calculate_particle_density" -ex "break spike_after_jit" \
      -ex run -ex "info breakpoints" -ex "maintenance info jit" \
      --args ./target/release/spike gdb-probe
Function "calculate_particle_density" not defined.
Breakpoint 1 (calculate_particle_density) pending.
...
Thread 1 "spike" hit Breakpoint 2, 0x... in spike_after_jit ()
Num     Type           Disp Enb Address            What
1       breakpoint     keep y   <PENDING>          calculate_particle_density
```

`maintenance info jit` prints nothing: GDB's JIT registration interface has no
entries. The kernel is anonymous executable memory — no symbol, no line table,
no source-level stepping, no variables.

What does exist is IR-level inspection. Build with `CUBECL_DEBUG_PLIRON` set,
then set it to a directory at run time:

```
CUBECL_DEBUG_PLIRON=1 cargo +1.98.1 build --release -j16
CUBECL_DEBUG_PLIRON=/tmp/dump ./target/release/spike gdb-probe
```

writes `/tmp/dump/<kernel_name>/` with 19 `N-after-<pass>.plir` snapshots plus
`llvm.ll` and `llvm.opt.ll`. Diffing consecutive passes is the supported way to
localise a miscompile. For debugging *kernel logic* rather than the compiler,
the practical answer stays a plain-Rust reference implementation like
`src/reference.rs` — which is what the determinism test in `docs/TODO.md` is
anyway.

## Results

`LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64 ./target/release/spike all`, RTX
5090 / 16 cores, release build. "1 launch" and "100 launches" are wall time
around the three kernels plus a `client.sync()`, after a warm-up launch.

| Runtime | 1 launch | 100 launches | per launch | max abs diff (density / near) |
|---|---:|---:|---:|---|
| `cubecl-cpu` (LLVM JIT) | 1.671 ms | 223.9 ms | 2.239 ms | 6.104e-5 / 1.831e-4 |
| `cubecl-cuda` | 0.099 ms | 6.78 ms | 0.068 ms | 9.155e-5 / 1.831e-4 |
| wgpu/Vulkan, our device | 0.533 ms | 7.12 ms | 0.071 ms | 6.104e-5 / 1.831e-4 |
| wgpu/Vulkan, `WgpuDevice::DiscreteGpu(0)` | 0.446 ms | 7.90 ms | 0.079 ms | 6.104e-5 / 1.831e-4 |
| wgpu/Vulkan, `WgpuDevice::Cpu` (llvmpipe) | 4.658 ms | 304.1 ms | 3.041 ms | 4.578e-5 / 1.831e-4 |
| wgpu/Vulkan, llvmpipe picked by name | 2.973 ms | 295.0 ms | 2.950 ms | 6.104e-5 / 1.831e-4 |

Per-cell particle counts and per-cell particle-id *sets* are bit-exact against
the reference on every runtime; only the float sums differ, and only because
the atomic grid build orders a cell's slots differently than the sequential
reference does. Relative error is ~4e-7 against densities of order 250.

Single-threaded plain-Rust reference, same input: 12.6 ms. The CPU runtime
spawns ~17 worker threads and beats it by 5.6x.

First-launch cost including JIT, on a **cold** compilation cache: CUDA 354 ms,
wgpu 45 ms, CPU 48 ms. The cache is persistent, so repeat runs show 7–75 ms.

## Resolved versions (from `Cargo.lock`)

| Crate | Version |
|---|---|
| `cubecl`, `cubecl-core`, `cubecl-cpu`, `cubecl-cuda`, `cubecl-wgpu`, `cubecl-runtime`, `cubecl-llvm`, `cubecl-macros`, `cubecl-ir`, `cubecl-opt`, `cubecl-std`, `cubecl-cpp`, `cubecl-common`, `cubecl-environment`, `cubecl-hip`, `cubecl-macros-internal`, `cubecl-zspace` | `0.11.0-pre.3` |
| `cubecl-hip-sys` | `7.14.6085001` |
| `wgpu` | `30.0.1` (single version in the graph) |
| `naga` | `30.0.1` |
| `egui-wgpu` | `0.36.2` |
| `cudarc` | `0.19.9` |
| `llvm-sys` | `221.1.0` |
| `tracel-llvm-bundler` | `22.1.4-6` (bundles LLVM 22.1.4) |
| `pliron`, `pliron-derive`, `pliron-llvm` | `0.17.0` (pinned by hand, see answer 2) |

Toolchain: rustc 1.98.1, CUDA 13.3 toolkit with 13.2 nvrtc used at run time,
driver 595.91.07, Mesa llvmpipe LLVM 21.1.8.
