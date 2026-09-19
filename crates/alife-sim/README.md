# alife-sim

The simulation: state, CubeCL kernels and their plain-Rust references, and the
step function. No windowing — that is `crates/alife`.

Ported from the C++/CUDA tree's soil-coupled `update_fluid(state, soil)` in
`src/systems/particle_fluid2.cu` and from `src/systems/soil.cu`. Those stay as
the reference until the port is trusted.

## Running the tests

```
cargo +1.98.1 test -p alife-sim --release -j16
```

The unit tests run each kernel against its plain-Rust twin over one fixed small
world, on the CPU runtime. `tests/parity.rs` additionally checks this port
against the C++ binary, and checks that a same-seed run is bit-reproducible on
every backend the box has. The CUDA backend needs
`LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64` on the dev box (`docs/perf.md`
says why); without it, `available(Cuda)` is false and that backend is skipped.

## Regenerating the parity references

`tests/parity.rs` compares one step of this port against the C++ binary, and
measures how much the C++ differs from itself over the same two steps. That
floor is the bar. The dumps are checked in under `resources/parity/` (2.5 MB
each, written by the C++ binary at commit `2cb23ab`) so a machine without CUDA
can still run the check; the test skips when they are absent.

To regenerate them, from the workspace root with the C++ tree built
(`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j16`):

```
./build/alife_cuda --headless --terrain-mode 1 --seed 42 --iterations 1 \
  --dump resources/parity/ref1.bin
./build/alife_cuda --headless --terrain-mode 1 --seed 42 --iterations 2 \
  --dump resources/parity/ref2a.bin
./build/alife_cuda --headless --terrain-mode 1 --seed 42 --iterations 2 \
  --dump resources/parity/ref2b.bin
./build/alife_cuda --headless --terrain-mode 1 --seed 42 --iterations 2 \
  --dump resources/parity/ref2c.bin
```

`ref1.bin` is the starting state; the `ref2*.bin` runs are three samples of the
same two steps, and their spread is the floor. Two are enough; the third makes
the floor estimate less jumpy and the test uses it when it is there.

`--terrain-mode 1` is the deterministic capillary-test terrain. Mode 0 works
too — the noise terrain reproduces the C++ bit for bit, which
`soil::tests::noise_terrain_matches_the_cpp` pins — but mode 1 keeps the
reference independent of the noise port.

The C++ dump carries the world's step count, and `--load` resumes the RNG
counter from it. A dump loaded without that would replay the random stream from
zero and evaporate a different set of particles.

A population travels in its own file instead: `--save-pop` / `--load-pop`,
documented above the writer in `src/popdump.rs`. It carries the genomes, the
lineage fields, the energies, the latent state and the anchors, and a load
re-grows every body from its genome and drops the seeds that were in flight.

Those references are **version 1** dumps, the C++ SoA at 50 bytes a particle.
This tree writes version 2, which appends the four organism fields, and reads
both — a version-1 dump loads with the organism fields at their defaults
(`dump.rs`). `--load` keeps a dump's non-body particles as the fluid and
reserves the body capacity on top of them, so a dump's organisms are not
resumed: the population tensors are not in a dump.

## Why every kernel has a reference

CubeCL kernels are not debuggable. The CPU runtime's LLVM JIT emits no symbols
and no line tables, so a debugger never sees the kernel — measured, not
inferred; see the spike outcome in `docs/organism.md`. The plain-Rust twin next
to each kernel, plus a test that runs both over one input, is the debugging
path. The other half is `CUBECL_DEBUG_PLIRON`, which dumps the IR after each
compiler pass.
