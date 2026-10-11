# alife-sim

The simulation: state, CubeCL kernels and their plain-Rust references, and the
step function. No windowing — that is `crates/alife`.

Ported from the C++/CUDA tree's soil-coupled `update_fluid(state, soil)` in
`src/systems/particle_fluid2.cu` and from `src/systems/soil.cu`. That tree is
deleted; the git tag `cpp-reference` keeps it. The port was checked against
the C++ binary before it went, and `docs/perf.md` records how closely it
agreed.

## Running the tests

```
cargo test -p alife-sim --release -j16
```

The unit tests run each kernel against its plain-Rust twin over one fixed small
world, on the CPU runtime. `tests/reproducibility.rs` checks that a same-seed
run is bit-reproducible on every backend the box has, and the other files in
`tests/` do the same for organisms. When the CUDA driver cannot take the
toolkit's PTX (`docs/perf.md`), `available(Cuda)` is false and that backend
is silently skipped — check the test output names it.

The noise terrain reproduces the C++ bit for bit, and
`soil::tests::noise_terrain_matches_the_cpp` pins it against values a probe
printed from the C++ tree's own headers.

## Dumps

A dump carries the world's step count, and `--load` resumes the RNG
counter from it. A dump loaded without that would replay the random stream from
zero and evaporate a different set of particles.

The C++ binary wrote **version 1** dumps, its SoA at 50 bytes a particle.
This tree writes version 2, which appends the four organism fields, and reads
both — a version-1 dump loads with the organism fields at their defaults
(`dump.rs`). `--load` keeps a dump's non-body particles as the fluid and
reserves the body capacity on top of them, so a dump's organisms are not
resumed: the population tensors are not in a dump.

A population travels in its own file instead: `--save-pop` / `--load-pop`,
documented above the writer in `src/popdump.rs`. It carries the genomes, the
lineage fields, the energies, the latent state and the anchors, and a load
re-grows every body from its genome and drops the seeds that were in flight.

## Why every kernel has a reference

CubeCL kernels are not debuggable. The CPU runtime's LLVM JIT emits no symbols
and no line tables, so a debugger never sees the kernel — measured, not
inferred; see the spike outcome in `docs/organism.md`. The plain-Rust twin next
to each kernel, plus a test that runs both over one input, is the debugging
path. The other half is `CUBECL_DEBUG_PLIRON`, which dumps the IR after each
compiler pass.
