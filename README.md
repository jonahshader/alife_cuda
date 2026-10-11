# ALife CUDA

Artificial-life simulation: an SPH particle fluid with an
evaporation/condensation cycle, a soil grid with porosity and capillary
action, and plants driven by evolved policy networks. Written in Rust with
CubeCL, so the same kernels run on the CPU, on CUDA, or on any GPU wgpu
reaches.

## Building

Needs rustc 1.95 or newer. `Cargo.lock` is committed and pins the CubeCL
pre-release this builds against.

```bash
git clone https://github.com/jonahshader/alife_cuda.git
cd alife_cuda
cargo build --release
```

The CUDA backend JITs from PTX at runtime, so it needs a driver at least as
new as the CUDA toolkit; without one the binary falls back to wgpu.

## Running

```bash
./target/release/alife                                # the window
./target/release/alife --headless --iterations 1000  # no window
./target/release/alife --help                         # every flag
```

`--runtime cpu|cuda|wgpu` picks the backend; `--founders N` seeds N plants.
Every simulation parameter is also a `config.toml` key, read from the
working directory; `--write-config` writes the defaults.

The original C++/CUDA version is in the history before its deletion (tag
`cpp-reference`).

## License

Add license details here.
