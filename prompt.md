# Handoff — run the organism refactor end to end (2026-09-18)

You are a Fable session orchestrating a long, mostly autonomous build. You
do **not** write bulk code yourself: you settle what is unsettled, brief
builders, merge, verify their claims by rerunning them, review, fix, and
queue what only Jonah can judge. The pattern is *The delegation cycle* and
*Code review* in the root `CLAUDE.md`; follow them literally. Nobody will
read the chat as it happens, so every decision Jonah must make and every
result only he can see goes to `docs/REVIEW.md` in the same commit as the
work.

## Read first, in this order

1. Root `CLAUDE.md` — conventions, the delegation cycle, review workflow.
2. `docs/organism.md` — the settled design and its dated decisions. This
   is the spec. Do not re-open a decision listed there; if a builder
   discovers it cannot be built as written, fix the spec in the same
   commit and say so in `REVIEW.md`.
3. `docs/TODO.md` — the chunk list. *Substrate* is the current chunk; the
   *Organisms* section is the sequence after it.
4. `docs/perf.md` — toolchain traps (driver rejects 13.3 PTX; rustc 1.98.1
   named toolchain; `Cargo.lock` is authoritative) and the numbers to beat.
5. `docs/scratch_port_brief.md` — the exact brief the port builder was
   given; re-issue or resume it from there.

## State of `main` (commit `2cb23ab` plus this handoff)

- Session of 2026-09-18 settled the whole design in chat and wrote it
  into `docs/organism.md`: one organism representation for plants and
  creatures, fixed-shape genome, Perceiver-IO brain over limb tokens,
  particle bodies, ecology as the only selection, plants first, fluid
  solver swap on hold. Substrate: Rust + CubeCL, exact pin `0.11.0-pre.3`.
- The CubeCL spike (`crates/spike`) proved device/buffer sharing with
  rendering, the CPU runtime, CUDA and Vulkan on this box; numbers and
  caveats in `perf.md` and the `organism.md` decisions.
- The C++ tree is the **reference**, not a maintenance target. Two fluid
  fixes landed there so the port follows correct behavior (grid tiles the
  width exactly, viscosity kernel takes squared distance), plus `--dump`
  for headless runs and a fix for a latent out-of-bounds write in the
  vapor step. `docs/perf.md` has before/after numbers.
- **The C++ sim is not bit-reproducible run to run** (atomic grid insert
  order). Parity with it is therefore statistical: one-step diffs at the
  run-to-run floor, plus 50-step aggregates. Reference dumps are committed
  in `resources/parity/` (terrain mode 1, seed 42); the floor between
  `ref2a`/`ref2b` is `pos` 3.8e-6, `density` 3.6e-3.

## In flight when this was written

A port builder (Opus, worktree) was launched with the brief in
`docs/scratch_port_brief.md` and had **not committed yet** at handoff. It
will not survive the session. On start:

```
git worktree list
git branch --list 'worktree-agent-*'
```

If a `worktree-agent-*` branch has commits on top of `2cb23ab`, inspect
them (`git log main..<branch>`), fast-forward or rebase what builds and
passes its tests, then sweep the worktree. Then launch a fresh builder from
the brief with a note naming the last landed commit-plan step, so it
resumes rather than restarts. If there are no commits, launch the brief
as is.

## The plan: cycles until the plant milestone runs

Each cycle: brief → build in a worktree → merge → **rerun the builder's
claims yourself** (build, tests, parity, headless run, the numbers) →
`/code-review-find` over the range (Sonnet finders; stage the diff once in
the parent) → verify every candidate both ways → fix → build + headless →
push on a green tree (delegation cycle step 6; Jonah works across
machines, so the push at the end of each green cycle is how work moves).

1. **Port** (`scratch_port_brief.md`). Acceptance = the parity protocol on
   every runtime available on the machine, tests green, GUI builds. Then
   delete `crates/spike`, promote the Rust commands in `CLAUDE.md`, and
   record in `REVIEW.md`: "GUI needs a visual check: run `cd resources &&
   cargo +1.98.1 run --release -- ...`" with the exact command.
2. **Genome + population tensors** and **particle bodies** — independent,
   two worktrees in parallel (both headless). Bodies need the particle
   kind enum, organism id and limb index per particle, distance and
   base-joint angle constraints, root pinning. Genome needs the discrete
   limb records, the fixed-shape brain tensor as SoA population buffers,
   both mutation operators, species distance.
3. **Brain forward pass** (depends on both above). Tokens, persistent
   gated latents, cross-attention in/out, trunk, sprout head; one batched
   pass per tick; fp16 weights, fp32 accumulate; a plain-Rust reference
   forward for parity, since CPU kernels are not debuggable.
4. **Energy and life cycle** + **evolutionary metrics** (`--metrics
   <path>` time series + summary at exit). Can run in parallel worktrees;
   metrics needs only the population buffers.
5. **Soil-specialization experiment** with its three controls, as a
   headless script plus a `REVIEW.md` entry with the numbers and plots
   path. Jonah is skeptical soil alone drives a split; the controls exist
   to separate soil from spatial isolation and column position. Report
   what the numbers say, not what we hope.

Between cycles, re-read `docs/TODO.md` and delete what shipped.

## Forks left open, and the default to take

Take the default, say so in `REVIEW.md`, keep going. Do not block.

- **Trunk width, latent count, max limbs per organism, max particles per
  limb:** start small (trunk ~16k params, 8 latents of 32, 16 limbs, 8
  particles per limb) and make each a sim param so it is a flag.
- **Energy constants** (light gain per leaf particle, water per root
  particle, upkeep per particle, seed cost): pick values that let a single
  plant of ~10 particles break even in open sun with wet soil, then let
  selection tune from there. Record the derivation in a comment.
- **Light model:** per-column occlusion from the top, linear attenuation
  per leaf particle crossed. Nothing fancier until a run shows a need.
- **Seed germination:** a seed particle that comes to rest with a soil
  cell below it pins there; no dormancy.
- **Adaptive mutation distribution:** off. Do not build it in this run.
- **Anything about mobile creatures:** milestone 2, out of scope.

## Machine notes

Written on the dev box (16 cores, 76 GB, RTX 5090, CUDA 13.3 toolkit with
a 13.2 driver — `perf.md`). Builds: `cargo +1.98.1 build --release -j16`,
`LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64` for the CUDA runtime.

On a different machine: install the `1.98.1` toolchain
(`rustup toolchain install 1.98.1 --profile minimal --component rustfmt
clippy`); the first CPU-runtime build downloads a 36 MB LLVM bundle. On an
AMD or Intel box use `--runtime wgpu` (or `hip` if the builder wired it;
it is in the pin); the C++ reference will not build without CUDA, so parity
uses the committed `resources/parity/` dumps and the C++ aggregate column
stays "measured on the dev box". Add a section for the machine to
`perf.md` with a measured `-j` and the per-runtime kernel timings. Never
open a GUI window from an agent; Jonah checks visuals.

## Things this session learned (don't relearn)

- Builders' reports are usually right and occasionally load-bearing-wrong:
  the fix builder found the C++ non-reproducibility that invalidated the
  "identical density" acceptance test in its brief, and the spike found
  that `cargo add cubecl` does not resolve without a hand-pinned transitive
  dep. Rerun what they claim; read what they say the brief got wrong.
- `compute-sanitizer` did not catch a one-row out-of-bounds write (it
  landed inside the rounded-up allocation). The `--dump` flag plus a short
  script did.
- A design discussion that starts at "which metric" can turn out to be
  "which substrate". Zooming out once cost an hour and saved the project a
  rewrite of the rewrite. If a chunk feels like it is fighting the design,
  stop and check the decisions log before pushing through.
