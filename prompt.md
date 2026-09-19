# Handoff — milestone 1 is built and measured; the next move is Jonah's (2026-09-19)

The overnight run of 2026-09-18/19 executed the plan in the previous
`prompt.md` end to end. Read `docs/REVIEW.md` first: it is the queue of
things only Jonah can judge, and the experiment's result is the top item.
Then `docs/organism.md` (status line and the dated decisions, which grew a
lot), `docs/TODO.md` (the left-open lists per chunk), `docs/perf.md`.

## What shipped (all on `main`, pushed through `ba38080`'s predecessor; the
## last two commits are local until the next push)

Seven delegation cycles, each: Opus builder in a worktree → merge →
claims rerun by the orchestrator → Sonnet find-only review → verified
fixes → push on green.

1. **Port** of the fluid and soil to Rust + CubeCL at parity with the C++
   (`resources/parity/`), bit-reproducible per runtime, plus a cleanup
   pass (one neighbour traversal, no double device init, RNG counter as
   scalars, GUI bind-group reuse).
2. **Genome**: `BrainShape` slices (18,890 params), population SoA,
   founder plant, mutation kernels with references, deterministic slot
   scan, species distance.
3. **Bodies**: `Body`/`Free` particle kinds, constraint pass, limb
   geometry, `--founders`, dump v2. Fluid byte-identical at zero founders.
4. **Metrics**: `--metrics <path>` CSV, per-column traits, species
   clustering, column extents from the capillary layout.
5. **Brain**: 22-launch Perceiver-IO tick with a plain-Rust twin, tolerance
   1e-5 on all runtimes; fp16 storage built and off. The spec's tick
   diverged; a parameter-free RMS scaling of the latent state fixed it.
6. **Life cycle**: light occlusion grid, energy budget, sprouting by the
   brain's timing, seeds, germination, death to organic matter. A 64-founder
   run turns over on its own (48 births, 90 deaths, generation 3 by step
   3000) and is bit-reproducible.
7. **Experiment**: terrain mode 2 with headroom, permutation and
   uniform-soil controls, population save/load, transplant, the runner and
   scorer scripts, and one full run. **Result: no specialization visible**,
   and the world has a knife-edge (`--column-top` 0.25 lives, 0.30 dies;
   populations either saturate all slots or go extinct). Habitability does
   follow the soil: the sand column never holds a plant.

133 tests, clippy clean, per step ~1.4 ms at 64 founders on CUDA.

## Decisions the orchestrator took without Jonah (queued in `REVIEW.md`)

- Brain latents get an RMS normalization outside the parameter table.
- The sprout head decides *when* a genome's limb records grow, not what
  they are (keeps the genome heritable; avoids Lamarckian growth).
- `contact` sensor = solid fraction > 0.5, which puts pure clay on the edge.
- Mode 1 frozen for parity; the experiment lives in mode 2.
- Four energy/terrain constants moved on measurement, each with its runs.
- Pushed at the end of each green cycle (delegation cycle step 6).

## What the next session should NOT do on its own

Do not tune the energy balance or start milestone 2. The experiment's
outcome is a design question — widen the survivable band (depth-dependent
root return, more valuable light, a wetter world), or accept that plants
this simple cannot specialize and move to mobile creatures — and it is
Jonah's call. Everything an agent can do without that call is done.

## What it CAN do

- Close the review queue's small items if Jonah has answered them.
- The GUI has still never been opened by anyone; Jonah checks it.
- Known-good follow-ups that need no design decision, all in `TODO.md`:
  per-column split by lineage of origin in the metrics, `--load-pop`
  resuming the fluid too, `read_free_slots` prefix read, `narrow_shadow`
  per row, `pick_sprout` O(limbs²), the CUDA-unusable exit segfault probe.

## Machine and process notes

- Dev box as before: `cargo +1.98.1`, `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`
  for the CUDA runtime (still required; re-verified), `Cargo.lock` is
  authoritative.
- The `code-review-find` workflow hung three times out of seven at the
  post-finder stage with one finder never returning; the candidates are
  recoverable from `journal.jsonl` under the workflow's transcript dir,
  which is how those reviews were closed. Stop the hung task with
  `TaskStop` afterwards.
- The session hit the usage limit once around 02:50; two builders died at
  startup and were relaunched with the same briefs after the reset.
- Scratch dumps used for byte-for-byte checks live in the session
  scratchpad, not the repo.
