# Handoff — retiring the C++ tree, paused mid-cycle (2026-10-10)

Jonah paused the session overnight with two builders stopped mid-work. Pick
the cycle back up; the design below is settled.

## The plan (settled in chat 2026-10-09)

1. Port what the C++ GUI has that Rust lacks — two builders, in parallel:
   - **A: mouse tool + density overlay.** LMB attract / RMB repel at the
     cursor (`attract_fluid` in `src/systems/particle_fluid2.cu`), keys
     `+`/`-` radius ×1.25/×0.8, `[`/`]` strength ×0.8/×1.25, start radius
     2.0 strength 0.15, cursor circle + "Strength: N" label. **Pan moves to
     middle-mouse drag.** Density-grid overlay (`calculate_density_grid_kernel`)
     behind a View checkbox, keeping red = cell over `max_particles_per_cell`,
     blue = density over 300. Both kernels get plain-Rust references + tests;
     headless `--founders 0` dumps must stay byte-identical.
   - **B: live profiler panel** (timing on only while open, rolling window),
     fps/frame-time readout, Esc quits, anti-aliased particle discs.
2. Merge, rerun the claims, `code-review-find` over the range, fix.
3. **Delete the C++** — `src/`, `CMakeLists.txt`, the C++-only files under
   `resources/` (verify each is unused by Rust first), `tests/parity.rs` and
   `resources/parity/` (Jonah: parity tests go with the C++; don't keep them
   frozen). Before deleting, migrate the knowledge only the C++ holds:
   the pressure-equation "mostly describes gas, not liquid" weakness →
   `TODO.md` fluid section; the tree torque constants (thickness⁴ stiffness,
   damping 0.96, 10 rad/s cap, I = mL²/3) → `organism.md` decisions, for
   actuated limbs; the hard one-hot soil alternative and the friction
   constants' meaning → `soil.rs` comments. Then rewrite every doc that
   names the C++ build (root `CLAUDE.md`, `docs/CLAUDE.md`, `perf.md`,
   `crates/alife-sim/README.md`, `TODO.md`'s C++ sections). The trees never
   ran in the C++ binary (their screen is commented out), so nothing live
   is lost with them. The tag `cpp-reference` (local, on `f56f9de`) keeps
   the tree recoverable.
4. Then Jonah's behavior reports from his GUI run (`--founders 20
   --max-organisms 10000`): growth is exponential until the cap; some
   plants spin wildly; plants clump where they start and spread very
   slowly. Not yet investigated.

## State when paused

- Builder B, worktree `.claude/worktrees/agent-a8d34101b5a78f889`: five
  commits on its branch (profiler, fps, Esc, AA discs, and "a free body
  slot's quad covers no pixel" — a pre-existing fix, check it). It had not
  reported, so its verification is unconfirmed: rerun build, tests, clippy.
- Builder A, worktree `.claude/worktrees/agent-abd54b71a60d5bd5d`: no
  commits; uncommitted `kernels/attract.rs`, `kernels/density_grid.rs`,
  edits to `kernels/mod.rs` and `sim.rs`. GUI side not started.
- Same session: resume each builder with SendMessage ("resume from your
  brief"). New session: brief fresh builders from step 1 into the existing
  worktrees, telling them what's already there.
- `main` is 25 commits ahead of `origin/main`, unpushed.
