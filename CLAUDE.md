# CLAUDE.md

Artificial-life simulation: an SPH particle fluid with an
evaporation/condensation cycle, a soil grid with porosity and capillary
action, L-system trees, and (planned) creatures driven by evolved policy
networks. The sibling repo `alife_cuda_2` is the source of several ports
still listed in `docs/TODO.md`.

The fluid and soil run on **Rust + CubeCL** (`crates/`), on CPU, CUDA or
wgpu. The C++/CUDA tree (`src/`) is the reference the port is checked
against and still owns the L-system trees; it is not where new sim work
goes. See *Build, test & run*.

**What belongs here:** cross-cutting conventions and non-obvious invariants a
session needs *before* editing. Subsystem APIs live in the owning header (the
source of truth) or in `docs/`; **link to them, don't restate them here** — a
copy in this file just goes stale. See `docs/CLAUDE.md` for the doc index +
maintenance contract (which doc to update for which kind of change, including
how `TODO.md` is kept current).

## Conventions

Keep **simulation state** and **rendering** separate. `crates/alife-sim`
holds the state, the kernels and the step function and knows nothing about
windows; `crates/alife` owns the CLI, the headless loop and the egui window.
Headless mode builds the state and steps it without ever creating a window —
keep it that way. (The C++ tree draws the same line between `src/systems/`,
`src/graphics/` and `src/screens/`.)

Structure-of-arrays types come from one field list: `define_soa!` in
`crates/alife-sim/src/soa.rs` (and `DEFINE_STRUCTS` in
`src/systems/soa_helper.h` on the C++ side). One declaration generates the
host struct, the device buffers and the dump order. Add a field by adding a
line, never by hand-editing a generated struct — and add it in both trees
while both exist, or a dump stops loading across them.

Simulation parameters are one declaration each in
`crates/alife-sim/src/params.rs` (and `src/config/sim_params.h`) — that
single line produces the struct field, the TOML key, the CLI flag, the help
text and the default-config writer. Precedence is CLI > TOML > compiled
default.

Every CubeCL kernel has a plain-Rust reference next to it and a test that
runs both over one fixed input. That is not belt and braces: the CPU
runtime's LLVM JIT emits no symbols, so a debugger never sees a kernel, and
the reference is the only way to find out what a kernel actually computed.

Naming is snake_case for functions/variables/files, PascalCase for types
(`.clang-tidy` enforces it on the C++ side, `rustfmt`/`clippy` on the Rust).
Comments explain *why*, not what; skip them when names and structure already
say it. **Never silently drop existing comments when refactoring** — carry
them to the new location or say why they went.

Docs follow the same discipline: state only what's true and load-bearing now
— no speculation, no architecture walkthroughs that bitrot. (Full contract:
`docs/CLAUDE.md`.)

Long orchestrated runs (overnight autonomy, agent chains) queue their
must-see outcomes in **`docs/REVIEW.md`** — the user's review queue — in the
same commit as the work. It substitutes for a conversation nobody will read
at the time; in interactive sessions, say it in the chat instead.

If you spot a pre-existing issue while working on something else (latent bug,
dead code, broken invariant): when the fix is obvious and low-risk, implement
it as its own commit **and** tell the user. When it is non-obvious, risky, or
a judgment call, surface it unfixed. Never silently fix, never silently
ignore.

## Version control

- **Commit directly to `main`.** Solo repo, no PR flow; feature branches add
  ceremony without payoff. This deliberately overrides the harness default of
  branching first. Use a throwaway branch only for a speculative spike.
- Don't `git push` unless asked.
- **Atomic commits by default** — one self-contained logical change per
  commit. When two agents share one tree, **commit by explicit path only**
  (`git commit <files>`, never `-a` / `add -A`).
- When the previous commit is unpushed and the new change is logically part
  of it, prefer `git commit --amend` over a "fix previous commit" commit.
  Never amend pushed commits.
- `docs/scratch_*.md` and `prompt*.md` are **tracked** but remain the
  ephemeral working tier — per-task memory, prunable, not a source of truth.
  `prompt.md` is the next session's handoff brief; rewrite it per handoff
  rather than accreting. Durable content migrates to the tracked docs (the
  *Two tiers* section in `docs/CLAUDE.md`).

## Build, test & run

The fluid and soil sim is **Rust + CubeCL** (`crates/alife-sim`, the
`alife` binary). The C++/CUDA tree is now **reference only**: it stays
buildable so the port can be checked against it, and it is what the
L-system trees still live in. New sim work goes in the Rust tree.

- Build: `cargo +1.98.1 build --release -j16`. The pinned CubeCL needs a
  newer rustc than the default stable, hence the named toolchain;
  `Cargo.lock` is authoritative, **don't regenerate it** (`cargo add` is
  fine). Before committing: `cargo +1.98.1 fmt` and `cargo +1.98.1 clippy
  --all-targets`, both clean.
- **The CUDA runtime needs `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64`** on
  the dev box (`docs/perf.md` says why). Without it `--runtime cuda` is not
  offered and the default falls through to wgpu.
- **Headless is the agent's default surface**: `./target/release/alife
  --headless --iterations N` steps the sim without a window and prints
  per-kernel timings on exit. `--runtime cpu|cuda|wgpu` picks the backend
  (default: CUDA if available, else wgpu, else CPU); `--dump <path>` writes
  the particle SoA and `--load <path>` resumes from one — version 2 of the
  C++ `--dump` format, and it still reads version 1
  (`crates/alife-sim/README.md`). `--founders N` seeds N plants along the
  soil surface; the default 0 leaves the world exactly as it was before
  organisms existed. `--metrics <path>` writes the evolutionary time series
  as CSV every `--metrics-every K` steps (default 100) and prints a summary
  at exit (`docs/organism.md`, *Metrics*). `config.toml` is read from the
  CWD and `--write-config` emits the defaults; every sim param is also a CLI flag
  (`--help` lists them; `crates/alife-sim/src/params.rs` is the
  authoritative list).
- **Same-seed runs are bit-reproducible** on a given backend, unlike the
  C++, so a change to sim code can be checked by diffing dumps.
  `cargo +1.98.1 test` runs each kernel against its plain-Rust reference and,
  when the C++ reference dumps are present, against the C++ binary itself
  (`crates/alife-sim/README.md` regenerates them).
- **Never leave a GUI run open on the user's desktop.** The app has no frame
  cap flag yet, so an uncapped window pegs the GPU. An agent starts a GUI run
  only when the user asked for one, under `timeout`, and says so. What an
  agent *can* check without a window is that the shaders and pipelines
  validate, which `cargo test -p alife` does.

The C++ reference tree:

- Configure once: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`. Build:
  `cmake --build build -j<N>` with the **measured** per-machine job count
  from `docs/perf.md`. Before committing: `cmake --build build --target
  format`.
- `CMAKE_CUDA_ARCHITECTURES` defaults to `native` (the local GPU). Pass it
  explicitly for a cross-machine binary; the generated `.clangd` follows
  whichever arch was resolved.
- **Run its GUI from `resources/` as CWD** (`cd resources &&
  ../build/alife_cuda`) — shaders load by relative path.
- `./build/alife_cuda --headless --iterations N --dump <path>` is how the
  parity references are made. It is not bit-reproducible run to run (atomic
  grid insertion order, and `calculate_accel` races on `sph.vel`), so
  compare aggregates or compare against its own run-to-run spread.

## Working with Claude

- **Auto-memory is machine-local scratch, not a project record.** Anything
  about the project — state, decisions, dead ends, conventions, preferences —
  goes in the tracked docs (`docs/CLAUDE.md` routes it) or this file, in the
  same commit as the work. Save a memory only for facts about *this machine*.
- **Design discussions**: work through forks one question at a time with a
  short recommendation — not multi-question walls or option dumps.
- **Build the correct design, not the smallest reading.** When the right
  shape is clear — the rule that generalises both cases, the invariant that
  holds everywhere — build that, even when it is more work. Queue a
  `REVIEW.md` item only for a genuine fork, never as a way to ship the
  shortcut and ask later.
- **Doc edits**: use Read + Edit (visible diffs), never sed/python-heredoc
  text manipulation on files the user reads.
- **Forks** inherit full context + all tools and autonomously continue
  whatever they read. Never spawn one to "poke" another agent; wait for its
  notification.
- **Interrupted background agents resume losslessly**: `SendMessage` the
  agent id a short "resume from your brief" note. A killed **Workflow** run
  resumes with `resumeFromRunId`. Plan long runs assuming an interruption.
- **Agent worktrees** fork from the local HEAD because
  `.claude/settings.json` pins `worktree.baseRef: "head"` — **don't remove
  that key**; the harness default forks from the remote tip and agents
  rediscover unpushed work. Brief agents to run builds in the **foreground**
  with a long timeout (600000 ms); an agent that backgrounds a build may end
  its turn "waiting" with nothing done. Merge a worktree branch by rebasing
  onto main then fast-forwarding; sweep worktrees at session end.

### The delegation cycle

How feature work ships once a design is settled.

1. **Settle the design in chat first** — one fork at a time. Agents build;
   they don't decide.
2. **Delegate one contained feature per agent**, with the settled design as
   the brief. Opus or lower for builders, Sonnet for read-only sweeps
   (`Explore`); the session model never does bulk agent work. Headless work
   goes to a worktree; anything that needs the GPU *and* a window runs in
   the main tree. The brief names: read the root `CLAUDE.md` first; the spec
   pointer ("build from it, don't re-derive"); the environment block
   (headless command, `-j` cap, what to clean up); atomic commits by path,
   format target before each, no push; and the **report shape** — commits,
   decisions the brief left open, verification (commands run, numbers
   observed), pre-existing issues found.
3. **Merge** worktree branches as they land.
4. **Review the chunk** with `/code-review-find` over the range, then verify
   every candidate in context both ways — and **don't trust the builder's
   report either**: rerun what it claims green.
5. **Address findings and commit.** Pre-existing breakage is its own commit.
6. **Push** only at the end of a cycle, on a green tree.

### Code review

- The `/code-review` skill at **high/xhigh/max** inherits the session model
  for its fan-out. Run the **Sonnet-pinned copy** instead:
  `Workflow({scriptPath: ".claude/workflows/code-review.js", args: "<level> <target>"})`.
- `/code-review-find` (`.claude/workflows/code-review-find.js`) is the light
  find-only variant. **Stage once in the parent** — the JS sandbox can't run
  git — and pass `git diff <range>` as `args.diff` plus `git log <range>` as
  `args.commits`.
- Inline (low/medium) reviews: finders are **read-only `Explore` agents on
  Sonnet**, never forks (forks race edits on the shared tree).
- Finder claims need in-context verification **both ways**: fabricated claims
  and missed real bugs both happen. A clean report is not clearance.
- **Default action — review *and fix*.** After verifying candidates, apply
  the fixes worth applying, then build and run headless. Surface, don't
  silently apply, anything behavior-changing or genuinely ambiguous. Report
  what was changed vs skipped.
