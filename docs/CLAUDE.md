# `docs/` — index + maintenance contract

Every file under `docs/` has a specific job. **Keep them current as you
change the code they describe.** When a doc's claim diverges from the
implementation, update the doc in the same commit — a stale design doc is
worse than no design doc.

## Two tiers: tracked docs vs scratch working notes

The files in the Index below are the **tracked, indexed** doc set — kept
current, the source of truth for what shipped. Alongside them may live
**`docs/scratch_*.md`**, the *working tier*: per-task / multi-session design
memory, running backlogs, and decision logs too speculative or in-flux to
promote. These are also tracked (so they survive across machines) but are
**not** indexed or authoritative — working notes, prunable at any time.

Scratch docs are **temporary by design**. As a design lands, its durable
parts migrate *out* into the tracked docs or into the code + its comments,
and the scratch text covering shipped work is trimmed or the file deleted. A
scratch doc that still describes shipped behavior is itself a staleness
smell. Each scratch file's header names where its durable content goes.
(`prompt*.md` at the repo root is the same working tier for a session's
handoff brief — rewritten per handoff, stale ones deleted.)

## Design vs. decisions & dead ends

A system doc's body is **design**: current truth, present tense — no dates,
no "shipped on X" narration (`git log` is the chronology). Docs that
accumulate decisions end with an append-only **Decisions & dead ends**
section: dated one-liners, only for (a) a choice a future session might
re-litigate, with the why, and (b) approaches tried and rejected, with the
failure mode. If an entry describes *what* shipped rather than *why* or
*what to avoid*, it belongs in the commit message instead.

## Index

### Activity tracking

- **`TODO.md`** — outstanding work, grouped by area. When you finish an
  item, **delete it** — `git log` is the record of what shipped. No
  `~~strikethrough~~`; `TODO.md` stays focused on open work. When an item's
  scope changes mid-flight, edit the bullet.
- **`REVIEW.md`** — Jonah's review queue for **long, orchestrated runs**
  (overnight autonomy, agent chains): work that ships while nobody will read
  the conversation about it. The queue is a *substitute* for the
  conversation, not a supplement — each finding gets exactly one channel. In
  an interactive session, say it in the chat instead. Append items (date,
  what, how-to-review, any decision needed) **in the same commit as the
  work**. He deletes items as he clears them. **Important things only**: a
  decision he must make, a result only he can verify (visual behavior of
  the sim), or a surprise. Never acknowledgements, never a changelog, never
  tuning a later milestone will invalidate.

### Design docs (the "why")

- **`organism.md`** — the settled organism design: one representation for
  plants and creatures, fixed-shape genome, Perceiver-IO brain over limb
  tokens, particle bodies, ecology as the only selection, evolutionary
  metrics, the soil-specialization experiment and its controls, milestones.
  **Specified, not yet implemented**; it is the spec the organism chunks in
  `TODO.md` are built from. Update it in the same commit as any
  implementation that diverges.

The fluid, soil, and tree models are still documented in their owning
headers (`src/systems/particle_fluid2.cuh`, `soil.cuh`, `tree_types.cuh`)
and in `git log`. Create `docs/<topic>.md` and add it here when a design
outgrows header comments: a model with tunables
whose *rationale* matters (why these porosity / capillary constants), a
cross-system contract (how erosion moves material between fluid and soil),
or a measured comparison a future session would otherwise redo.

### Reference (measured)

- **`perf.md`** — dev-box hardware, toolchain versions, the **measured
  build job ceiling**, and headless baselines (command + numbers). Update
  when a machine is added, a toolchain moves, or a baseline is re-measured.

## Maintenance triggers

| Change                                          | Doc(s) to update                          |
|-------------------------------------------------|-------------------------------------------|
| New sim parameter                               | none — the X-macro line in `sim_params.h` *is* the doc; `--write-config` regenerates the TOML |
| New CLI mode or flag                            | root `CLAUDE.md` *Build, test & run* if it changes how agents run the app |
| Build ceiling / toolchain / baseline re-measured | `perf.md`                                 |
| New machine set up                              | `perf.md` (its own section)               |
| Item completed                                  | delete from `TODO.md`                     |
| Autonomous run ships something Jonah must see   | `REVIEW.md`, same commit                  |
| New cross-cutting subsystem or model rationale  | a new `docs/<topic>.md` + add to this index |
| Design settled in chat that a future session might re-litigate | the owning doc's *Decisions & dead ends* (create the doc if needed) |

When in doubt: if a future Claude instance walking into the repo would form
an inaccurate mental model from reading the docs, fix the docs.
