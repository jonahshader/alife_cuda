# Handoff — the growth model is built; the energy balance is the next call (2026-09-19)

The session of 2026-09-19 (interactive, after the overnight run) fixed the
GUI (eframe's default device limits), the driver/toolchain workarounds, and
the light grid (soil is opaque), and then designed and shipped the growth
model in four delegation cycles. Read `docs/organism.md`'s *Growth* section
and the decisions entry "Growth is continuous..." first; `docs/TODO.md` has
what is left; `docs/perf.md` has every run's numbers.

## What shipped (all on `main`, `8944ec9`; nothing pushed this session)

1. **Continuous length** (`grown` per limb, fractional last segment), the
   genome's `growth_bias` / `growth_rate` per limb, softmax allocation of
   `energy − seed_cost` over growing limbs, child progress ≤ parent
   progress. Whole-limb sprouting and `sprout_cost` are gone.
2. **Thickness and volume**: `target_thickness` per limb, live thickness
   `target × sqrt(progress)`, growth buys volume, upkeep is per volume
   (`upkeep_per_volume` replaced `upkeep_per_particle`). Segments draw at
   thickness.
3. **Breaking**: the constraint pass keeps each joint's worst bend; a
   non-root limb past `break_angle × clamp(thickness / break_thickness_ref,
   0, 1)` sheds itself and its descendants as organic matter and regrows.
4. **The brain's part**: `head_growth` (one logit per limb, added to
   `growth_bias`) and the actuator's first output nudging the joint target
   by `actuator_gain × tanh(a)`. Brain is 18,659 params; population file
   version 4 reshapes older rows.

Each cycle: Opus builder in a worktree → fast-forward merge → claims rerun
→ Sonnet find-only review → verified fixes committed. 154 tests, clippy
clean, `--founders 0` byte-identical throughout, ~1.46 ms/step at 64
founders on CUDA.

## Open, and Jonah's to decide (also said in chat)

- **The energy balance.** With volume pricing a seedling's first segment
  is cheap, so nothing starves: the 64-founder run fills all 256 slots by
  step 3000 and `max_organisms` is the ceiling, not light or space. The
  knobs are what a root earns, what a seed germinates with, and whether the
  growth reserve is `seed_cost` (`TODO.md`, first growth bullet, and the
  `--column-top` bullet, which is the same question).
- **`--actuator-gain`.** At the shipped 0.3 rad an untrained brain wanders
  every joint and limb breaking more than doubles (109 → 238 in the
  reference run); 0.1 gave the tallest, highest-energy bodies of the sweep
  (`perf.md`). One flag, no code change.
- **The window.** Plants should now grow visibly, thicken, and snap.
  Nobody has watched it: `./target/release/alife --founders 64 --seed 42`.

## What the next session should NOT do on its own

Do not tune the energy constants or the actuator gain; both are Jonah's.

## Machine notes

- Driver 610 (CUDA 13.3), default toolchain 1.98.1: plain `cargo` and plain
  `./target/release/alife --runtime cuda` work. No `+1.98.1`, no
  `LD_LIBRARY_PATH` (builders' reports that mention it were setting a path
  that no longer matters).
- The `code-review-find` workflow ran four for four this session with a
  target string (no staged diff); ~5 minutes and ~0.9M tokens each.
