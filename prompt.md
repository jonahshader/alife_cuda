# Handoff — the loose threads after the stability session (2026-10-13)

The last two sessions turned the evolution loop from a branching process
into an ecology and then fixed what kept it grass: the water ledger
(2026-10-11), and on 2026-10-12 the joint-axis rule, body drag and the
speed cap, angle stiffness 0.9, the window drawing settled positions, and
defaults that evolve (`--founders 20`, 1024 slots). Each has a dated entry
in `docs/organism.md`'s *Decisions & dead ends* with the numbers; read
those entries and `docs/TODO.md`'s life-cycle bullets first.

Work one question at a time with a short recommendation (root `CLAUDE.md`,
*Design discussions*). Jonah runs the window; Claude measures headless. A
300k-step run is six minutes; `--timing` is off by default and must stay
off for runs whose result is the simulation. Builds: `taskset -c 0-3 cargo
… -j4`. Compare settings by seed-matched runs, a few seeds each.

## Threads, in the order Jonah raised them

1. **Seed redesign — design settled in chat, not built.** Jonah wants a
   discrete part that houses a growing seed, holds it, and releases it
   under brain control. Recommended and not yet confirmed: the existing
   seed part type is the housing (the "yellow limbs" mutation already
   makes); once fully grown it fills a seed by drawing `seed_energy` from
   the pool over ticks; the brain's reserved per-limb head output is the
   release gate (seed leaves from the limb's tip with the tip's velocity);
   no seed limb, no reproduction; founders get one on the stem, so seed
   count and placement become heritable. The brain is recurrent, so
   nothing about its tick rate changes with this.
2. **Seeds and the ground — Jonah's open question.** A seed today passes
   through soil as water does and germinates when slow inside it. The two
   options he named: soil solid with zero friction (seed rests on and
   slides along the surface, water carries it, germinates when at rest on
   soil), or stop dead at first contact. Recommended: solid and sliding,
   because it is the only dispersal beyond the drop. He has not answered.
   Seeds in flight are also in no limb, so they get neither drag nor the
   speed cap; give them both here.
3. **Yellow limbs.** Mutation's add draws a type from root/stem/leaf/seed
   and retype from stem/leaf/seed; a seed-type limb earns nothing, drinks
   nothing and costs upkeep. Held unfixed because thread 1 gives them a
   job; if thread 1 stalls, restrict both draws (same random stream, two
   constants in `genome/mutate.rs`).
4. **Growing tips.** A limb's newest segment is millimetres long and flips
   under the fluid every step. The axis rule stopped that being read as a
   bend; the motion remains. Proposed: move a growing particle with its
   predecessor along the limb's axis — kinematic, no SPH forces — until its
   segment is some fraction of `limb_segment_length`.
5. **Actuated limbs (milestone 2).** Off until a rate limit and an energy
   cost per radian exist; strength is a global param until there is a cost
   to trade against. Breaking by stretch as well as bend is cheap and
   belongs alongside.
6. **Generation time and drift.** ~9000 steps a generation; a population
   of a few hundred is one lineage by step 40k–115k. The lever is the
   economy's pace (light gain, upkeep, drink rate scale together), not the
   tick. Take up after thread 1, which changes generation time too.
7. **The water cycle as carrying capacity.** Late in every long run most
   water is vapor or inside plants; the condensation return rate has never
   been tuned as the ceiling it is. Two thirds of `alive` is seeds in
   flight holding slots.
8. **Step cost.** Constraint pass a quarter of kernel time (one thread per
   organism, serial over limbs × particles × sweeps); the three neighbour
   kernels a third. Both in `docs/TODO.md`; neither is a one-liner.
9. **Limb compliance.** Angle stiffness under 0.8 is a Gauss-Seidel limit
   cycle, not a bend; limbs that bend in flow by a set amount need XPBD.
10. Jonah said he has **more low-hanging fruit** he was holding until
    Claude caught up. Ask.

## Parked

- **GUI tool palette**: a Select tool (hover stats, click to inspect) with
  the water grabber as a tool you switch to. Agreed in principle, not
  built; an inspector would help once single plants are what matters.
- **No test compares backends with each other** since the parity test went.
- `docs/REVIEW.md` still has Jonah's older review items.
