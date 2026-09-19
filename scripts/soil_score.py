#!/usr/bin/env python3
"""Score the soil-specialization experiment.

    python3 scripts/soil_score.py [runs/soil]

Reads the metrics CSVs `soil_experiment.sh` wrote and prints, per condition
averaged over seeds, the last-quarter mean per soil column of every trait, and
then the three comparisons the experiment exists to make:

1.  Do the per-column trait differences in the main condition exceed the ones
    spatial isolation produces on its own?  The isolation control has the same
    six separated habitats and one soil in all of them, so whatever spread it
    shows is drift across barriers.  The main condition has to beat it.
2.  Does the pattern follow the soil or the position?  In the permuted runs
    each soil sits somewhere different, so the same numbers can be grouped by
    soil or by position.  Whichever grouping explains more of the variance is
    what the trait is tracking.
3.  What does a transplanted organism earn?  The energy of the two swapped
    columns in the swapped run against the same columns in the run where
    nobody moved.  Which seed and which pair of columns that was is in
    `transplant.txt`, because it depends on where the main runs left a
    population to move (see `--pick`).

    python3 scripts/soil_score.py --pick [runs/soil]

prints `<seed> <column> <column>` — the main run with the most plants
standing in columns, and the two columns of it that hold the most — or exits
non-zero when no main run left plants in two columns at once.  That is how
`soil_experiment.sh` decides what to transplant, rather than naming a pair up
front that may well be empty by the end of a run.

Standard library only, and no plots.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

# The per-column traits, in the order the CSV writes them.
TRAITS = ["alive", "root_frac", "height", "leaf_count", "energy_mean", "species"]
GLOBALS = ["alive", "species", "lineages", "generation_max", "births", "deaths"]
SEEDS = [1, 2, 3]


class Run:
    """One metrics CSV: its column labels, in position order, and its rows."""

    def __init__(self, path: Path):
        with path.open() as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise SystemExit(f"{path} holds no samples")
        self.path = path
        self.labels = self._labels()

    def _labels(self) -> list[str]:
        # `alive_<label>` appears once per column, in ascending x, so the
        # header alone gives both the soil (the label) and the position (its
        # index).  Taken from `alive_` rather than any other trait because it
        # is the first field of each per-column block.
        out = []
        for name in self.rows[0]:
            if name.startswith("alive_"):
                out.append(name[len("alive_") :])
        return out

    def last_quarter(self) -> list[dict[str, str]]:
        """The rows in the final quarter of the run, by step."""
        last = int(self.rows[-1]["step"])
        return [r for r in self.rows if int(r["step"]) >= 0.75 * last]

    def column_trait(self, label: str, trait: str) -> float:
        """Mean of one column's trait over the last quarter."""
        rows = self.last_quarter()
        return statistics.fmean(float(r[f"{trait}_{label}"]) for r in rows)

    def global_trait(self, name: str) -> float:
        rows = self.last_quarter()
        return statistics.fmean(float(r[name]) for r in rows)

    def at(self, fraction: float, name: str) -> float:
        """One field at a fraction of the way through the run."""
        i = min(len(self.rows) - 1, int(fraction * (len(self.rows) - 1)))
        return float(self.rows[i][name])


def load(directory: Path, condition: str) -> dict[int, Run]:
    runs = {}
    for seed in SEEDS:
        path = directory / f"{condition}_{seed}.csv"
        if path.exists():
            runs[seed] = Run(path)
    return runs


def table(title: str, runs: dict[int, Run], by_position: bool = False) -> None:
    """Per-column trait means, averaged over seeds, and their spread."""
    if not runs:
        print(f"{title}: no runs found\n")
        return
    first = next(iter(runs.values()))
    if by_position:
        keys = [f"pos{i}" for i in range(len(first.labels))]

        def value(run: Run, i: int, trait: str) -> float:
            return run.column_trait(run.labels[i], trait)
    else:
        keys = first.labels

        def value(run: Run, i: int, trait: str) -> float:
            return run.column_trait(keys[i], trait)

    print(f"{title}  ({len(runs)} seeds, last-quarter means)")
    print("  " + f"{'column':<12}" + "".join(f"{t:>12}" for t in TRAITS))
    rows = []
    for i, key in enumerate(keys):
        cells = [statistics.fmean(value(r, i, t) for r in runs.values()) for t in TRAITS]
        rows.append(cells)
        print("  " + f"{key:<12}" + "".join(f"{c:>12.3f}" for c in cells))
    spread = [max(c) - min(c) for c in zip(*rows)]
    stdev = [statistics.pstdev(c) for c in zip(*rows)]
    print("  " + f"{'max-min':<12}" + "".join(f"{c:>12.3f}" for c in spread))
    print("  " + f"{'stdev':<12}" + "".join(f"{c:>12.3f}" for c in stdev))
    print()


def globals_table(conditions: dict[str, dict[int, Run]]) -> None:
    print("whole population  (last-quarter means over seeds)")
    print("  " + f"{'condition':<14}" + "".join(f"{g:>16}" for g in GLOBALS))
    for name, runs in conditions.items():
        if not runs:
            continue
        cells = [
            statistics.fmean(r.global_trait(g) for r in runs.values()) for g in GLOBALS
        ]
        print("  " + f"{name:<14}" + "".join(f"{c:>16.2f}" for c in cells))
    print()


def spread_of(runs: dict[int, Run], trait: str) -> float:
    """Standard deviation across columns of the seed-averaged column means."""
    first = next(iter(runs.values()))
    per_column = [
        statistics.fmean(r.column_trait(label, trait) for r in runs.values())
        for label in first.labels
    ]
    return statistics.pstdev(per_column)


def eta_squared(values: dict[tuple[int, str], float], keys: list[str]) -> float:
    """Fraction of the variance the grouping `keys` explains.

    `values` is one number per (seed, group).  Between-group variance over
    total variance: 1 means the group decides the value outright, 0 means it
    tells you nothing.
    """
    flat = list(values.values())
    if len(flat) < 2:
        return float("nan")
    total = statistics.pvariance(flat)
    if total == 0:
        return float("nan")
    grand = statistics.fmean(flat)
    between = 0.0
    for key in keys:
        group = [v for (_, k), v in values.items() if k == key]
        if group:
            between += len(group) * (statistics.fmean(group) - grand) ** 2
    return between / len(flat) / total


def compare_isolation(main: dict[int, Run], isolation: dict[int, Run]) -> None:
    print("--- 1. soil against spatial isolation alone ---")
    print("  The isolation control is the same six separated habitats with one")
    print("  soil in all of them, so its spread is what drift across barriers")
    print("  produces. The main condition has to beat it for the soil to be")
    print("  doing anything.")
    print()
    if not main or not isolation:
        print("  missing runs\n")
        return
    print("  " + f"{'trait':<12}{'main stdev':>14}{'isolation':>14}{'ratio':>10}")
    for trait in TRAITS:
        a, b = spread_of(main, trait), spread_of(isolation, trait)
        ratio = a / b if b else float("inf")
        print("  " + f"{trait:<12}{a:>14.3f}{b:>14.3f}{ratio:>10.2f}")
    print()


def compare_permutation(permuted: dict[int, Run]) -> None:
    print("--- 2. does the pattern follow the soil or the position? ---")
    print("  Each seed deals the six soils to the six positions differently, so")
    print("  the same numbers group two ways. eta^2 is the fraction of the")
    print("  variance a grouping explains; the bigger one is what the trait is")
    print("  tracking.")
    print()
    if len(permuted) < 2:
        print("  needs at least two permuted seeds\n")
        return
    first = next(iter(permuted.values()))
    n = len(first.labels)
    soils = sorted({label for r in permuted.values() for label in r.labels})
    positions = [f"pos{i}" for i in range(n)]
    print("  " + f"{'trait':<12}{'eta^2 by soil':>16}{'eta^2 by position':>20}{'':>4}")
    for trait in TRAITS:
        by_soil = {}
        by_position = {}
        for seed, run in permuted.items():
            for i, label in enumerate(run.labels):
                value = run.column_trait(label, trait)
                by_soil[(seed, label)] = value
                by_position[(seed, positions[i])] = value
        s = eta_squared(by_soil, soils)
        p = eta_squared(by_position, positions)
        if s != s or p != p:  # NaN: the trait is the same in every column
            verdict = "flat"
        elif abs(s - p) < 1e-6:  # eta^2 is a fraction, so this is "the same"
            verdict = "tie"
        else:
            verdict = "soil" if s > p else "position"
        print("  " + f"{trait:<12}{s:>16.3f}{p:>20.3f}   {verdict}")
    print()


def pick_transplant(directory: Path) -> tuple[int, str, str] | None:
    """The seed and pair of columns worth transplanting between.

    The main run whose columns hold the most plants, and its two fullest
    columns. A pair named up front — sand and clay, say — is no use when both
    are empty by the end of the run, which is the usual outcome.
    """
    best = None
    for seed, run in load(directory, "main").items():
        counts = sorted(
            ((run.column_trait(label, "alive"), label) for label in run.labels),
            reverse=True,
        )
        if len(counts) < 2 or counts[1][0] <= 0:
            continue
        total = sum(c for c, _ in counts)
        if best is None or total > best[0]:
            best = (total, seed, counts[0][1], counts[1][1])
    return None if best is None else (best[1], best[2], best[3])


def read_choice(directory: Path) -> tuple[int, str, str] | None:
    """What `soil_experiment.sh` recorded that it transplanted."""
    path = directory / "transplant.txt"
    if not path.exists():
        return None
    seed, source, destination = path.read_text().split()
    return int(seed), source, destination


def compare_transplant(directory: Path) -> None:
    print("--- 3. the transplant ---")
    choice = read_choice(directory)
    if choice is None:
        print("  no transplant.txt: the runs recorded no transplant\n")
        return
    seed, a, b = choice
    print(f"  Seed {seed}'s population re-loaded twice into the same world: once")
    print(f"  with the {a} and {b} columns' organisms swapped, once with nobody")
    print("  moved. A specialist away from home earns less than the residents.")
    print()
    moved_path = directory / f"transplant_{seed}.csv"
    home_path = directory / f"resident_{seed}.csv"
    if not moved_path.exists() or not home_path.exists():
        print("  missing runs\n")
        return
    moved, home = Run(moved_path), Run(home_path)

    print("  last-quarter means")
    print(
        "  "
        + f"{'column':<10}{'residents':>12}{'incomers':>12}"
        + f"{'incomer/resident':>20}{'incomer/its home':>20}"
    )
    for column, source in [(a, b), (b, a)]:
        resident = home.column_trait(column, "energy_mean")
        incomer = moved.column_trait(column, "energy_mean")
        at_home = home.column_trait(source, "energy_mean")
        print(
            "  "
            + f"{column:<10}{resident:>12.3f}{incomer:>12.3f}"
            + f"{incomer / resident if resident else float('nan'):>20.2f}"
            + f"{incomer / at_home if at_home else float('nan'):>20.2f}"
        )
    print()
    # The level a column reaches is mostly the energy its organisms walked in
    # with — a transplanted plant carries its own — so what a specialist
    # claim rests on is the *gain*: what they earn in the column they are in
    # now.  Worse away from home means a smaller gain than the residents'.
    print("  energy through the run (fraction of it elapsed)")
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    header = "".join(f"{f:>10.0%}" for f in fractions) + f"{'gain':>10}"
    print("  " + f"{'series':<24}" + header)
    for name, run in [("nobody moved", home), ("swapped", moved)]:
        for column in (a, b):
            series = [run.at(f, f"energy_mean_{column}") for f in fractions]
            cells = "".join(f"{v:>10.2f}" for v in series)
            print(
                "  "
                + f"{name + ', ' + column:<24}"
                + cells
                + f"{series[-1] - series[0]:>10.2f}"
            )
    print()


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "--pick":
        directory = Path(argv[1] if len(argv) > 1 else "runs/soil")
        choice = pick_transplant(directory)
        if choice is None:
            raise SystemExit("no main run left plants standing in two columns")
        print("{} {} {}".format(*choice))
        return

    directory = Path(argv[0] if argv else "runs/soil")
    conditions = {name: load(directory, name) for name in ("main", "isolation", "permuted")}
    any_run = next((r for runs in conditions.values() for r in runs.values()), None)
    if any_run is None:
        raise SystemExit(f"no metrics CSVs under {directory}")
    steps = int(any_run.rows[-1]["step"])

    print(f"=== soil specialization, {directory}, {steps} steps, seeds {SEEDS} ===\n")
    table("main condition, by soil", conditions["main"])
    table("isolation control (--uniform-soil silt), by position", conditions["isolation"])
    table("position control (--soil-permutation), by soil", conditions["permuted"])
    table(
        "position control (--soil-permutation), by position",
        conditions["permuted"],
        by_position=True,
    )
    globals_table(conditions)
    compare_isolation(conditions["main"], conditions["isolation"])
    compare_permutation(conditions["permuted"])
    compare_transplant(directory)


if __name__ == "__main__":
    main()
