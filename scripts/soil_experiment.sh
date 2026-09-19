#!/usr/bin/env bash
#
# The soil-specialization experiment and its three controls
# (`docs/organism.md`, *The first experiment*). Eleven headless runs, then
# `soil_score.py` over what they wrote.
#
#   ./scripts/soil_experiment.sh [N]
#
# N is the step count of a main run, default 60000 (about two and a half
# minutes each on the dev box, `docs/perf.md`); the two transplant runs are
# N/4. Output goes under `runs/soil/`, which is gitignored.
#
# Needs nothing but the release binary and python3's standard library. Set
# RUNTIME to pick a backend (default cuda) and ALIFE to point at another
# binary.
set -euo pipefail

N=${1:-60000}
SHORT=$((N / 4))
SEEDS="1 2 3"
RUNTIME=${RUNTIME:-cuda}
ALIFE=${ALIFE:-./target/release/alife}
OUT=runs/soil

# The CUDA runtime JITs from PTX and needs the 13.2 libraries on this box
# (`docs/perf.md` says why). Only set if the caller has not.
CUDA_LIBS=/usr/local/cuda-13.2/lib64
if [ -d "$CUDA_LIBS" ] && [ -z "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH=$CUDA_LIBS
fi

if [ ! -x "$ALIFE" ]; then
  echo "no binary at $ALIFE — cargo +1.98.1 build --release -j16" >&2
  exit 1
fi
mkdir -p "$OUT"

# Every run shares these. `--max-organisms 1024` rather than the default 256
# because a run left to itself fills every organism slot (`organism.md`,
# decisions); raising it here moves the wall without moving the default.
# `--founders 60` is ten per column, since founders are spread over the soil
# and not over the world.
COMMON="--headless --runtime $RUNTIME --terrain-mode 2 --founders 60 \
        --max-organisms 1024 --metrics-every 100"

run() {
  local name=$1
  shift
  local started=$SECONDS
  echo "=== $name ==="
  # Unquoted on purpose: COMMON is a flag list, not one argument.
  # shellcheck disable=SC2086
  "$ALIFE" $COMMON "$@" >"$OUT/$name.log" 2>&1 ||
    { echo "$name failed; see $OUT/$name.log" >&2; exit 1; }
  grep -E "alive=|births=" "$OUT/$name.log" || true
  echo "    $((SECONDS - started)) s"
}

for seed in $SEEDS; do
  # (a) the main condition: the six soils where they are.
  run "main_$seed" --seed "$seed" --iterations "$N" \
    --metrics "$OUT/main_$seed.csv" --save-pop "$OUT/main_$seed.pop"

  # (b) the isolation control: the same geometry, one soil everywhere, so
  # whatever splits the columns there is the gaps and not the soil.
  run "isolation_$seed" --seed "$seed" --iterations "$N" \
    --uniform-soil silt --metrics "$OUT/isolation_$seed.csv"

  # (c) the position control: the same six soils dealt out to the six
  # positions differently, so a pattern that follows the position rather than
  # the soil shows up as one that moves with the permutation.
  run "permuted_$seed" --seed "$seed" --iterations "$N" \
    --soil-permutation "$seed" --metrics "$OUT/permuted_$seed.csv"
done

# (d) the transplant. Two runs from one saved population and one world seed:
# one with two columns' organisms swapped, one with nobody moved. The pair is
# the comparison — what the moved organisms earn against what the residents
# earn in the world they were already in.
#
# Which seed and which two columns is not fixed in advance: naming a pair up
# front only works if both columns still hold plants at the end of the main
# run, and mostly they do not. `--pick` names the main run with the most
# plants standing in columns and its two fullest columns, and the choice is
# written down beside the runs so the scorer compares what was actually moved.
rm -f "$OUT/transplant.txt"
if CHOICE=$(python3 scripts/soil_score.py --pick "$OUT"); then
  read -r TSEED TA TB <<<"$CHOICE"
  echo "$CHOICE" >"$OUT/transplant.txt"
  echo "transplanting seed $TSEED, $TA <-> $TB"

  run "transplant_$TSEED" --seed "$TSEED" --iterations "$SHORT" \
    --load-pop "$OUT/main_$TSEED.pop" \
    --transplant "$TA:$TB" --transplant "$TB:$TA" \
    --metrics "$OUT/transplant_$TSEED.csv"

  run "resident_$TSEED" --seed "$TSEED" --iterations "$SHORT" \
    --load-pop "$OUT/main_$TSEED.pop" --metrics "$OUT/resident_$TSEED.csv"
else
  echo "no main run left plants standing in two columns: no transplant to run"
fi

echo
python3 scripts/soil_score.py "$OUT"
