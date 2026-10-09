#!/usr/bin/env bash
# W1.6/W1.7 — run Arm B2 (MLP) and Arm B3 (GRU), then score each, T1 and T2.
#
# Launch as a TRANSIENT USER SERVICE, under a memory cap (the machine is shared):
#   systemd-run --user --unit=run-b23 --collect \
#     -p MemoryMax=14G -p MemorySwapMax=0 -p CPUQuota=600% \
#     --setenv=POLARS_MAX_THREADS=4 --setenv=OMP_NUM_THREADS=4 \
#     /bin/bash -lc './run_b2_b3.sh'
#
# Not `--scope`: a scope dies with the shell that launched it (see tune_b2_b3.sh).
#
# Run AFTER `tune_b2_b3.sh`: a run must not start before its params file exists,
# because `load_params` falls back to the untuned defaults silently and the run
# would then be an untuned one wearing a tuned arm's name. This script checks.
#
# Sequential. B2 and B3 both take `--train-rows` (the per-fold training cap,
# default 250k, recorded in metadata either way — a capped run is not directly
# comparable to B1's uncapped one, so the number matters). B3 also builds its
# sequence cache on first use; that costs one pass over the eligible rows.
set -u

THREADS=${B23_THREADS:-4}
DEVICE=${B23_DEVICE:-auto}
export POLARS_MAX_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS"
ROOT=/home/godwin/Documents/Workflow/Work/Peep/lattice24-assess
cd "$ROOT" || exit 1

MANIFEST=data/manifests/eagle_parquet.folds.json
CANONICAL=data/canonical/eagle_parquet.parquet

# Refuse to run untuned: every arm here needs both tasks' params present.
for ARM in b2 b3; do
  for TASK in t1 t2; do
    [ -f "params/$ARM-$TASK.json" ] || {
      echo "refusing to start: params/$ARM-$TASK.json is missing (run tune_b2_b3.sh first)" >&2
      exit 1
    }
  done
done

step() { echo; echo "=== $* ==="; date -Is; }

for ARM in B2 B3; do
  step "arm $ARM (T1 + T2 in one sweep, control on 5 folds x 3)"
  python3 -m bench.arms.cli --arm "$ARM" \
      --manifest "$MANIFEST" --canonical "$CANONICAL" \
      --out runs --control-folds 5 --threads "$THREADS" --device "$DEVICE" || exit 1

  for TASK in t1 t2; do
    step "evaluate $ARM-eagle_parquet-safe-$TASK"
    python3 -m bench.eval.cli \
        --predictions "runs/$ARM-eagle_parquet-safe-$TASK/predictions.parquet" \
        --manifest "$MANIFEST" --canonical "$CANONICAL" \
        --arm-metadata "runs/$ARM-eagle_parquet-safe-$TASK/metadata.json" \
        --task "$(echo "$TASK" | tr '[:lower:]' '[:upper:]')" \
        --unit month --out results --threads 4 || exit 1
  done
done

echo; echo "=== ALL DONE ==="; date -Is
