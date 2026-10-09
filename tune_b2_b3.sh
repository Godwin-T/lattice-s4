#!/usr/bin/env bash
# W1.6/W1.7 tuning — one-shot Optuna for B2 (MLP) and B3 (GRU), T1 and T2.
#
# Launch as a TRANSIENT USER SERVICE, under a memory cap (the machine is shared):
#   systemd-run --user --unit=tune-b23 --collect \
#     -p MemoryMax=14G -p MemorySwapMax=0 -p CPUQuota=600% \
#     --setenv=POLARS_MAX_THREADS=4 --setenv=OMP_NUM_THREADS=4 \
#     /bin/bash -lc './tune_b2_b3.sh'
#
# NOT `--scope`. A scope is owned by the shell that launched it: on 8 October the
# B1 run was launched as `b1-eagle-2057.scope` from a session-managed shell, the
# shell was stopped, systemd stopped the scope, and the run was SIGABRT'd mid
# fold 40 of 45 with nothing written for that step. `systemd-run` without
# `--scope` hands the unit to the user manager, which outlives the launching
# shell. This tuning is hours long; it must not ride on a session.
#
# Sequential on purpose. Four studies, each 50 trials, and each trial is a full
# pass over its arm's validation block: B2 fits an MLP per block month, B3 fits a
# GRU. Two of these at once does not halve the wall clock, it doubles both.
#
# The block is the arm's own (arm-b-planning.md §8.1, decision B-D1a):
#   B1, B2 -> the full eight months 2018-12..2019-07
#   B3     -> the four most recent, 2019-04..2019-07 (`BLOCK_MONTHS_CAP`)
# B3's reduced block is written into params/b3-*.json as `validation_months`, so
# the deviation is visible in the artefact rather than only here.
#
# Studies persist to studies/<ARM>-<TASK>-<dataset>.db and are resumable: re-running
# a finished study is nearly free, and an interrupted one continues from its last
# trial. `params/<arm>-<task>.json` is written only when the study finishes.
#
# A run must not start before its params file exists — `load_params` falls back to
# the untuned defaults silently, so starting early would produce an untuned run
# wearing a tuned arm's name.
set -u

THREADS=${B23_THREADS:-4}
DEVICE=${B23_DEVICE:-auto}
export POLARS_MAX_THREADS="$THREADS" OMP_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       NUMEXPR_NUM_THREADS="$THREADS"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1

MANIFEST=data/manifests/eagle_parquet.folds.json
CANONICAL=data/canonical/eagle_parquet.parquet

step() { echo; echo "=== $* ==="; date -Is; }

for ARM in B2 B3; do
  for TASK in T1 T2; do
    step "tune $ARM $TASK (50 trials)"
    python3 -m bench.arms.tuning --arm "$ARM" --task "$TASK" \
        --manifest "$MANIFEST" --canonical "$CANONICAL" \
        --out params --trials 50 --threads "$THREADS" --device "$DEVICE" || exit 1
  done
done

echo; echo "=== ALL DONE ==="; date -Is
ls -la params/
