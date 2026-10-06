"""
Constants for the split generator.

Every number here is a decision recorded in `splits.md` section 2 or
`folds_manifest.md` section 2.2. Keeping them in one place stops the manifest
and the code from drifting apart.
"""
from __future__ import annotations

MANIFEST_SCHEMA = "lattice24-benchmark/folds/1"

# --- rules --------------------------------------------------------------
TIME_KEY = "submit_time"
MONTH_DERIVATION = "submit_time[0:7]"
REFERENCE_WINDOW = 24          # Arm A's look-back; a floor, not a cap
MIN_HISTORY_JOBS = 25          # a user needs this many jobs to be scoreable
HISTORY_RULE = "history_jobs_must_end_before_target_submit"
POSITIVE_STATE = "TIMEOUT"     # T1's positive class (DEADLINE counts as negative)

# --- split-generator decisions (splits.md 2.1) ---------------------------
MIN_MONTH_ROWS = 1_000         # S1: months below this are artefacts, not months
SAMPLE_N = 20_000              # how many test jobs every arm is also scored on
SAMPLE_FLOOR = 25              # S2: minimum per stratum
ACTIVITY_QUARTILES = 4         # S3: quiet -> busy
LOCKED_MONTHS = 6              # the holdout window

# --- guardrails (folds_manifest.md 2.2) ---------------------------------
MIN_MONTHS = 6
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 50
MIN_TRAIN_POS = 5
MIN_TEST_POS = 5
MIN_FOLDS = 3

DEFAULT_SEED = 42
