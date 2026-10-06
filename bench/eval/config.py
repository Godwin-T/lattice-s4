"""
Every constant the evaluator uses. Mirrors `metrics.md`; keep them in one place
so the document and the code cannot drift apart.
"""
from __future__ import annotations

# Operating points: we report at several review budgets, and 5% is the headline.
K_VALUES = (1, 2, 5, 10, 20)
HEADLINE_K = 5

# Calibration.
ECE_BINS = 15

# Uncertainty.
BOOTSTRAP_RESAMPLES = 1_000
ALPHA = 0.05

# The label-shuffle control must land in this band (metrics.md M1). The wider
# tool-style tolerance is reported alongside it for context.
CONTROL_BAND = (0.45, 0.55)
CONTROL_TOLERANCE_FLOOR = 0.02

# A metric computed on fewer months than this is suppressed, with the count
# reported (metrics.md section 9).
MIN_MONTHS_FOR_METRIC = 3

# The interesting outcome per task. T1 counts only TIMEOUT: a DEADLINE job was
# killed at an absolute cutoff, not by its own time limit, so it is a negative
# (plan.md 14.6). A CANCELLED job is a person's decision, so it is never a
# failure either.
POSITIVE_STATES = {
    "T1": {"TIMEOUT"},
    "T2": {"FAILED", "OUT_OF_MEMORY", "NODE_FAIL"},
}

# Columns an arm must hand in (metrics.md section 7, and section 2 on why
# calibration is a run variant rather than an extra column).
PREDICTION_COLUMNS = (
    "run_id", "arm", "task", "fold_id", "test_month",
    "job_id", "score", "probability", "latency_ms",
)

# Columns the evaluator adds by joining the canonical table.
TRUTH_COLUMNS = ("label", "state", "energy_j", "energy_tier")
