"""
Arm B's shared feature layer — the one piece of new data machinery in Arm B.

Arm A's `common.build_windows` emits four columns and throws the rest away. B
needs a fuller picture of the same history: what the target *asked for*, and
what the user's past jobs look like. That is all this module builds. The three
learners (B1/B2/B3) read one table and differ only in the learner, so a
difference between them is a difference in the model and nothing else.

**The legal-history rule is Arm A's, reused rather than reimplemented** — the
same `RULE_SAFE` as-of join (ended strictly before the target's submit, matched
at full microsecond resolution), and the same definition of `scoreable`
(`idx_end >= WINDOW - 1`). If B built its own ordering or its own strictness, a
B-vs-A difference could be the window rather than the features, and the
comparison the whole arm exists to make would be unreadable. The four window
features are likewise taken from `common._rolling` — the same code object, not a
copy — so `B ⊇ A` is true by construction (arm-b-planning.md §3 B-D4).

**What is deliberately absent** (arm-b-planning.md §1): anything that needs to
know what a *good* request looks like. No restart chains, no over-request ratio,
no change-point, no request-habit-vs-p95, no calibration. Those are Arm D. The
dividing line is: *if anyone holding the user's job log could compute it, it is
B; if it needs a notion of a correct request, it is D.*

The two acceptance tests that matter are in `bench/tests/test_features.py`: the
history a row is built from must all have ended before it was submitted, and
corrupting the *future* must leave the row byte-identical.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from . import common
from .common import FEATURES, RULE_SAFE

# Bump when the meaning of any column below changes, so a stale cache is rebuilt
# rather than silently mis-read. Independent of common.CACHE_VERSION: the two
# tables evolve for different reasons.
# v2: the cache also carries T2's label (`label_t2`); v1 carried T1's only.
FEATURE_VERSION = 2

# The four window features Arm A scores on, then the request fields, then the
# history aggregates. Order is the column order of the cached table.
REQUEST_FEATURES = (
    "req_timelimit_log", "req_cpus", "req_mem_log", "req_mem_unset",
    "req_gpus", "req_gpus_any", "req_nodes",
    "req_partition", "req_qos",
    "submit_hour", "submit_dow", "submit_weekend",
)
HISTORY_FEATURES = (
    "hist_n", "hist_span_days", "hist_since_last_end_s",
    "hist_gap_mean_s", "hist_gap_max_s",
    "hist_ratio_mean", "hist_ratio_std", "hist_ratio_min", "hist_ratio_max",
    "hist_ratio_last",
    "hist_timeout_rate", "hist_fail_rate", "hist_cancel_rate",
    "hist_last_state",
    "hist_timelimit_mean", "hist_timelimit_std",
    "hist_cpus_mean", "hist_cpus_std",
    "hist_queue_mean", "hist_queue_max",
    "hist_same_partition_rate",
)
# `hist_queue_*` read `queue_wait_s`, which Eagle does not carry: they come out
# null there rather than zero, so a learner sees "unknown", not "no wait".
FEATURE_NAMES = (*FEATURES, *REQUEST_FEATURES, *HISTORY_FEATURES)

# Strings, kept as strings in the cache; B1 hands these to LightGBM's native
# categorical handling and B2/B3 will need a code map. Nothing here is ordered.
CATEGORICAL_FEATURES = ("req_partition", "req_qos", "hist_last_state")

# Same set as `bench/eval/config.POSITIVE_STATES["T2"]`. Kept as a literal rather
# than imported so `bench.arms` does not depend on `bench.eval`; if either moves,
# the label test in test_features.py is what catches the drift.
FAIL_STATES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL")

_OUTPUT_COLUMNS = ["job_id", "row_id", "user_hash", "month", "scoreable",
                   "label", "label_t2", "energy_j", *FEATURE_NAMES]

# The columns read from the canonical table. `name`, `work_dir` and `submit_line`
# are sensitive and never named here, so they are never decoded (canonical_table.md).
_SOURCE_COLUMNS = [
    "job_id", "row_id", "user_hash", "submit_time", "end_time",
    "elapsed_s", "timelimit_s", "state", "energy_j",
    "partition", "qos", "cpus_req", "mem_req", "gpus_req", "nodes_req",
    "queue_wait_s",
]


def _moments(col: str, tag: str) -> list[pl.Expr]:
    """
    Running (sum, count of non-null, sum of squares) for one column, per user.

    Cumulative rather than windowed because B's history is unbounded
    (B-D4): every legal prior job contributes, and the running trio only ever
    needs the current row. Nulls are filled to 0 for the sums and counted
    separately, so a column that is absent on a dataset (Eagle's `queue_wait_s`)
    yields a null mean instead of a spurious zero.
    """
    v = pl.col(col)
    filled = v.fill_null(0.0)
    return [
        filled.cum_sum().over("user_hash").alias(f"_{tag}_sum"),
        v.is_not_null().cast(pl.Int32).cum_sum().over("user_hash").alias(f"_{tag}_n"),
        (filled * filled).cum_sum().over("user_hash").alias(f"_{tag}_sq"),
    ]


def _mean(tag: str) -> pl.Expr:
    """Running mean, null where nothing has been seen yet."""
    return (pl.when(pl.col(f"_{tag}_n") > 0)
            .then(pl.col(f"_{tag}_sum") / pl.col(f"_{tag}_n"))
            .otherwise(None))


def _std(tag: str) -> pl.Expr:
    """Running population sd (ddof=0, matching _rolling's `f_std`)."""
    n = pl.col(f"_{tag}_n")
    mean = pl.col(f"_{tag}_sum") / n
    var = (pl.col(f"_{tag}_sq") / n) - mean * mean
    return pl.when(n > 1).then(var.clip(lower_bound=0.0).sqrt()).otherwise(None)


def _cum_extreme(col: str, tag: str, *, largest: bool) -> pl.Expr:
    """
    Running min or max that ignores nulls.

    Polars' `cum_min`/`cum_max` propagate nulls, so a single null would poison
    every later row of that user. Substituting the identity element (∓inf) and
    mapping it back to null when nothing was seen avoids that.
    """
    v = pl.col(col)
    identity = float("-inf") if largest else float("inf")
    ext = (pl.when(v.is_not_null()).then(v).otherwise(identity)
           .cum_max() if largest else
           pl.when(v.is_not_null()).then(v).otherwise(identity).cum_min())
    seen = v.is_not_null().any().over("user_hash")
    return pl.when(seen).then(ext).otherwise(None).alias(f"_{tag}")


def build_features(lf: pl.LazyFrame, rule: str = RULE_SAFE) -> pl.LazyFrame:
    """
    Canonical rows -> one feature row per canonical row (`row_id`-keyed).

    `lf` must already carry `row_id` (see `common.with_row_id`). Output is one
    row per input row — including rows with no legal history, whose history
    features are null and whose `scoreable` is false. Keeping them (rather than
    dropping) is what lets a caller prove its `scoreable` set matches Arm A's.

    Only `RULE_SAFE` is supported: Arm B is never scored under the published
    rule, and building a leaky variant we do not use would be a lie waiting to
    happen.
    """
    if rule != RULE_SAFE:
        raise ValueError(
            f"Arm B builds only under {RULE_SAFE!r}, got {rule!r} "
            "(see arm-b-planning.md §5.1)")

    ordered = lf.select(_SOURCE_COLUMNS).with_columns([
        # Byte-for-byte the expression `common.build_windows` uses, so the four
        # window features below are Arm A's features and not merely equivalent.
        (pl.col("elapsed_s") / pl.col("timelimit_s")).alias("ratio"),
        pl.col("submit_time").dt.strftime("%Y-%m").alias("month"),
    ])

    # (1) The target's own row, and (2) the history ordered by end time.
    targets = ordered.select([
        "job_id", "row_id", "user_hash", "submit_time", "month", "state",
        "energy_j", "partition", "qos", "timelimit_s", "cpus_req", "mem_req",
        "gpus_req", "nodes_req",
    ]).with_columns(
        (pl.col("submit_time") - pl.duration(microseconds=1)).alias("submit_prev"))

    # `_rolling` sorts by (user_hash, end_time, job_id) itself and returns the
    # four window features plus `idx`, the row's position within its user — which
    # is Arm A's `idx_end` and therefore Arm A's `scoreable` guard.
    by_end = common._rolling(ordered, ["end_time", "job_id"]).rename({"idx": "idx_end"})

    states = pl.col("state")
    by_end = by_end.with_columns([
        *_moments("ratio", "ratio"),
        *_moments("timelimit_s", "tl"),
        *_moments("cpus_req", "cpus"),
        *_moments("queue_wait_s", "queue"),
        _cum_extreme("ratio", "ratio_min", largest=False),
        _cum_extreme("ratio", "ratio_max", largest=True),
        # Inter-arrival gaps: end-to-end between consecutive jobs, reset per user.
        pl.col("end_time").diff().over("user_hash").dt.total_seconds().alias("_gap"),
        pl.col("end_time").first().over("user_hash").alias("_first_end"),
        states.eq("TIMEOUT").cast(pl.Int32).cum_sum().over("user_hash").alias("_n_timeout"),
        states.is_in(FAIL_STATES).cast(pl.Int32).cum_sum().over("user_hash").alias("_n_fail"),
        (states == "CANCELLED").cast(pl.Int32).cum_sum().over("user_hash").alias("_n_cancel"),
    ]).with_columns([
        *_moments("_gap", "gap"),
        _cum_extreme("_gap", "gap_max", largest=True),
        _cum_extreme("queue_wait_s", "queue_max", largest=True),
    ])

    # How often the user lands on the partition *this* target asks for. Counted
    # per (user, partition) so the as-of join can look up the right stream; a
    # partition the user has never used matches nothing and the count fills to 0.
    partition_running = (
        by_end
        .with_columns(pl.col("partition").cum_count()
                      .over(["user_hash", "partition"]).alias("_part_n"))
        .select(["user_hash", "partition", "end_time", "_part_n"])
        .sort(["user_hash", "partition", "end_time"])
    )

    # A target inherits the last history row that ended strictly before it was
    # submitted — the identical `<= submit - 1us` construction as Arm A's
    # build_windows, for the identical reason (a job that never ran stores
    # `end_time == submit_time` and would match the target to itself).
    joined = targets.join_asof(
        by_end.select([
            "user_hash", "end_time", "idx_end", "ratio", "state", *FEATURES,
            "_ratio_sum", "_ratio_n", "_ratio_sq", "_ratio_min", "_ratio_max",
            "_tl_sum", "_tl_n", "_tl_sq",
            "_cpus_sum", "_cpus_n", "_cpus_sq",
            "_queue_sum", "_queue_n", "_queue_sq", "_queue_max",
            "_gap_sum", "_gap_n", "_gap_sq", "_gap_max",
            "_first_end", "_n_timeout", "_n_fail", "_n_cancel",
            "partition",
        ]).rename({
            "end_time": "hist_end_time",
            "ratio": "hist_ratio_last",
            "state": "hist_last_state",
        }),
        left_on="submit_prev", right_on="hist_end_time", by="user_hash",
        strategy="backward",
    )
    joined = joined.join_asof(
        partition_running.rename({"end_time": "hist_part_end"}),
        left_on="submit_prev", right_on="hist_part_end",
        by=["user_hash", "partition"], strategy="backward",
    )

    hist_n = pl.col("idx_end") + 1
    n_ratio = pl.col("_ratio_n")
    n_gap = pl.col("_gap_n")

    return joined.with_columns([
        # --- request fields: known at submit, no history needed -------------
        pl.col("timelimit_s").log1p().alias("req_timelimit_log"),
        pl.col("cpus_req").cast(pl.Float64).alias("req_cpus"),
        pl.col("mem_req").log1p().alias("req_mem_log"),
        # 62.5% of Eagle's `mem_req` is exactly 0, which is "unset", not "zero
        # bytes" — an indicator keeps that from being read as a real request.
        (pl.col("mem_req").fill_null(0.0) <= 0.0).cast(pl.Int8).alias("req_mem_unset"),
        pl.col("gpus_req").cast(pl.Float64).alias("req_gpus"),
        (pl.col("gpus_req").fill_null(0) > 0).cast(pl.Int8).alias("req_gpus_any"),
        pl.col("nodes_req").cast(pl.Float64).alias("req_nodes"),
        pl.col("partition").alias("req_partition"),
        pl.col("qos").alias("req_qos"),
        pl.col("submit_time").dt.hour().cast(pl.Int8).alias("submit_hour"),
        pl.col("submit_time").dt.weekday().cast(pl.Int8).alias("submit_dow"),
        (pl.col("submit_time").dt.weekday() >= 6).cast(pl.Int8).alias("submit_weekend"),

        # --- history aggregates: all legal prior jobs (B-D4) ---------------
        pl.when(hist_n > 0).then(hist_n).otherwise(None).alias("hist_n"),
        (pl.col("hist_end_time") - pl.col("_first_end")).dt.total_seconds()
          .truediv(86400.0).alias("hist_span_days"),
        (pl.col("submit_time") - pl.col("hist_end_time")).dt.total_seconds()
          .alias("hist_since_last_end_s"),

        _mean("gap").alias("hist_gap_mean_s"),
        pl.col("_gap_max").alias("hist_gap_max_s"),

        _mean("ratio").alias("hist_ratio_mean"),
        _std("ratio").alias("hist_ratio_std"),
        pl.when(n_ratio > 0).then(pl.col("_ratio_min")).otherwise(None)
          .alias("hist_ratio_min"),
        pl.when(n_ratio > 0).then(pl.col("_ratio_max")).otherwise(None)
          .alias("hist_ratio_max"),
        pl.col("hist_ratio_last"),

        (pl.col("_n_timeout") / hist_n).alias("hist_timeout_rate"),
        (pl.col("_n_fail") / hist_n).alias("hist_fail_rate"),
        (pl.col("_n_cancel") / hist_n).alias("hist_cancel_rate"),
        pl.when(pl.col("hist_last_state") == "COMPLETED").then(pl.lit("COMPLETED"))
          .when(pl.col("hist_last_state") == "TIMEOUT").then(pl.lit("TIMEOUT"))
          .when(pl.col("hist_last_state") == "CANCELLED").then(pl.lit("CANCELLED"))
          .when(pl.col("hist_last_state").is_in(FAIL_STATES)).then(pl.lit("FAILED"))
          .otherwise(pl.lit("OTHER")).alias("hist_last_state"),

        _mean("tl").alias("hist_timelimit_mean"),
        _std("tl").alias("hist_timelimit_std"),
        _mean("cpus").alias("hist_cpus_mean"),
        _std("cpus").alias("hist_cpus_std"),
        _mean("queue").alias("hist_queue_mean"),
        pl.col("_queue_max").alias("hist_queue_max"),
        (pl.col("_part_n").fill_null(0) / hist_n).alias("hist_same_partition_rate"),

        # --- label and scoreability, defined exactly as Arm A defines them --
        # `label` is T1's, under the name Arm A uses; `label_t2` is the failure
        # set. Both are the target's own outcome, so both are read from the
        # target's row, never from history.
        (pl.col("state") == "TIMEOUT").alias("label"),
        pl.col("state").is_in(FAIL_STATES).alias("label_t2"),
        ((pl.col("idx_end").is_not_null()) & (pl.col("idx_end") >= common.WINDOW - 1))
        .alias("scoreable"),
    ]).select(_OUTPUT_COLUMNS)


def ensure_features(lf: pl.LazyFrame, canonical_path: str | Path,
                    cache_dir: str | Path, rebuild: bool = False,
                    max_chunk_rows: int = common.WINDOW_MAX_CHUNK_ROWS,
                    rule: str = RULE_SAFE) -> Path:
    """
    Build the feature table once and keep it as Parquet.

    Same chunking, month-clustering and staleness key as Arm A's window cache —
    literally the same code (`common._ensure_cached`) — so the two cannot drift.
    Keyed on canonical sha256 + `FEATURE_VERSION` + window rule.
    """
    if rule != RULE_SAFE:
        raise ValueError(f"Arm B builds only under {RULE_SAFE!r}, got {rule!r}")
    return common._ensure_cached(
        lf, canonical_path, cache_dir,
        filename=f"{Path(canonical_path).stem}.{rule}.features.parquet",
        version=FEATURE_VERSION, rule=rule,
        builder=lambda frame: build_features(frame, rule=rule),
        max_chunk_rows=max_chunk_rows, rebuild=rebuild, what="features",
        extra_meta={"feature_version": FEATURE_VERSION, "n_features": len(FEATURE_NAMES)})
