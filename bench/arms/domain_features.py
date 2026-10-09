"""Submit-time-safe domain features for Arm D.

This module is deliberately separate from :mod:`features`.  B's cache must
remain unchanged so that a D-versus-B comparison can attribute any difference
to the domain features.  Every output row is keyed by the canonical ``row_id``
and is built from jobs whose ``end_time`` is strictly before the target's
``submit_time``.

The first slice contains request-habit and historical failure features.  The
Kestrel D2/D4 prior-evidence features will be added on top of this stable
interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import common


DOMAIN_FEATURE_VERSION = 1
HISTORY_WINDOW = 50
FAILURE_STATES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL")

FEATURE_COLUMNS = (
    "domain_history_available",
    "hist_elapsed_p50_s",
    "hist_elapsed_p95_s",
    "hist_elapsed_mean_s",
    "hist_timelimit_p95_s",
    "request_vs_elapsed_p95",
    "request_excess_over_elapsed_p95_s",
    "request_exceeds_elapsed_p95",
    "timelimit_vs_hist_timelimit_p95",
    "failures_last_5",
    "failures_last_10",
    "timeouts_last_5",
    "timeouts_last_10",
    "time_since_last_failure_s",
    "time_since_last_timeout_s",
    "prior_timeout_chain",
    "prior_timeout_same_script",
    "prior_timeout_chain_gap_s",
    "prior_timeout_row_id",
    "prior_duplicate_workload",
    "prior_duplicate_gap_s",
    "prior_duplicate_runtime_delta_s",
    "prior_duplicate_energy_sum_j",
    "prior_duplicate_same_script",
    "prior_duplicate_same_shape",
)

SOURCE_COLUMNS = (
    "row_id", "job_id", "user_hash", "submit_time", "end_time",
    "elapsed_s", "timelimit_s", "state",
)

KESTREL_COLUMNS = (
    "script_hash", "partition", "qos", "cpus_req", "mem_req",
    "gpus_req", "nodes_req", "energy_j",
)


@dataclass(frozen=True)
class DomainFeatureConfig:
    """Configuration recorded with a domain-feature cache."""

    history_window: int = HISTORY_WINDOW
    dataset: str = "unknown"
    rule: str = "safe"


def feature_availability(dataset: str) -> dict[str, bool | str]:
    """Return explicit capability flags for the initial feature slice."""
    # These features are available on all three supported snapshots.  The
    # Kestrel-only detector features are intentionally not marked available
    # until they are joined by the D2/D4 integration.
    return {
        "request_habits": True,
        "failure_history": True,
        "timeout_history": True,
        "kestrel_restart_history": False,
        "kestrel_duplicate_history": False,
        "queue_history": False,
        "dataset": dataset,
    }


def _require(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"domain feature input is missing columns: {missing}")


def _history_stream(frame: pl.DataFrame, window: int) -> pl.DataFrame:
    """Build rolling values at each legal historical end point."""
    if window < 2:
        raise ValueError("history_window must be at least 2")

    ordered = frame.select(list(SOURCE_COLUMNS)).sort(
        ["user_hash", "end_time", "job_id"]
    )
    failed = pl.col("state").is_in(FAILURE_STATES).cast(pl.Int8)
    timeout = (pl.col("state") == "TIMEOUT").cast(pl.Int8)

    return ordered.with_columns([
        pl.col("elapsed_s").rolling_quantile(
            0.50, interpolation="linear", window_size=window,
            min_samples=1,
        ).over("user_hash").alias("_elapsed_p50"),
        pl.col("elapsed_s").rolling_quantile(
            0.95, interpolation="linear", window_size=window,
            min_samples=1,
        ).over("user_hash").alias("_elapsed_p95"),
        pl.col("elapsed_s").rolling_mean(
            window, min_samples=1,
        ).over("user_hash").alias("_elapsed_mean"),
        pl.col("timelimit_s").rolling_quantile(
            0.95, interpolation="linear", window_size=window,
            min_samples=1,
        ).over("user_hash").alias("_timelimit_p95"),
        failed.rolling_sum(window, min_samples=1).over("user_hash").alias("_fail_n"),
        failed.rolling_sum(5, min_samples=1).over("user_hash").alias("_fail_5"),
        failed.rolling_sum(10, min_samples=1).over("user_hash").alias("_fail_10"),
        timeout.rolling_sum(5, min_samples=1).over("user_hash").alias("_timeout_5"),
        timeout.rolling_sum(10, min_samples=1).over("user_hash").alias("_timeout_10"),
        pl.when(failed == 1).then(pl.col("end_time")).otherwise(None)
        .forward_fill().over("user_hash").alias("_last_failure_end"),
        pl.when(timeout == 1).then(pl.col("end_time")).otherwise(None)
        .forward_fill().over("user_hash").alias("_last_timeout_end"),
    ])


def _with_optional_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Add capability columns as nulls when a dataset does not provide them."""
    out = frame
    specs = {
        "script_hash": pl.String,
        "partition": pl.String,
        "qos": pl.String,
        "cpus_req": pl.Int64,
        "mem_req": pl.Float64,
        "gpus_req": pl.Int64,
        "nodes_req": pl.Int64,
        "energy_j": pl.Float64,
    }
    for name, dtype in specs.items():
        if name not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(name))
        else:
            out = out.with_columns(pl.col(name).cast(dtype, strict=False))
    return out


def build_domain_features(
    frame: pl.DataFrame,
    *,
    dataset: str = "unknown",
    config: DomainFeatureConfig | None = None,
) -> pl.DataFrame:
    """Build one submit-time-safe domain row for every canonical row.

    ``end_time < submit_time`` is enforced by joining targets at
    ``submit_time - 1 microsecond``.  The target's own outcome and energy are
    never read when constructing its domain features.
    """
    _require(frame, SOURCE_COLUMNS)
    frame = _with_optional_columns(frame)
    config = config or DomainFeatureConfig(dataset=dataset)
    if config.rule != "safe":
        raise ValueError("domain features only support the safe rule")

    targets = frame.select([
        "row_id", "job_id", "user_hash", "submit_time", "timelimit_s",
        *KESTREL_COLUMNS,
    ]).with_columns(
        (pl.col("submit_time") - pl.duration(microseconds=1)).alias("_submit_prev")
    ).sort(["user_hash", "_submit_prev", "job_id"])

    history = _history_stream(frame, config.history_window).select([
        "user_hash", "end_time", "job_id", "elapsed_s", "timelimit_s",
        "_elapsed_p50", "_elapsed_p95", "_elapsed_mean", "_timelimit_p95",
        "_fail_5", "_fail_10", "_timeout_5", "_timeout_10",
        "_last_failure_end", "_last_timeout_end",
    ]).rename({"end_time": "_hist_end_time", "job_id": "_hist_job_id"})

    joined = targets.join_asof(
        history.sort(["user_hash", "_hist_end_time", "_hist_job_id"]),
        left_on="_submit_prev", right_on="_hist_end_time", by="user_hash",
        strategy="backward",
    )

    # D2: latest prior timeout for the same user and requested time limit.
    # Matching on the limit is the restart-chain definition; script equality is
    # retained as evidence rather than used to hide a changed script.
    timeout_prior = (
        frame.filter(pl.col("state") == "TIMEOUT")
        .select(["user_hash", "timelimit_s", "end_time", "row_id", "script_hash"])
        .rename({
            "end_time": "_d2_end", "row_id": "_d2_row_id",
            "script_hash": "_d2_script_hash",
        })
        .sort(["user_hash", "timelimit_s", "_d2_end", "_d2_row_id"])
    )
    joined = joined.join_asof(
        timeout_prior,
        left_on="_submit_prev", right_on="_d2_end",
        by=["user_hash", "timelimit_s"], strategy="backward",
    ).with_columns([
        (pl.col("submit_time") - pl.col("_d2_end")).dt.total_seconds()
        .alias("_d2_gap_s"),
        (pl.col("script_hash") == pl.col("_d2_script_hash"))
        .alias("_d2_same_script"),
    ])

    # D4: latest prior same-user/script/resource-shape workload.  The exact
    # shape is intentional: a matching script with a different allocation is
    # not evidence of duplicate work.
    shape = [
        "user_hash", "script_hash", "partition", "qos", "cpus_req",
        "mem_req", "gpus_req", "nodes_req", "timelimit_s",
    ]
    duplicate_prior = (
        frame.select([*shape, "end_time", "row_id", "elapsed_s", "energy_j"])
        .rename({
            "end_time": "_d4_end", "row_id": "_d4_row_id",
            "elapsed_s": "_d4_elapsed_s", "energy_j": "_d4_energy_j",
        })
        .filter(pl.col("script_hash").is_not_null())
        .sort([*shape, "_d4_end", "_d4_row_id"])
    )
    joined = joined.join_asof(
        duplicate_prior,
        left_on="_submit_prev", right_on="_d4_end", by=shape,
        strategy="backward",
    ).with_columns([
        (pl.col("submit_time") - pl.col("_d4_end")).dt.total_seconds()
        .alias("_d4_gap_s"),
        (pl.col("elapsed_s") - pl.col("_d4_elapsed_s")).abs()
        .alias("_d4_runtime_delta_s"),
        pl.when(pl.col("_d4_elapsed_s") > 0)
        .then((pl.col("elapsed_s") - pl.col("_d4_elapsed_s")).abs()
              / pl.col("_d4_elapsed_s"))
        .otherwise(float("inf")).alias("_d4_runtime_delta_ratio"),
    ])

    history_available = pl.col("_hist_end_time").is_not_null()
    elapsed_p95 = pl.col("_elapsed_p95")
    timelimit_p95 = pl.col("_timelimit_p95")
    return joined.with_columns([
        history_available.cast(pl.Int8).alias("domain_history_available"),
        pl.col("_elapsed_p50").cast(pl.Float64).alias("hist_elapsed_p50_s"),
        elapsed_p95.cast(pl.Float64).alias("hist_elapsed_p95_s"),
        pl.col("_elapsed_mean").cast(pl.Float64).alias("hist_elapsed_mean_s"),
        timelimit_p95.cast(pl.Float64).alias("hist_timelimit_p95_s"),
        pl.when(elapsed_p95 > 0)
        .then(pl.col("timelimit_s") / elapsed_p95)
        .otherwise(None).cast(pl.Float64).alias("request_vs_elapsed_p95"),
        (pl.col("timelimit_s") - elapsed_p95).cast(pl.Float64)
        .alias("request_excess_over_elapsed_p95_s"),
        pl.when(elapsed_p95 > 0)
        .then((pl.col("timelimit_s") > elapsed_p95).cast(pl.Int8))
        .otherwise(None).alias("request_exceeds_elapsed_p95"),
        pl.when(timelimit_p95 > 0)
        .then(pl.col("timelimit_s") / timelimit_p95)
        .otherwise(None).cast(pl.Float64)
        .alias("timelimit_vs_hist_timelimit_p95"),
        pl.col("_fail_5").fill_null(0).cast(pl.Int64).alias("failures_last_5"),
        pl.col("_fail_10").fill_null(0).cast(pl.Int64).alias("failures_last_10"),
        pl.col("_timeout_5").fill_null(0).cast(pl.Int64).alias("timeouts_last_5"),
        pl.col("_timeout_10").fill_null(0).cast(pl.Int64).alias("timeouts_last_10"),
        (pl.col("submit_time") - pl.col("_last_failure_end"))
        .dt.total_seconds().cast(pl.Float64).alias("time_since_last_failure_s"),
        (pl.col("submit_time") - pl.col("_last_timeout_end"))
        .dt.total_seconds().cast(pl.Float64).alias("time_since_last_timeout_s"),
        pl.when(pl.col("script_hash").is_not_null())
        .then((
            pl.col("_d2_gap_s").is_between(0.0, 2 * 3600, closed="both")
            & pl.col("_d2_row_id").is_not_null()
        ).cast(pl.Int8)).otherwise(None)
        .alias("prior_timeout_chain"),
        pl.when(pl.col("_d2_row_id").is_not_null())
        .then(pl.col("_d2_same_script")).otherwise(None)
        .alias("prior_timeout_same_script"),
        pl.when(pl.col("_d2_row_id").is_not_null())
        .then(pl.col("_d2_gap_s")).otherwise(None)
        .cast(pl.Float64).alias("prior_timeout_chain_gap_s"),
        pl.col("_d2_row_id").cast(pl.Int64).alias("prior_timeout_row_id"),
        pl.when(pl.col("script_hash").is_not_null())
        .then((
            pl.col("_d4_gap_s").is_between(0.0, 24 * 3600, closed="both")
            & (pl.col("_d4_runtime_delta_ratio") <= 0.10)
            & pl.col("_d4_row_id").is_not_null()
        ).cast(pl.Int8)).otherwise(None)
        .alias("prior_duplicate_workload"),
        pl.when(pl.col("_d4_row_id").is_not_null())
        .then(pl.col("_d4_gap_s")).otherwise(None).cast(pl.Float64)
        .alias("prior_duplicate_gap_s"),
        pl.when(pl.col("_d4_row_id").is_not_null())
        .then(pl.col("_d4_runtime_delta_s")).otherwise(None).cast(pl.Float64)
        .alias("prior_duplicate_runtime_delta_s"),
        pl.when(pl.col("_d4_row_id").is_not_null())
        .then(pl.col("_d4_energy_j")).otherwise(None).cast(pl.Float64)
        .alias("prior_duplicate_energy_sum_j"),
        pl.when(pl.col("script_hash").is_not_null() & pl.col("_d4_row_id").is_not_null())
        .then(pl.lit(True)).otherwise(None).alias("prior_duplicate_same_script"),
        pl.when(pl.col("script_hash").is_not_null() & pl.col("_d4_row_id").is_not_null())
        .then(pl.lit(True)).otherwise(None).alias("prior_duplicate_same_shape"),
    ]).select(["row_id", "job_id", "user_hash", "submit_time", *FEATURE_COLUMNS])


def validate_domain_features(features: pl.DataFrame, canonical: pl.DataFrame) -> None:
    """Validate row alignment before a D learner joins the feature cache."""
    _require(features, ("row_id", "job_id", *FEATURE_COLUMNS))
    _require(canonical, ("row_id", "job_id"))
    if features.height != canonical.height:
        raise ValueError("domain features must contain one row per canonical row")
    if features["row_id"].n_unique() != features.height:
        raise ValueError("domain features contain duplicate row_id values")
    expected = canonical.select(["row_id", "job_id"]).sort("row_id")
    actual = features.select(["row_id", "job_id"]).sort("row_id")
    if not actual.equals(expected):
        raise ValueError("domain features are not aligned to canonical row IDs")


def ensure_domain_features(
    lf: pl.LazyFrame,
    canonical_path: str | Path,
    cache_dir: str | Path,
    *,
    dataset: str = "unknown",
    rebuild: bool = False,
    max_chunk_rows: int = common.WINDOW_MAX_CHUNK_ROWS,
    rule: str = "safe",
) -> Path:
    """Build or reuse a versioned domain-feature Parquet cache.

    The cache uses the shared canonical checksum/chunking machinery, but is not
    clustered by month: domain history is user-ordered and D2/D4 joins need
    each user's rows to remain together.  Metadata records the feature version,
    configuration and capability flags so an unavailable detector cannot be
    mistaken for a zero-valued feature.
    """
    if rule != "safe":
        raise ValueError("domain features only support the safe rule")
    config = DomainFeatureConfig(dataset=dataset, rule=rule)
    availability = feature_availability(dataset)
    return common._ensure_cached(
        lf, canonical_path, cache_dir,
        filename=f"{Path(canonical_path).stem}.{rule}.domain_features.parquet",
        version=DOMAIN_FEATURE_VERSION,
        rule=rule,
        builder=lambda frame: build_domain_features(
            frame, dataset=dataset, config=config),
        max_chunk_rows=max_chunk_rows,
        rebuild=rebuild,
        what="domain features",
        cluster_by=None,
        count_column=None,
        extra_meta={
            "domain_feature_version": DOMAIN_FEATURE_VERSION,
            "feature_columns": list(FEATURE_COLUMNS),
            "history_window": config.history_window,
            "dataset": dataset,
            "capabilities": availability,
        },
    )
