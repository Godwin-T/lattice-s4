"""First capability-safe Peepalytics detectors.

These are rule detectors over already-ingested canonical rows.  They do not
look at future rows and they do not train a model.  The learner-based
prediction layer is added later; these outputs are the auditable findings that
feed T4/T5.
"""
from __future__ import annotations

import polars as pl

from .contract import FINDING_COLUMNS, validate_findings

FAILURE_STATES = ("FAILED", "OUT_OF_MEMORY", "NODE_FAIL")


def _require(frame: pl.DataFrame, *columns: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"detector input is missing columns: {missing}")


def _empty() -> pl.DataFrame:
    return pl.DataFrame({
        "row_id": pl.Series([], dtype=pl.Int64),
        "job_id": pl.Series([], dtype=pl.String),
        "detector": pl.Series([], dtype=pl.String),
        "detector_version": pl.Series([], dtype=pl.String),
        "finding_type": pl.Series([], dtype=pl.String),
        "score": pl.Series([], dtype=pl.Float64),
        "submit_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        "state": pl.Series([], dtype=pl.String),
        "energy_j": pl.Series([], dtype=pl.Float64),
        "energy_tier": pl.Series([], dtype=pl.String),
        "evidence_elapsed_s": pl.Series([], dtype=pl.Float64),
        "evidence_timelimit_s": pl.Series([], dtype=pl.Float64),
        "evidence_ratio": pl.Series([], dtype=pl.Float64),
        "evidence_queue_wait_s": pl.Series([], dtype=pl.Float64),
        "evidence_cpus_req": pl.Series([], dtype=pl.Int64),
        "evidence_cpus_used": pl.Series([], dtype=pl.Int64),
        "evidence_failure_state": pl.Series([], dtype=pl.String),
        "evidence_script_hash": pl.Series([], dtype=pl.String),
        "evidence_prior_row_id": pl.Series([], dtype=pl.Int64),
        "evidence_gap_s": pl.Series([], dtype=pl.Float64),
        "evidence_runtime_delta_s": pl.Series([], dtype=pl.Float64),
        "evidence_same_script": pl.Series([], dtype=pl.Boolean),
    })


def _findings(frame: pl.DataFrame, mask: pl.Expr, detector: str,
              version: str, finding_type: str, score: pl.Expr) -> pl.DataFrame:
    _require(frame, "row_id", "job_id", "submit_time", "state", "energy_j",
             "energy_tier", "elapsed_s", "timelimit_s", "queue_wait_s",
             "cpus_req", "cpus_used", "script_hash")
    out = frame.filter(mask).select([
        pl.col("row_id"), pl.col("job_id"), pl.lit(detector).alias("detector"),
        pl.lit(version).alias("detector_version"), pl.lit(finding_type).alias("finding_type"),
        score.cast(pl.Float64).alias("score"), pl.col("submit_time"), pl.col("state"),
        pl.col("energy_j"), pl.col("energy_tier"),
        pl.col("elapsed_s").cast(pl.Float64).alias("evidence_elapsed_s"),
        pl.col("timelimit_s").cast(pl.Float64).alias("evidence_timelimit_s"),
        (pl.col("elapsed_s") / pl.col("timelimit_s")).cast(pl.Float64).alias("evidence_ratio"),
        pl.col("queue_wait_s").cast(pl.Float64).alias("evidence_queue_wait_s"),
        pl.col("cpus_req").cast(pl.Int64).alias("evidence_cpus_req"),
        pl.col("cpus_used").cast(pl.Int64).alias("evidence_cpus_used"),
        pl.when(pl.col("state").is_in(FAILURE_STATES)).then(pl.col("state"))
        .otherwise(pl.lit(None, dtype=pl.String)).alias("evidence_failure_state"),
        pl.col("script_hash").alias("evidence_script_hash"),
        pl.lit(None, dtype=pl.Int64).alias("evidence_prior_row_id"),
        pl.lit(None, dtype=pl.Float64).alias("evidence_gap_s"),
        pl.lit(None, dtype=pl.Float64).alias("evidence_runtime_delta_s"),
        pl.lit(None, dtype=pl.Boolean).alias("evidence_same_script"),
    ]).select(FINDING_COLUMNS)
    validate_findings(out)
    return out


def detect_d1_timeout(frame: pl.DataFrame) -> pl.DataFrame:
    """Emit one finding for each observed TIMEOUT job."""
    return _findings(
        frame, pl.col("state") == "TIMEOUT", "D1", "d1-v1",
        "timeout_waste", pl.lit(1.0),
    )


def detect_d3_failure(frame: pl.DataFrame) -> pl.DataFrame:
    """Emit one finding for each observed hard failure state."""
    return _findings(
        frame, pl.col("state").is_in(FAILURE_STATES), "D3", "d3-v1",
        "repeated_failure", pl.lit(1.0),
    )


def detect_d5_wallclock(frame: pl.DataFrame, *, utilization_threshold: float = 0.25) -> pl.DataFrame:
    """Find jobs using at most ``utilization_threshold`` of their time limit.

    This is the wallclock-only D5 variant.  It excludes TIMEOUTs because a
    timeout consumed the requested wallclock by definition, and it requires
    positive finite request/elapsed values.  The score is the avoidable-looking
    wallclock fraction (1 - elapsed/timelimit), not a claim of saved energy.
    """
    if not 0 < utilization_threshold < 1:
        raise ValueError("utilization_threshold must be between 0 and 1")
    ratio = pl.col("elapsed_s") / pl.col("timelimit_s")
    mask = (
        pl.col("state") != "TIMEOUT"
        ) & pl.col("elapsed_s").is_not_null() & pl.col("timelimit_s").is_not_null() \
        & (pl.col("elapsed_s") >= 0) & (pl.col("timelimit_s") > 0) \
        & ratio.is_finite() & (ratio <= utilization_threshold)
    return _findings(
        frame, mask, "D5", "d5-wallclock-v1", "over_request_wallclock",
        (1.0 - ratio).clip(0.0, 1.0),
    )


def _pair_output(frame: pl.DataFrame, *, detector: str, version: str,
                 finding_type: pl.Expr, score: pl.Expr) -> pl.DataFrame:
    """Project pair-derived rows into the common finding contract."""
    out = frame.select([
        pl.col("row_id").cast(pl.Int64), pl.col("job_id").cast(pl.String),
        pl.lit(detector).alias("detector"), pl.lit(version).alias("detector_version"),
        finding_type.alias("finding_type"), score.cast(pl.Float64).alias("score"),
        pl.col("submit_time"), pl.col("state"), pl.col("energy_j"), pl.col("energy_tier"),
        pl.col("elapsed_s").cast(pl.Float64).alias("evidence_elapsed_s"),
        pl.col("timelimit_s").cast(pl.Float64).alias("evidence_timelimit_s"),
        (pl.col("elapsed_s") / pl.col("timelimit_s")).cast(pl.Float64).alias("evidence_ratio"),
        pl.col("queue_wait_s").cast(pl.Float64).alias("evidence_queue_wait_s"),
        pl.col("cpus_req").cast(pl.Int64).alias("evidence_cpus_req"),
        pl.col("cpus_used").cast(pl.Int64).alias("evidence_cpus_used"),
        pl.when(pl.col("state").is_in(FAILURE_STATES)).then(pl.col("state"))
        .otherwise(pl.lit(None, dtype=pl.String)).alias("evidence_failure_state"),
        pl.col("script_hash").alias("evidence_script_hash"),
        pl.col("_prior_row_id").cast(pl.Int64).alias("evidence_prior_row_id"),
        pl.col("_gap_s").cast(pl.Float64).alias("evidence_gap_s"),
        pl.col("_runtime_delta_s").cast(pl.Float64).alias("evidence_runtime_delta_s"),
        pl.col("_same_script").cast(pl.Boolean).alias("evidence_same_script"),
    ]).select(FINDING_COLUMNS)
    validate_findings(out)
    return out


def detect_d2_restart_chain(frame: pl.DataFrame, *, max_gap_s: float = 2 * 3600) -> pl.DataFrame:
    """Link Kestrel TIMEOUTs to the next same-user, same-limit job.

    Kestrel has script hashes but no usable array/dependency fields.  A pair is
    a chain when the next submission occurs after the timeout ends and within
    two hours, with an identical requested time limit.  Matching script hashes
    are labelled ``intentional_checkpoint``; other pairs are labelled
    ``repeated_failure``.  The finding is keyed by the later job and carries
    the timeout's row ID as evidence.
    """
    _require(frame, "row_id", "job_id", "user_hash", "submit_time", "end_time",
             "state", "timelimit_s", "script_hash", "energy_j", "energy_tier",
             "elapsed_s", "queue_wait_s", "cpus_req", "cpus_used")
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    if frame["script_hash"].null_count() == frame.height:
        raise ValueError("D2 requires Kestrel script_hash values")
    keys = ["user_hash", "timelimit_s"]
    ordered = frame.sort([*keys, "submit_time", "job_id"]).with_columns([
        pl.col("row_id").shift(1).over(keys).alias("_prior_row_id"),
        pl.col("state").shift(1).over(keys).alias("_prior_state"),
        pl.col("end_time").shift(1).over(keys).alias("_prior_end_time"),
        pl.col("script_hash").shift(1).over(keys).alias("_prior_script_hash"),
    ]).with_columns([
        (pl.col("submit_time") - pl.col("_prior_end_time")).dt.total_seconds()
        .alias("_gap_s"),
        (pl.col("script_hash") == pl.col("_prior_script_hash")).alias("_same_script"),
    ])
    mask = (
        pl.col("_prior_state") == "TIMEOUT"
    ) & pl.col("_prior_row_id").is_not_null() \
      & pl.col("_gap_s").is_between(0, max_gap_s, closed="both") \
      & pl.col("script_hash").is_not_null() \
      & pl.col("_prior_script_hash").is_not_null()
    pairs = ordered.filter(mask).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("_runtime_delta_s")
    )
    return _pair_output(
        pairs, detector="D2", version="d2-kestrel-v1",
        finding_type=pl.when(pl.col("_same_script")).then(pl.lit("intentional_checkpoint"))
        .otherwise(pl.lit("repeated_failure")),
        score=pl.lit(1.0),
    )


def detect_d4_duplicate(frame: pl.DataFrame, *, window_s: float = 24 * 3600,
                        runtime_tolerance: float = 0.10) -> pl.DataFrame:
    """Find near-identical Kestrel workloads within a submission window.

    The resource shape is the exact tuple of partition, QoS, requested CPUs,
    memory, GPUs, nodes, and time limit.  A later job is a duplicate when its
    nearest prior same-user/script/shape job is within ``window_s`` and its
    runtime differs by no more than ``runtime_tolerance`` proportionally.  One
    finding is emitted per later job, avoiding duplicate evidence rows.
    """
    _require(frame, "row_id", "job_id", "user_hash", "script_hash", "submit_time",
             "state", "energy_j", "energy_tier", "elapsed_s", "timelimit_s",
             "queue_wait_s", "cpus_req", "cpus_used", "partition", "qos",
             "mem_req", "gpus_req", "nodes_req")
    if window_s <= 0 or runtime_tolerance < 0:
        raise ValueError("window_s must be positive and runtime_tolerance non-negative")
    if frame["script_hash"].null_count() == frame.height:
        raise ValueError("D4 requires Kestrel script_hash values")
    keys = ["user_hash", "script_hash", "partition", "qos", "cpus_req",
            "mem_req", "gpus_req", "nodes_req", "timelimit_s"]
    ordered = frame.filter(pl.col("script_hash").is_not_null()).sort(
        [*keys, "submit_time", "job_id"]
    ).with_columns([
        pl.col("row_id").shift(1).over(keys).alias("_prior_row_id"),
        pl.col("submit_time").shift(1).over(keys).alias("_prior_submit_time"),
        pl.col("elapsed_s").shift(1).over(keys).alias("_prior_elapsed_s"),
    ]).with_columns([
        (pl.col("submit_time") - pl.col("_prior_submit_time")).dt.total_seconds()
        .alias("_gap_s"),
        (pl.col("elapsed_s") - pl.col("_prior_elapsed_s")).abs().alias("_runtime_delta_s"),
    ]).with_columns([
        pl.when(pl.col("_prior_elapsed_s") > 0)
        .then(pl.col("_runtime_delta_s") / pl.col("_prior_elapsed_s"))
        .otherwise(pl.lit(float("inf"))).alias("_runtime_delta_ratio"),
    ])
    mask = (
        pl.col("_prior_row_id").is_not_null()
    ) & pl.col("_gap_s").is_between(0, window_s, closed="both") \
      & pl.col("elapsed_s").is_not_null() & (pl.col("elapsed_s") >= 0) \
      & pl.col("_prior_elapsed_s").is_not_null() & (pl.col("_prior_elapsed_s") >= 0) \
      & (pl.col("_runtime_delta_ratio") <= runtime_tolerance)
    pairs = ordered.filter(mask).with_columns(
        pl.lit(True, dtype=pl.Boolean).alias("_same_script")
    )
    return _pair_output(
        pairs, detector="D4", version="d4-kestrel-v1",
        finding_type=pl.lit("duplicate_workload"), score=pl.lit(1.0),
    )
