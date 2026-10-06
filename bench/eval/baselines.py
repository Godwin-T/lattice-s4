"""
The two naive baselines (metrics.md section 6.1).

Every trained arm must beat both. If a model cannot beat "flag the longest time
limits", it has not earned its complexity. Both use only information available
when the job was submitted, so neither is a leak.
"""
from __future__ import annotations

import polars as pl

from .predictions import expected_test_rows


def _test_rows(canonical_path: str, manifest: dict) -> pl.LazyFrame:
    return (
        pl.scan_parquet(canonical_path)
        .select(["job_id", "user_hash", "submit_time", "timelimit_s", "state"])
        .join(expected_test_rows(canonical_path, manifest), on="job_id", how="inner")
    )


def repeat_last_outcome(canonical_path: str, manifest: dict,
                        positive_state: str = "TIMEOUT") -> pl.DataFrame:
    """
    Score = 1 if this user's previous job timed out, else 0.

    Ties are broken by `job_id` at ranking time (metrics.md M5), which the
    downstream flagging handles naturally because ties are broken
    deterministically and identically on every run.
    """
    rows = _test_rows(canonical_path, manifest).sort(
        ["user_hash", "submit_time", "job_id"])
    rows = rows.with_columns(
        pl.col("state").shift(1).over("user_hash").alias("previous_state"))
    return rows.select([
        pl.col("job_id"),
        pl.col("test_month"),
        (pl.col("previous_state") == positive_state)
        .cast(pl.Float64).alias("score"),
    ]).collect()


def longest_time_limits(canonical_path: str, manifest: dict) -> pl.DataFrame:
    """Score = the time limit the job asked for. Dumb, and surprisingly hard to beat."""
    return _test_rows(canonical_path, manifest).select([
        pl.col("job_id"), pl.col("test_month"),
        pl.col("timelimit_s").cast(pl.Float64).alias("score"),
    ]).collect()
