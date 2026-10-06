"""
Month derivation and the month floor (splits.md S1).

A job's month comes from its **submit** time, never its end time: the no-leakage
rule is stated in submit terms, and a monthly source file can contain jobs
submitted outside that file's nominal month (Kestrel has 30 submit-months across
29 files).

Months below `MIN_MONTH_ROWS` are treated as artefacts and dropped from the
timeline *before* the holdout is chosen. The rows themselves are not deleted;
they simply take no part in the fold structure.
"""
from __future__ import annotations

import polars as pl

from .config import MIN_MONTH_ROWS


def month_expr() -> pl.Expr:
    """The `YYYY-MM` label of a row, from its submit time."""
    return pl.col("submit_time").dt.strftime("%Y-%m").alias("month")


def month_counts(lf: pl.LazyFrame) -> pl.DataFrame:
    """Rows per submit-month, ascending. Includes months below the floor."""
    return (
        lf.with_columns(month_expr())
        .group_by("month").len()
        .sort("month")
        .collect()
    )


def timeline(lf: pl.LazyFrame, min_rows: int = MIN_MONTH_ROWS):
    """
    Return `(months, dropped)`:

    * `months`  -- the usable submit-months, in order
    * `dropped` -- [(month, rows), ...] artefacts that fell below the floor
    """
    counts = month_counts(lf)
    keep = counts.filter(pl.col("len") >= min_rows)
    drop = counts.filter(pl.col("len") < min_rows)
    months = keep["month"].to_list()
    dropped = [(m, int(n)) for m, n in
               zip(drop["month"].to_list(), drop["len"].to_list())]
    return months, dropped
