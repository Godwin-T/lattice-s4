"""
Row drop rules (canonical_table.md §3.6), applied after an adapter has produced
canonical columns. Each drop is counted so the audit file reconciles.
"""
from __future__ import annotations

import polars as pl

from .schema import TERMINAL_STATES


def apply_filters(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Drop rows that cannot support an honest answer, counting each reason.

    Order matters only for attribution; the surviving set is the same.
    """
    stats: dict[str, int] = {"rows_in": df.height}

    def _drop(frame: pl.DataFrame, mask: pl.Expr, reason: str) -> pl.DataFrame:
        before = frame.height
        out = frame.filter(mask)
        stats[f"dropped_{reason}"] = before - out.height
        return out

    # 1. Only terminal states have an outcome to learn from.
    df = _drop(df, pl.col("state").is_in(TERMINAL_STATES), "non_terminal_state")

    # 2. Required fields must be present.
    for col, reason in (("submit_time", "null_submit"),
                        ("end_time", "null_end"),
                        ("user_hash", "null_user")):
        df = _drop(df, pl.col(col).is_not_null(), reason)

    # 3. A positive time limit is required for any ratio.
    df = _drop(df, pl.col("timelimit_s").is_not_null() & (pl.col("timelimit_s") > 0),
               "bad_timelimit")

    # 4. A missing used-duration means the job never ran (null end / cancelled).
    df = _drop(df, pl.col("elapsed_s").is_not_null(), "no_elapsed")

    stats["rows_out"] = df.height
    return df, stats
