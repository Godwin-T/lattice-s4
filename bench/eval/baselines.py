"""
The two naive baselines (metrics.md section 6.1).

Every trained arm must beat both. If a model cannot beat "flag the longest time
limits", it has not earned its complexity. Both use only information available
when the job was submitted, so neither is a leak.
"""
from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ..arms.common import with_row_id
from .predictions import expected_test_rows


def _test_rows(canonical_path: str, manifest: dict) -> pl.LazyFrame:
    """
    The manifest's test rows, with the canonical columns the baselines need.

    Joined on `row_id`, which is unique. `job_id` is not — the canonical table
    holds repeated ids across attempts — so an inner join on it would both
    multiply rows and mis-assign a previous outcome to a later attempt.
    """
    return (
        with_row_id(pl.scan_parquet(canonical_path))
        .select(["row_id", "job_id", "user_hash", "submit_time",
                 "timelimit_s", "state"])
        .join(expected_test_rows(canonical_path, manifest)
              .select(["row_id", "test_month"]), on="row_id", how="inner")
    )


def repeat_last_outcome(canonical_path: str, manifest: dict,
                        positive_states: str | Iterable[str] = "TIMEOUT",
                        ) -> pl.DataFrame:
    """
    Score = 1 if this user's previous job ended in one of `positive_states`.

    Takes a *collection* of states, not one: T1's label is a single state
    (`TIMEOUT`) but T2's is three (`FAILED`, `OUT_OF_MEMORY`, `NODE_FAIL`), and a
    baseline that could only name one of them would silently answer a different
    question from the arm it is meant to be judged against — and, being easier,
    would make the arm look better than it is. `POSITIVE_STATES[task]` is passed
    straight through.

    Ties are broken by `job_id` at ranking time (metrics.md M5), which the
    downstream flagging handles naturally because ties are broken
    deterministically and identically on every run.
    """
    states = ([positive_states] if isinstance(positive_states, str)
              else list(positive_states))
    if not states:
        raise ValueError("positive_states is empty: the baseline would flag "
                         "nothing and every score would be 0")
    rows = _test_rows(canonical_path, manifest).sort(
        ["user_hash", "submit_time", "row_id"])
    rows = rows.with_columns(
        pl.col("state").shift(1).over("user_hash").alias("previous_state"))
    return rows.select([
        pl.col("row_id"),
        pl.col("job_id"),
        pl.col("test_month"),
        # The first test row of each user has no previous outcome. "No evidence"
        # is not "flag it", and a null would reach `auc` as NaN — which `argsort`
        # sorts *above* every real score, silently making the baseline look
        # maximally risky on exactly those rows. False, so it scores 0.
        pl.col("previous_state").is_in(sorted(states)).fill_null(False)
        .cast(pl.Float64).alias("score"),
    ]).collect()


def longest_time_limits(canonical_path: str, manifest: dict) -> pl.DataFrame:
    """Score = the time limit the job asked for. Dumb, and surprisingly hard to beat."""
    return _test_rows(canonical_path, manifest).select([
        pl.col("row_id"), pl.col("job_id"), pl.col("test_month"),
        # Same guard as `repeat_last_outcome`: an unset time limit is ranked
        # last, never NaN. Neither frozen dataset has such a row today, but a
        # null here would poison the ranking in exactly the same silent way.
        pl.col("timelimit_s").cast(pl.Float64).fill_null(0.0).alias("score"),
    ]).collect()
