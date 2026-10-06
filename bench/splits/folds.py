"""
Forward-chained folds (splits.md T4).

Expanding window: for each open month, train on every month before it and test
on that month. Never reversed, never shuffled. A fold is kept only if it clears
the guardrails; if fewer than `MIN_FOLDS` survive, the dataset cannot support an
honest answer and the splitter refuses.

Month-level row/positive counts are computed once and then accumulated, so
building 50 folds costs one scan, not fifty.
"""
from __future__ import annotations

import polars as pl

from .config import (
    LOCKED_MONTHS, MIN_FOLDS, MIN_TEST_POS, MIN_TEST_ROWS,
    MIN_TRAIN_POS, MIN_TRAIN_ROWS, POSITIVE_STATE,
)
from .months import month_expr


def month_stats(lf: pl.LazyFrame,
                positive_state: str = POSITIVE_STATE) -> dict[str, dict]:
    """`{month: {"rows": n, "pos": p}}` for the frame it is given."""
    df = (
        lf.with_columns(month_expr())
        .group_by("month")
        .agg(pl.len().alias("rows"),
             (pl.col("state") == positive_state).sum().alias("pos"))
        .collect()
    )
    return {m: {"rows": int(r), "pos": int(p)}
            for m, r, p in df.iter_rows()}


def build_folds(months: list[str], stats: dict[str, dict]) -> tuple[list[dict], list[str], list[str]]:
    """
    Return `(folds, locked_months, open_months)`.

    Raises ValueError if fewer than `MIN_FOLDS` folds survive.
    """
    if len(months) < LOCKED_MONTHS + 1:
        raise ValueError(
            f"only {len(months)} usable month(s); need more than {LOCKED_MONTHS} "
            "to leave anything for training before the holdout"
        )
    locked = months[-LOCKED_MONTHS:]
    open_months = months[:-LOCKED_MONTHS]

    folds: list[dict] = []
    for i in range(1, len(open_months)):
        train_months = open_months[:i]
        test_month = open_months[i]
        n_train = sum(stats[m]["rows"] for m in train_months)
        n_test = stats[test_month]["rows"]
        pos_train = sum(stats[m]["pos"] for m in train_months)
        pos_test = stats[test_month]["pos"]
        if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
            continue
        if pos_train < MIN_TRAIN_POS or pos_test < MIN_TEST_POS:
            continue
        folds.append({
            "fold_id": len(folds) + 1,
            "test_month": test_month,
            "train_months": list(train_months),
            "n_train": n_train,
            "n_test": n_test,
            "pos_train": pos_train,
            "pos_test": pos_test,
        })

    if len(folds) < MIN_FOLDS:
        raise ValueError(
            f"only {len(folds)} usable fold(s); at least {MIN_FOLDS} are needed. "
            "The export is too short or too sparse for a forward-chained test."
        )
    return folds, locked, open_months
