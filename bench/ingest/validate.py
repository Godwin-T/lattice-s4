"""
canonical_table.md §6 acceptance tests, run on a canonical frame.

These are cheap structural checks; the reconciliation of row counts is done by
the caller (filters.apply_filters returns the counts, cli.py writes them).
"""
from __future__ import annotations

import polars as pl

from .schema import CANONICAL_COLUMNS, ENERGY_TIERS, FORBIDDEN_COLUMNS, TERMINAL_STATES


def run_checks(df: pl.DataFrame) -> list[str]:
    """Return a list of problems; empty means the frame is valid."""
    problems: list[str] = []

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing canonical columns: {missing}")
    extra = [c for c in df.columns if c not in CANONICAL_COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")

    # No sensitive/raw identifier columns may survive ingest.
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(df.columns))
    if forbidden:
        problems.append(f"forbidden columns present: {forbidden}")

    if "state" in df.columns:
        bad_states = sorted(set(df["state"].unique().to_list()) - set(TERMINAL_STATES))
        if bad_states:
            problems.append(f"non-terminal states present: {bad_states}")

    if "energy_tier" in df.columns:
        bad_tiers = sorted(set(df["energy_tier"].unique().to_list()) - set(ENERGY_TIERS))
        if bad_tiers:
            problems.append(f"unknown energy tiers: {bad_tiers}")

        # Invariant: energy_j is null exactly when the tier is "none".
        mismatch = df.filter(
            pl.col("energy_j").is_null() != (pl.col("energy_tier") == "none")
        ).height
        if mismatch:
            problems.append(
                f"{mismatch} rows where energy_j / energy_tier disagree "
                "(a value must exist iff the tier is not 'none')"
            )

    return problems


def assert_valid(df: pl.DataFrame) -> None:
    problems = run_checks(df)
    if problems:
        raise ValueError("canonical table failed validation:\n  - " + "\n  - ".join(problems))


def lazy_checks(lf: pl.LazyFrame) -> list[str]:
    """
    The same rules as `run_checks`, but for a LazyFrame that may be far too large
    to hold in memory. Only aggregates are collected, so this is safe on an
    11M-row table.
    """
    problems: list[str] = []
    cols = list(lf.collect_schema().names())

    missing = [c for c in CANONICAL_COLUMNS if c not in cols]
    if missing:
        problems.append(f"missing canonical columns: {missing}")
    extra = [c for c in cols if c not in CANONICAL_COLUMNS]
    if extra:
        problems.append(f"unexpected columns: {extra}")
    forbidden = sorted(FORBIDDEN_COLUMNS.intersection(cols))
    if forbidden:
        problems.append(f"forbidden columns present: {forbidden}")

    if {"state", "energy_tier", "energy_j"} <= set(cols):
        agg = lf.select([
            (~pl.col("state").is_in(TERMINAL_STATES)).sum().alias("bad_states"),
            (~pl.col("energy_tier").is_in(ENERGY_TIERS)).sum().alias("bad_tiers"),
            (pl.col("energy_j").is_null() != (pl.col("energy_tier") == "none"))
            .sum().alias("mismatch"),
        ]).collect().row(0, named=True)
        if agg["bad_states"]:
            problems.append(f"{agg['bad_states']} rows with a non-terminal state")
        if agg["bad_tiers"]:
            problems.append(f"{agg['bad_tiers']} rows with an unknown energy tier")
        if agg["mismatch"]:
            problems.append(
                f"{agg['mismatch']} rows where energy_j / energy_tier disagree"
            )
    return problems
