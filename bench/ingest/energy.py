"""
Energy channels (canonical_table.md §3.5).

`energy_j` is filled only from the source's declared channel, and `energy_tier`
records which tier it is. Tiers are never mixed.
"""
from __future__ import annotations

import polars as pl

J_PER_KWH = 3_600_000.0


def measured_j(col: str) -> pl.Expr:
    """Slurm-measured joules, passed through as-is."""
    return pl.col(col).cast(pl.Float64, strict=False)


def modelled_j(avg_power_col: str, nodes_col: str, elapsed_s_col: str) -> pl.Expr:
    """avg_power(W) x nodes x elapsed(s) -> joules (the Eagle 3-month formula)."""
    return (
        pl.col(avg_power_col).cast(pl.Float64, strict=False)
        * pl.col(nodes_col).cast(pl.Float64, strict=False)
        * pl.col(elapsed_s_col).cast(pl.Float64, strict=False)
    )


def tier(value: str) -> pl.Expr:
    return pl.lit(value)


def tier_from(joules: pl.Expr, name: str) -> pl.Expr:
    """
    The tier for a row: the declared tier where a value exists, 'none' where it
    does not. Guarantees `energy_j is null  <=>  energy_tier == 'none'`, which
    validate.py checks.
    """
    return pl.when(joules.is_null()).then(pl.lit("none")).otherwise(pl.lit(name))
