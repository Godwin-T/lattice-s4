"""
State normalisation (canonical_table.md §3.2).

sacct states can carry detail ('CANCELLED by 1234'); Kestrel ships a ready-made
`state_simple`. Both reduce to an uppercase token, then must be in the terminal
set to survive filtering.
"""
from __future__ import annotations

import polars as pl


def normalise_state(value: str | None) -> str:
    """'CANCELLED by 1234' -> 'CANCELLED'; '' / None -> ''."""
    if not value:
        return ""
    return value.strip().upper().split()[0] if value.strip() else ""


def state_expr(col: str) -> pl.Expr:
    """Polars expression: uppercase first whitespace-delimited token."""
    return (
        pl.col(col)
        .cast(pl.String, strict=False)
        .str.strip_chars()
        .str.to_uppercase()
        .str.extract(r"^(\S+)", 1)
    )
