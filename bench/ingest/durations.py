"""
Duration parsing (canonical_table.md §3.4).

Four encodings are in play across the sources:
    seconds            float, already seconds           36000.0
    iso8601            P{d}DT{h}H{m}M{s}S                P0DT0H30M0S
    slurm              [DD-]HH:MM:SS                     1-00:00:00
    arrow_duration_ns  Arrow duration[ns]                43_200_000_000_000

`duration_expr` returns a Polars expression producing Float64 seconds; the
`parse_*` helpers do the same for a single Python value (used by tests and by
any row-by-row path).
"""
from __future__ import annotations

import re

import polars as pl

KINDS = ("seconds", "iso8601", "slurm", "arrow_duration_ns")

# ISO-8601 duration, tolerant of omitted day/hour/minute/second parts.
_ISO = re.compile(
    r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$"
)
# Slurm wall-clock: [DD-]HH:MM:SS (hours optional, seconds may be fractional).
_SLURM = re.compile(
    r"^(?:(?P<d>\d+)-)?(?:(?P<h>\d+):)?(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)$"
)


def parse_iso8601(value: str | None) -> float | None:
    """'P0DT0H30M0S' -> 1800.0. Returns None if unparseable."""
    if not value:
        return None
    m = _ISO.match(value.strip())
    if not m:
        return None
    g = m.groupdict()
    return (int(g["d"] or 0) * 86400 + int(g["h"] or 0) * 3600
            + int(g["m"] or 0) * 60 + float(g["s"] or 0))


def parse_slurm(value: str | None) -> float | None:
    """'1-00:00:00' -> 86400.0, '04:00:00' -> 14400.0, '30:00' -> 1800.0."""
    if not value:
        return None
    m = _SLURM.match(value.strip())
    if not m:
        return None
    g = m.groupdict()
    return (int(g["d"] or 0) * 86400 + int(g["h"] or 0) * 3600
            + int(g["m"]) * 60 + float(g["s"]))


def _from_groups(col: str, pattern: str, weights: dict[str, float]) -> pl.Expr:
    """Sum regex-named-group captures, each multiplied by its weight."""
    groups = pl.col(col).str.extract_groups(pattern)
    total: pl.Expr | None = None
    for name, weight in weights.items():
        term = groups.struct.field(name).cast(pl.Float64, strict=False).fill_null(0.0) * weight
        total = term if total is None else total + term
    return total


def duration_expr(col: str, kind: str) -> pl.Expr:
    """
    A Polars expression converting column `col` to seconds (Float64).

    Null input yields null, so the caller's null-filter still applies. An
    unparseable non-null string also yields null -- it is never silently 0.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown duration kind {kind!r}; expected one of {KINDS}")

    if kind == "seconds":
        return pl.col(col).cast(pl.Float64, strict=False)

    if kind == "iso8601":
        expr = _from_groups(col, _ISO.pattern,
                            {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0})
        # Preserve nulls: a null input must stay null, not become 0.
        return pl.when(pl.col(col).is_null() | (pl.col(col) == "")).then(None).otherwise(expr)

    if kind == "slurm":
        expr = _from_groups(col, _SLURM.pattern,
                            {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0})
        return pl.when(pl.col(col).is_null() | (pl.col(col) == "")).then(None).otherwise(expr)

    # Arrow duration[ns] -> seconds.
    return pl.col(col).dt.total_nanoseconds().cast(pl.Float64) / 1e9
