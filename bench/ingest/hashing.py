"""
Identifier hashing (canonical_table.md §3.1).

Only sources that ship RAW ids need this. Eagle 11M has synthetic ids
(`user0001`); the Eagle 3-month JSON and Kestrel already ship hashed ids, so
their adapters pass those through unchanged.

The salt is fixed per dataset and supplied at runtime (never committed), so
hashes are stable across arms and runs without being recomputable by anyone who
only has the table.
"""
from __future__ import annotations

import hashlib
import os

import polars as pl

ENV_SALT_TEMPLATE = "BENCH_SALT_{dataset}"   # e.g. BENCH_SALT_EAGLE


def resolve_salt(dataset: str, explicit: str | None = None) -> str | None:
    """
    Find the salt for a dataset: explicit argument, then BENCH_SALT_<DATASET>,
    then BENCH_SALT. Returns None if none is set (callers that need raw-id
    hashing must then refuse).
    """
    if explicit:
        return explicit
    env = ENV_SALT_TEMPLATE.format(dataset=dataset.upper())
    return os.environ.get(env) or os.environ.get("BENCH_SALT")


def salted_hash(value: str | None, salt: str, length: int = 16) -> str | None:
    """sha256(salt + value) truncated to `length` hex chars."""
    if value is None:
        return None
    return hashlib.sha256((salt + value).encode("utf-8")).hexdigest()[:length]


def hash_expr(col: str, salt: str, length: int = 16) -> pl.Expr:
    """
    Polars expression hashing a string column. Uses map_elements (Python)
    because a salted sha256 is not a native Polars op; acceptable at these
    sizes, and the 11M-row Eagle path is the only heavy user.
    """
    def _h(v):
        return salted_hash(v, salt, length)

    return pl.col(col).cast(pl.String, strict=False).map_elements(_h, return_dtype=pl.String)


def hash_column(series: pl.Series, salt: str, length: int = 16) -> pl.Series:
    """
    Hash a whole identifier column.

    Built as a value -> hash map rather than a per-row function: identifier
    cardinality is tiny (Eagle 11M has 936 users), so we hash each distinct
    value once and replace. That is orders of magnitude faster than mapping
    over 11M rows, and the result is identical (the hash depends only on the
    value, so chunking cannot change it).
    """
    uniques = series.drop_nulls().unique().to_list()
    mapping = {value: salted_hash(value, salt, length) for value in uniques}
    return series.replace_strict(mapping, default=None, return_dtype=pl.String)
