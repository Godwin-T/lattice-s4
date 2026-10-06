"""
Eligible users (splits.md T3, S4).

A user needs at least `MIN_HISTORY_JOBS` jobs before any of theirs can be
scored: the window is 24 previous jobs plus the one being predicted. Users below
the floor are dropped for every arm equally, so the scored population is
identical across the benchmark.

The set is shipped as a count plus a hash rather than a list of ids (S4): it is
recomputable from the canonical table, and a hash is enough for a consumer to
check it reproduced the same set.
"""
from __future__ import annotations

import hashlib

import polars as pl

from .config import MIN_HISTORY_JOBS


def eligible_users(lf: pl.LazyFrame, min_jobs: int = MIN_HISTORY_JOBS) -> pl.DataFrame:
    """Users with at least `min_jobs` jobs: columns `user_hash`, `len`."""
    return (
        lf.group_by("user_hash").len()
        .filter(pl.col("len") >= min_jobs)
        .sort("user_hash")
        .collect()
    )


def hash_ids(ids) -> str:
    """A stable sha256 over a sorted id list, one id per line."""
    h = hashlib.sha256()
    for value in sorted(ids):
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def eligible_summary(users: pl.DataFrame) -> dict:
    """`{count, sha256}` for a frame of eligible users (S4)."""
    ids = users["user_hash"].to_list()
    return {"count": len(ids), "sha256": hash_ids(ids)}
