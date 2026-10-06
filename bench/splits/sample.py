"""
The frozen 20,000-job sample (splits.md T5, S2, S3).

Every arm is additionally scored on the same 20,000 test jobs. The sample is:

* drawn from the **open test months only** (never the locked holdout);
* stratified by `state x activity_quartile`;
* allocated proportionally, with a floor so rare strata are not starved (S2);
* chosen deterministically, so two runs give the same ids.

Activity quartiles are computed once over the whole dataset on `log1p(jobs per
user)` (S3): per-month quartiles would let the same user move between strata as
the timeline advances.
"""
from __future__ import annotations

import polars as pl

from .config import ACTIVITY_QUARTILES, SAMPLE_FLOOR, SAMPLE_N
from .eligibility import hash_ids
from .months import month_expr


def with_activity_quartile(users: pl.DataFrame) -> pl.DataFrame:
    """
    Add `activity_quartile` (1 = quietest quarter of users ... 4 = busiest) to a
    frame of eligible users with a `len` column.
    """
    n = users.height
    if n == 0:
        return users.with_columns(pl.lit(None, dtype=pl.Int8).alias("activity_quartile"))
    return (
        users.sort(["len", "user_hash"])
        .with_columns(
            (((pl.int_range(pl.len()) * ACTIVITY_QUARTILES) // n) + 1)
            .cast(pl.Int8).alias("activity_quartile")
        )
    )


def allocate_quotas(sizes: dict[tuple, int], n: int = SAMPLE_N,
                    floor: int = SAMPLE_FLOOR) -> dict[tuple, int]:
    """
    Proportional allocation with a per-stratum floor, adjusted to sum to `n`
    where the strata have enough members to allow it (S2).
    """
    total = sum(sizes.values())
    if total <= n:                       # tiny dataset: take everything
        return dict(sizes)

    quota = {k: min(v, max(floor, int(v * n / total))) for k, v in sizes.items()}

    # Trim over-allocation from the largest quotas, never below the floor.
    while sum(quota.values()) > n:
        candidates = [k for k, q in quota.items() if q > floor and q > 0]
        if not candidates:
            break
        k = max(candidates, key=lambda k: quota[k])
        quota[k] -= 1

    # Give any remaining capacity to the stratum with the most unsampled members.
    while sum(quota.values()) < n:
        candidates = [k for k, q in quota.items() if q < sizes[k]]
        if not candidates:
            break                        # every stratum exhausted
        k = max(candidates, key=lambda k: sizes[k] - quota[k])
        quota[k] += 1

    return quota


def build_sample(lf: pl.LazyFrame,
                 test_months: list[str],
                 users_with_quartile: pl.DataFrame,
                 seed: int,
                 n: int = SAMPLE_N,
                 floor: int = SAMPLE_FLOOR) -> dict:
    """
    Draw the frozen sample and return a block ready for the manifest:
    `{n, drawn_from, stratify_by, activity_metric, strata[], job_ids,
      job_ids_sha256}`.
    """
    population = (
        lf.with_columns(month_expr())
        .filter(pl.col("month").is_in(test_months))
        .select(["job_id", "state", "user_hash"])
        .join(users_with_quartile.lazy().select(["user_hash", "activity_quartile"]),
              on="user_hash", how="inner")
        .collect()
    )

    sizes_df = population.group_by(["state", "activity_quartile"]).len()
    sizes = {(s, int(q)): int(c) for s, q, c in sizes_df.iter_rows()}
    quotas = allocate_quotas(sizes, n=n, floor=floor)

    quota_df = pl.DataFrame({
        "state": [k[0] for k in quotas],
        "activity_quartile": [k[1] for k in quotas],
        "quota": [v for v in quotas.values()],
    }).with_columns(pl.col("activity_quartile").cast(pl.Int8))

    # Deterministic per-stratum choice: rank by a seeded hash of the job id, so
    # the selection does not depend on row order or on chunking.
    ranked = (
        population
        .with_columns(pl.col("job_id").hash(seed=seed).alias("_key"))
        .with_columns(
            pl.col("_key").rank("ordinal")
            .over(["state", "activity_quartile"]).alias("_rank")
        )
        .join(quota_df, on=["state", "activity_quartile"], how="inner")
    )
    chosen = ranked.filter(pl.col("_rank") <= pl.col("quota"))
    job_ids = sorted(chosen["job_id"].to_list())

    taken = {(s, int(q)): int(c) for s, q, c in
             chosen.group_by(["state", "activity_quartile"]).len().iter_rows()}
    strata = [
        {"state": k[0], "activity_quartile": k[1],
         "quota": quotas[k], "taken": taken.get(k, 0), "available": sizes[k]}
        for k in sorted(sizes, key=lambda k: (str(k[0]), k[1]))
    ]

    return {
        "n": len(job_ids),
        "drawn_from": "open_test_months",
        "stratify_by": ["state", "activity_quartile"],
        "activity_metric": "jobs_per_user_log_quartile",
        "strata": strata,
        "job_ids_sha256": hash_ids(job_ids),
        "job_ids": job_ids,
    }
