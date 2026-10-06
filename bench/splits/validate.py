"""
Manifest validation (splits.md T6, folds_manifest.md section 8).

Run before writing: a manifest that fails any of these is a bug, and it is
better to crash than to hand arms a split that quietly breaks comparability.

Most checks need only the manifest. Two need the data:
  * "sum(n_test) equals the open test rows" needs per-month row counts;
  * "every sampled job lies in an open test month" needs the job -> month map.
Both are optional; when the caller supplies them, they are checked.
"""
from __future__ import annotations

import polars as pl

from .config import (
    MANIFEST_SCHEMA, MIN_TEST_POS, MIN_TEST_ROWS, MIN_TRAIN_POS, MIN_TRAIN_ROWS,
)
from .eligibility import hash_ids
from .months import month_expr


def check_manifest(manifest: dict,
                   month_counts: dict[str, int] | None = None,
                   lf: pl.LazyFrame | None = None) -> list[str]:
    """Return a list of problems; empty means the manifest is valid."""
    problems: list[str] = []
    folds = manifest.get("folds") or []
    locked = set(manifest.get("locked", {}).get("months") or [])

    if manifest.get("schema") != MANIFEST_SCHEMA:
        problems.append(f"unknown schema {manifest.get('schema')!r}")
    if not folds:
        problems.append("no folds present")
        return problems

    # --- fold structure ----------------------------------------------------
    test_months = [f["test_month"] for f in folds]
    if test_months != sorted(test_months):
        problems.append("test months are not in ascending order")
    if len(set(test_months)) != len(test_months):
        problems.append("duplicate test months")

    for f in folds:
        train = f["train_months"]
        if f["test_month"] in train:
            problems.append(f"fold {f['fold_id']}: test month inside train months")
        if any(m >= f["test_month"] for m in train):
            problems.append(f"fold {f['fold_id']}: train month not before test month")
        if locked & set(train) or locked & {f["test_month"]}:
            problems.append(f"fold {f['fold_id']}: holdout month used in a fold")
        if f["n_train"] < MIN_TRAIN_ROWS or f["n_test"] < MIN_TEST_ROWS:
            problems.append(f"fold {f['fold_id']}: below the row guardrail")
        if f["pos_train"] < MIN_TRAIN_POS or f["pos_test"] < MIN_TEST_POS:
            problems.append(f"fold {f['fold_id']}: below the positive guardrail")

    # --- sample ------------------------------------------------------------
    sample = manifest.get("sample") or {}
    ids = sample.get("job_ids") or []
    if sample.get("job_ids_sha256") != hash_ids(ids):
        problems.append("sample.job_ids_sha256 does not match the id list")
    if locked and ids and lf is not None:
        rows = (lf.filter(pl.col("job_id").is_in(ids))
                  .select(pl.col("job_id"), month_expr())
                  .collect())
        outside = rows.filter(~pl.col("month").is_in(set(test_months)))
        if outside.height:
            problems.append(
                f"{outside.height} sampled jobs are outside the open test months"
            )

    # --- reconciliation ----------------------------------------------------
    if month_counts is not None:
        expected = sum(month_counts.get(m, 0) for m in test_months)
        got = sum(f["n_test"] for f in folds)
        if expected != got:
            problems.append(
                f"sum(n_test)={got:,} does not equal the open test rows={expected:,}"
            )
    return problems


def assert_valid(manifest: dict, **kwargs) -> None:
    problems = check_manifest(manifest, **kwargs)
    if problems:
        raise ValueError("manifest failed validation:\n  - " + "\n  - ".join(problems))
