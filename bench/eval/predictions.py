"""
Load an arm's hand-in table, check it against the frozen split, and attach the
truth (metrics.md sections 2 and 7).

The contract is enforced, not assumed: if a fold's test rows are not each
scored exactly once, the run is refused. Silent gaps would make two arms look
like they answered the same question when they did not.
"""
from __future__ import annotations

import polars as pl

from ..splits.eligibility import eligible_users
from .config import POSITIVE_STATES, PREDICTION_COLUMNS


def load_predictions(path: str) -> pl.DataFrame:
    """Read a hand-in table from Parquet or CSV."""
    return (pl.read_parquet(path) if str(path).endswith(".parquet")
            else pl.read_csv(path, infer_schema_length=10_000))


def label_expr(task: str) -> pl.Expr:
    """The interesting outcome for a task, from the canonical `state`."""
    states = POSITIVE_STATES.get(task)
    if states is None:
        raise ValueError(f"no label defined for task {task!r}")
    return pl.col("state").is_in(sorted(states)).alias("label")


def expected_test_rows(canonical_path: str, manifest: dict) -> pl.LazyFrame:
    """
    Every row a run is required to score: the eligible population, restricted to
    the manifest's test months.
    """
    lf = pl.scan_parquet(canonical_path)
    users = eligible_users(lf)                      # same rule the arms apply
    months = [f["test_month"] for f in manifest["folds"]]
    return (
        lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
        .with_columns(pl.col("submit_time").dt.strftime("%Y-%m").alias("test_month"))
        .filter(pl.col("test_month").is_in(months))
        .select(["job_id", "test_month"])
    )


def check_contract(preds: pl.DataFrame, manifest: dict,
                   canonical_path: str) -> list[str]:
    """Return a list of contract violations; empty means the run is usable."""
    problems: list[str] = []

    missing = [c for c in PREDICTION_COLUMNS if c not in preds.columns]
    if missing:
        return [f"predictions are missing columns: {missing}"]
    if preds.height == 0:
        return ["predictions table is empty"]

    duplicates = preds.height - preds["job_id"].n_unique()
    if duplicates:
        problems.append(f"{duplicates} job(s) were scored more than once")

    known_months = {f["test_month"] for f in manifest["folds"]}
    stray = sorted(set(preds["test_month"].unique().to_list()) - known_months)
    if stray:
        problems.append(f"predictions mention months outside the manifest: {stray}")

    expected = expected_test_rows(canonical_path, manifest).collect()
    scored = preds.select(["job_id", "test_month"])

    extra = scored.join(expected, on=["job_id", "test_month"], how="anti")
    if extra.height:
        problems.append(f"{extra.height} predicted job(s) are not test rows")
    absent = expected.join(scored, on=["job_id", "test_month"], how="anti")
    if absent.height:
        problems.append(f"{absent.height} test row(s) were not scored at all")

    return problems


def attach_truth(preds: pl.DataFrame, canonical_path: str,
                 task: str) -> pl.DataFrame:
    """Join the label and the energy reading onto the predictions."""
    truth = pl.scan_parquet(canonical_path).select(
        ["job_id", "state", "energy_j", "energy_tier"])
    return (
        preds.lazy()
        .join(truth, on="job_id", how="left", coalesce=True)
        .with_columns(label_expr(task))
        .collect()
    )
