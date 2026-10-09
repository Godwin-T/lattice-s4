"""
Load an arm's hand-in table, check it against the frozen split, and attach the
truth (metrics.md sections 2 and 7).

The contract is enforced, not assumed: if a fold's test rows are not each
scored exactly once, the run is refused. Silent gaps would make two arms look
like they answered the same question when they did not.
"""
from __future__ import annotations

import polars as pl

from ..arms.common import with_row_id
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

    Keyed by `row_id`, not `job_id`. The manifest counts *rows* —
    `Σ fold.n_test` is the number of rows in the open test months, not the number
    of distinct ids — and `job_id` is not unique in the canonical table, so it
    cannot say whether a given test row was scored. `row_id` is derived here by
    the same rule the arms use (`arms.common.with_row_id`), from the same file.
    """
    lf = with_row_id(pl.scan_parquet(canonical_path))
    users = eligible_users(lf)                      # same rule the arms apply
    months = [f["test_month"] for f in manifest["folds"]]
    return (
        lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
        .with_columns(pl.col("submit_time").dt.strftime("%Y-%m").alias("test_month"))
        .filter(pl.col("test_month").is_in(months))
        .select(["row_id", "job_id", "test_month"])
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

    duplicates = preds.height - preds["row_id"].n_unique()
    if duplicates:
        problems.append(f"{duplicates} row(s) were scored more than once")

    known_months = {f["test_month"] for f in manifest["folds"]}
    stray = sorted(set(preds["test_month"].unique().to_list()) - known_months)
    if stray:
        problems.append(f"predictions mention months outside the manifest: {stray}")

    expected = expected_test_rows(canonical_path, manifest).collect()
    # `row_id` is unique on both sides, so these anti-joins count exactly: no
    # multiplication, and a row filed under the wrong month counts as both extra
    # and absent rather than passing. Keyed on `job_id` they would multiply —
    # one kestrel id covers 10,000 rows.
    key = ["row_id", "test_month"]
    scored = preds.select(key)

    extra = scored.join(expected.select(key), on=key, how="anti")
    if extra.height:
        problems.append(f"{extra.height} predicted row(s) are not test rows")
    absent = expected.select(key).join(scored, on=key, how="anti")
    if absent.height:
        problems.append(f"{absent.height} test row(s) were not scored at all")

    return problems


def attach_truth(preds: pl.DataFrame, canonical_path: str,
                 task: str, *, with_probability: bool = False) -> pl.DataFrame:
    """
    Join the label and the energy reading onto the predictions.

    On `row_id`, which is one-to-one. Joining the canonical table on `job_id`
    instead multiplies each prediction by the number of canonical rows sharing
    its id — up to 10,000 on kestrel — so a 7M-row hand-in table would not fit in
    memory.

    Both sides are projected to just the columns the evaluator reads, which is
    what keeps peak memory flat. The hand-in table's own `run_id`, `arm`, `task`,
    `fold_id` and `latency_ms` are dead weight here, and carrying `energy_tier` —
    or the predictions' `job_id` — onto nine million rows buys nothing the
    metrics use. `user_hash` *is* taken from the truth side so that a user-level
    bootstrap groups by a stable key; the predictions' `job_id` cannot serve,
    being neither unique nor a user.

    `probability` is the one hand-in column the ranking metrics do not read but
    calibration does, so it is carried only when asked for: it is a float per
    row, and nothing else in this module wants it.
    """
    keep = ["row_id", "test_month", "score"]
    if with_probability and "probability" in preds.columns:
        keep.append("probability")
    truth = with_row_id(pl.scan_parquet(canonical_path)).select(
        ["row_id", "state", "energy_j", "user_hash"])
    return (
        preds.lazy()
        .select(keep)
        .join(truth, on="row_id", how="left")
        .with_columns(label_expr(task))
        .collect()
    )
