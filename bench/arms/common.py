"""
Shared machinery for the arms.

**Windowing.** Every arm that predicts a job's outcome looks back over that
user's history. What differs between arms is which features they take from it;
what must not differ is *which history is legal*. So the history rule lives here,
once:

* jobs are ordered by **submit time** within a user (`plan.md` §14.4);
* the window is the user's previous `WINDOW` jobs in that order;
* a target is only scoreable if **every one of those jobs had ended before the
  target was submitted** — otherwise the window would contain an outcome that
  was not yet knowable.

That last rule is deliberately conservative: if one of the previous 24 jobs was
still running, the target is marked unscorable rather than reaching further back.
It can only lose rows, never leak. How many it loses is reported.

**The hand-in table.** `write_predictions` enforces the contract: one row per
test job per fold, `score`/`probability` null for jobs an arm cannot score, and
never a row for a job that is not in that fold's test set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

WINDOW = 24
FEATURES = ("f_mean", "f_std", "f_range", "f_madiff")

PREDICTION_SCHEMA = {
    "run_id": pl.String,
    "arm": pl.String,
    "task": pl.String,
    "fold_id": pl.Int64,
    "test_month": pl.String,
    "job_id": pl.String,
    "score": pl.Float64,
    "probability": pl.Float64,
    "latency_ms": pl.Float64,
}


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_windows(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Turn the canonical table into one row per (target job -> its 24-job history).

    The four features are the published Lattice24 set: mean, standard deviation,
    range, and mean absolute successive difference of `elapsed / timelimit` over
    the window. Nothing about the target itself enters its own features.

    **Which 24 jobs?** The 24 that had most recently *ended* before the target
    was submitted. The naive reading — "the previous 24 jobs by submit order,
    and require that they all ended first" — is far too strict for real
    clusters, where array jobs submit thousands of jobs in the same second and
    none of them has ended yet. That reading discards almost the entire dataset
    (measured: ~1.5% of rows survive on Eagle), whereas this one keeps the
    history a submit-time decision could genuinely have used.

    Implementation: jobs are ordered by **end** time within each user; the
    rolling statistics are then the window ending at each job. An as-of join
    maps each target to the last job that had ended before its submit time, and
    the target inherits that row's statistics. Ordering by end time is safe:
    anything that ended before the target was submitted was necessarily
    submitted before it too.
    """
    ordered = (
        lf.select(["job_id", "user_hash", "submit_time", "end_time",
                   "elapsed_s", "timelimit_s", "state", "energy_j"])
        .with_columns([
            (pl.col("elapsed_s") / pl.col("timelimit_s")).alias("ratio"),
            pl.col("submit_time").dt.epoch("s").alias("submit_s"),
            pl.col("end_time").dt.epoch("s").alias("end_s"),
            pl.col("submit_time").dt.strftime("%Y-%m").alias("month"),
        ])
    )

    # Rolling statistics of the 24 jobs ending at (and before) each job, in end
    # order. Attached to the last job of each window.
    by_end = ordered.sort(["user_hash", "end_s", "job_id"]).with_columns([
        pl.int_range(pl.len()).over("user_hash").cast(pl.Int32).alias("idx_end"),
        pl.col("ratio").rolling_mean(WINDOW).over("user_hash").alias("f_mean"),
        # ddof=0 to match the published method (numpy's default); Polars uses 1.
        pl.col("ratio").rolling_std(WINDOW, ddof=0).over("user_hash").alias("f_std"),
        (pl.col("ratio").rolling_max(WINDOW).over("user_hash")
         - pl.col("ratio").rolling_min(WINDOW).over("user_hash")).alias("f_range"),
        pl.col("ratio").diff().abs()
          .rolling_mean(WINDOW - 1).over("user_hash").alias("f_madiff"),
    ])

    # Each target takes the statistics of the last job that ended before it was
    # submitted. A target whose match has fewer than 24 jobs behind it cannot be
    # scored.
    targets = ordered.sort(["user_hash", "submit_s", "job_id"]).select([
        "job_id", "user_hash", "submit_s", "month", "state", "energy_j",
    ])
    joined = targets.join_asof(
        by_end.select(["user_hash", "end_s", "idx_end", *FEATURES]),
        left_on="submit_s", right_on="end_s", by="user_hash",
        strategy="backward",
    )

    return joined.with_columns([
        ((pl.col("idx_end").is_not_null()) & (pl.col("idx_end") >= WINDOW - 1))
        .alias("scoreable"),
        (pl.col("state") == "TIMEOUT").alias("label"),
    ]).select(["job_id", "user_hash", "month", "scoreable", "label", "energy_j",
               *FEATURES])


def write_predictions(rows: pl.DataFrame, outdir: str | Path) -> Path:
    """Write the hand-in table in the contract shape."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "predictions.parquet"
    exprs = [
        (pl.col(name).cast(dtype, strict=False) if name in rows.columns
         else pl.lit(None, dtype=dtype)).alias(name)
        for name, dtype in PREDICTION_SCHEMA.items()
    ]
    rows.select(exprs).write_parquet(path)
    return path


def test_rows_for_fold(windows: pl.DataFrame, fold: dict) -> pl.DataFrame:
    """Every test row of a fold, scoreable or not, so none can go missing."""
    return windows.filter(pl.col("month") == fold["test_month"])


def train_rows_for_fold(windows: pl.DataFrame, fold: dict) -> pl.DataFrame:
    """Scoreable windows whose target falls in the fold's training months."""
    return windows.filter(
        pl.col("scoreable")
        & pl.col("month").is_in(fold["train_months"])
    )


# ---------------------------------------------------------------------------
# Light-weight path: cache the windows once, then stream one fold at a time.
# ---------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_windows(lf: pl.LazyFrame, canonical_path: str | Path,
                   cache_dir: str | Path, rebuild: bool = False) -> Path:
    """
    Build the window table once and keep it as Parquet.

    `lf` is the already-filtered frame to build from (the arms pre-filter to the
    manifest's eligible population). Building windows is the memory-heavy step —
    one row per job, eleven million of them on Eagle — so writing them out means
    a run can then read a fold at a time instead of holding everything in RAM.
    The cache is keyed to the canonical table's hash, so a rebuilt table
    invalidates it automatically.
    """
    canonical_path = Path(canonical_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{canonical_path.stem}.windows.parquet"
    meta = cache.with_suffix(".meta.json")

    fingerprint = file_sha256(canonical_path)
    if not rebuild and cache.exists() and meta.exists():
        try:
            if json.loads(meta.read_text())["canonical_sha256"] == fingerprint:
                return cache
        except (OSError, ValueError, KeyError):
            pass

    frame = build_windows(lf).collect()
    frame.write_parquet(cache)
    meta.write_text(json.dumps({
        "canonical": str(canonical_path),
        "canonical_sha256": fingerprint,
        "rows": frame.height,
        "scoreable": int(frame["scoreable"].sum()),
    }, indent=2), encoding="utf-8")
    return cache
