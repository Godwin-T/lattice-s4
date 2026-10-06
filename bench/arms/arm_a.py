"""
Arm A — the published Lattice24 method, run inside the harness.

The method, unchanged from the publication: take a user's previous 24 jobs,
summarise `elapsed / timelimit` with four statistics (mean, standard deviation,
range, mean absolute successive difference), and fit a logistic regression.
Untuned — `max_iter=1000`, otherwise defaults — exactly as published.

What is *not* the published implementation is the bookkeeping: this runs on the
frozen manifest, orders history by submit time, and requires every history job
to have ended before the target was submitted (`plan.md` §14.4). The published
tool orders by end time, which is slightly looser. That difference is deliberate
and is reported alongside the results, per §14.4.

Two run variants, per `metrics.md` M3:
  * raw        — the model's own probability;
  * isotonic   — the same ranking, with probabilities calibrated on an inner
                 validation month carved out of the training months.
Calibration is monotone, so it changes calibration but never the ranking.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..splits.eligibility import eligible_summary, eligible_users
from .common import (
    FEATURES, WINDOW_MAX_CHUNK_ROWS, ensure_windows, load_manifest,
    write_predictions,
)

ARM = "A"
TASK = "T1"

# A fold is skipped (not crashed on) if its training window set is too small to
# fit, or its test window set is too small to score. These mirror the reference
# implementation's own guards (`tr.sum() >= 100`, `te.sum() >= 50`).
MIN_TRAIN_WINDOWS = 100
MIN_TEST_WINDOWS = 50


def _fit(train: pl.DataFrame, labels: np.ndarray, seed: int):
    """Standardise, then fit. Mirrors the published preprocessing exactly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train.select(FEATURES).to_numpy())
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(scaler.transform(train.select(FEATURES).to_numpy()), labels)
    return scaler, model


def _predict(scaler, model, frame: pl.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(frame.select(FEATURES).to_numpy()))[:, 1]


def run(*, manifest_path: str, canonical_path: str, outdir: str,
        seed: int = 42, calibrate: bool = False, control_repeats: int = 3,
        control_folds: int | None = None, windows_dir: str | None = None,
        rebuild_windows: bool = False, folds_limit: int | None = None,
        max_chunk_rows: int = WINDOW_MAX_CHUNK_ROWS) -> dict:
    """
    Score every fold of a manifest and write the hand-in table.

    Returns a metadata dict (timings, counts, control AUCs, fold AUCs).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    run_id = f"A-{dataset}-{'iso' if calibrate else 'raw'}"

    lf = pl.scan_parquet(canonical_path)

    # The arm must reproduce the manifest's eligible population exactly, or it
    # would be scoring a different set of rows than every other arm.
    users = eligible_users(lf)
    got = eligible_summary(users)
    if got != manifest.get("eligible_users"):
        raise SystemExit(
            f"eligible users do not match the manifest: {got} vs "
            f"{manifest.get('eligible_users')}"
        )

    # Windows are built once and cached to Parquet, then read a fold at a time.
    # Holding eleven million windows in memory, and re-filtering that frame for
    # each of 45 folds, is what made an earlier run stall the machine.
    eligible = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    cache_dir = Path(windows_dir) if windows_dir else Path(outdir).parent / "windows"
    cache = ensure_windows(eligible, canonical_path, cache_dir,
                           rebuild=rebuild_windows, max_chunk_rows=max_chunk_rows)
    windows = pl.scan_parquet(cache)

    rows, fold_aucs, control_aucs = [], [], []
    skipped: list[dict] = []
    fit_seconds = inference_seconds = 0.0
    n_test = n_unscorable = 0
    folds = manifest["folds"]
    if folds_limit:
        # The FIRST folds, not the last: their training sets are the smallest,
        # so this is the cheapest way to check that a run works end to end.
        folds = folds[:folds_limit]
    control_on = folds if not control_folds else folds[-control_folds:]

    for fold in folds:
        # Only the columns a fold actually needs, so peak memory stays flat.
        train = (
            windows
            .filter(pl.col("scoreable") & pl.col("month").is_in(fold["train_months"]))
            .select([*FEATURES, "label"])
            .collect()
        )
        test = (
            windows
            .filter(pl.col("month") == fold["test_month"])
            .select(["job_id", "month", "scoreable", "label", "energy_j", *FEATURES])
            .collect()
        )
        scoreable = test.filter(pl.col("scoreable"))
        n_test += test.height
        n_unscorable += test.height - scoreable.height

        if train.height < MIN_TRAIN_WINDOWS or scoreable.height < MIN_TEST_WINDOWS:
            skipped.append({
                "fold_id": fold["fold_id"], "test_month": fold["test_month"],
                "train_windows": train.height, "test_windows": scoreable.height,
            })
            rows.append(_blank_rows(test, run_id, fold).with_columns([
                pl.lit(None, dtype=pl.Float64).alias("score"),
                pl.lit(None, dtype=pl.Float64).alias("probability"),
                pl.lit(None, dtype=pl.Float64).alias("latency_ms"),
            ]))
            continue

        started = time.time()
        scaler, model = _fit(train, train["label"].to_numpy(), seed)
        fit_seconds += time.time() - started

        started = time.time()
        scores = _predict(scaler, model, scoreable) if scoreable.height else np.array([])
        probabilities = scores

        if calibrate and scoreable.height:
            inner_train_months = fold["train_months"][:-1]
            inner_val_month = fold["train_months"][-1]
            # Re-filter the cache lazily rather than carrying the `month`
            # column (millions of strings) on the training frame.
            i_tr = (windows
                    .filter(pl.col("scoreable")
                            & pl.col("month").is_in(inner_train_months))
                    .select([*FEATURES, "label"]).collect())
            i_val = (windows
                     .filter(pl.col("scoreable") & (pl.col("month") == inner_val_month))
                     .select([*FEATURES, "label"]).collect())
            if i_tr.height and i_val.height and 0 < i_val["label"].sum() < i_val.height:
                s2, m2 = _fit(i_tr, i_tr["label"].to_numpy(), seed)
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(_predict(s2, m2, i_val), i_val["label"].to_numpy())
                probabilities = calibrator.predict(scores)
        inference_seconds += time.time() - started

        frame = _blank_rows(test, run_id, fold)
        if scoreable.height:
            preds = scoreable.select(["job_id"]).with_columns([
                pl.Series("score", scores),
                pl.Series("probability", probabilities),
            ])
            frame = frame.join(preds, on="job_id", how="left")
        else:
            frame = frame.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("score"),
                pl.lit(None, dtype=pl.Float64).alias("probability"),
            ])
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("latency_ms"))
        rows.append(frame)

        if scoreable.height and 0 < int(scoreable["label"].sum()) < scoreable.height:
            fold_aucs.append(float(roc_auc_score(scoreable["label"].to_numpy(), scores)))

        # Label-shuffle control: scrambled labels must score like a coin flip.
        if fold in control_on and train.height:
            for rep in range(control_repeats):
                rng = np.random.default_rng(seed + rep)
                shuffled = train["label"].to_numpy().copy()
                rng.shuffle(shuffled)
                s3, m3 = _fit(train, shuffled, seed)
                if scoreable.height:
                    auc = float(roc_auc_score(scoreable["label"].to_numpy(),
                                              _predict(s3, m3, scoreable)))
                    if not np.isnan(auc):
                        control_aucs.append(auc)

    # One directory per run, so scoring a second dataset or variant cannot
    # silently overwrite the first.
    run_dir = Path(outdir) / run_id
    predictions = pl.concat(rows, how="vertical_relaxed")
    path = write_predictions(predictions, run_dir)

    meta = {
        "run_id": run_id,
        "arm": ARM,
        "task": TASK,
        "dataset": dataset,
        "manifest": str(manifest_path),
        "canonical": str(canonical_path),
        "seed": seed,
        "calibrated": calibrate,
        "folds": len(folds),
        "test_rows": n_test,
        "unscorable_rows": n_unscorable,
        "unscorable_fraction": (n_unscorable / n_test) if n_test else None,
        "fit_seconds_total": round(fit_seconds, 2),
        "inference_seconds_total": round(inference_seconds, 2),
        "inference_seconds_per_1m": (inference_seconds / (n_test / 1e6)) if n_test else None,
        "cost_usd_per_1m": 0.0,
        "runs_offline": True,
        "data_leaves_site": False,
        "fold_auc": fold_aucs,
        "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
        "control_aucs": control_aucs,
        "control_mean": float(np.mean(control_aucs)) if control_aucs else None,
        "control_repeats": control_repeats,
        "control_folds": len(control_on),
        "skipped_folds": skipped,
        "predictions": str(path),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _blank_rows(test: pl.DataFrame, run_id: str, fold: dict) -> pl.DataFrame:
    """
    One row per test job with null scores.

    Used for jobs the arm cannot score (insufficient history) and for whole
    folds it must skip. Emitting the rows rather than omitting them keeps the
    "every test row appears exactly once" contract intact, and makes every
    exclusion visible.
    """
    return (
        test.select(["job_id", "month"])
        .with_columns([
            pl.lit(run_id).alias("run_id"),
            pl.lit(ARM).alias("arm"),
            pl.lit(TASK).alias("task"),
            pl.lit(fold["fold_id"], dtype=pl.Int64).alias("fold_id"),
            pl.col("month").alias("test_month"),
        ])
    )
