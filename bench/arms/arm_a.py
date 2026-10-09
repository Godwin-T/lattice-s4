"""
Arm A — the published Lattice24 method, run inside the harness.

The method, unchanged from the publication: take a user's previous 24 jobs,
summarise `elapsed / timelimit` with four statistics (mean, standard deviation,
range, mean absolute successive difference), and fit a logistic regression.
Untuned — `max_iter=1000`, otherwise defaults — exactly as published.

What is *not* the published implementation is the bookkeeping, and there are two
of them here (`--window`) — because that choice is the whole of Arm A's score:

* `safe` (default) — the window is the 24 jobs that had most recently **ended
  before the target was submitted**, and a target with any history job still
  running is unscorable. This is `folds_manifest.md`'s
  `history_jobs_must_end_before_target_submit` and `plan.md` §14.4; it is what
  every arm is scored under.

* `published` — the window is the previous 24 jobs in **submit order**, ended or
  not, which is what the publication actually does
  (`21913139/kestrel_replicate.py`: `sort_values('submit_time')`, then
  `idxs[k:k+WINDOW]`). It is reproduced solely so the two rules can be reported
  side by side under one evaluator. It is **not** submit-time safe: measured on
  Kestrel, 97.9% of its windows contain a job that had not finished when the
  target was submitted, so the feature uses a final `elapsed / timelimit` that
  does not exist at decision time. The 0.774 → 0.954 gap is this rule and
  nothing else.

Both rules are implemented once, in `common.build_windows`, so the only thing
that varies between them is how the rolling statistics attach to a target.

> The reference implementation was long misidentified in this repo. The
> published 0.954 comes from `21913139/kestrel_replicate.py` (submit order), not
> from `lattice24_assess/core.py`, which sorts by end (`core.py:228`) and is a
> third, separate tool that did not produce the published number.

Two run variants, per `metrics.md` M3:
  * raw        — the model's own probability;
  * isotonic   — the same ranking, with probabilities calibrated on an inner
                 validation month carved out of the training months.
Calibration is monotone, so it changes calibration but never the ranking.

With the window rule these give the run id `A-<dataset>-<safe|published>-<raw|iso>`.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..splits.eligibility import eligible_summary, eligible_users
from .common import (
    FEATURES, PREDICTION_SCHEMA, RULE_SAFE, WINDOW_MAX_CHUNK_ROWS, WINDOW_RULES,
    _peak_rss_mb, ensure_windows, load_manifest, normalise_predictions,
    with_row_id,
)

ARM = "A"
TASK = "T1"

# A fold is skipped (not crashed on) if its training window set is too small to
# fit, or its test window set is too small to score. These mirror the reference
# implementation's own guards (`tr.sum() >= 100`, `te.sum() >= 50`).
MIN_TRAIN_WINDOWS = 100
MIN_TEST_WINDOWS = 50


def _standardise(train: pl.DataFrame):
    """Fit the standardiser the published method applies before the model."""
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit(train.select(FEATURES).to_numpy())


def _fit_scaled(x: np.ndarray, labels: np.ndarray, seed: int):
    """Fit the published logistic regression on an already-standardised matrix."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(x, labels)
    return model


def _fit_matrix(x: np.ndarray, labels: np.ndarray, seed: int):
    """
    Fit the published pipeline to a raw feature matrix, standardising in place.

    Returns `(scaler, scaled, model)`. The scaled matrix is reused for every fit
    in the fold — the main model and each label-shuffle control — rather than
    being rebuilt and re-transformed for each. Shuffling the labels leaves the
    features untouched, so the standardiser belongs to the fold, not the fit.

    Like the publication, the standardiser is refit per fold (per run variant),
    not shared across folds.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x, copy=False)
    return scaler, scaled, _fit_scaled(scaled, labels, seed)


def _fit(train: pl.DataFrame, labels: np.ndarray, seed: int):
    """Standardise, then fit. Mirrors the published preprocessing exactly."""
    scaler = _standardise(train)
    model = _fit_scaled(scaler.transform(train.select(FEATURES).to_numpy()),
                        labels, seed)
    return scaler, model


def _predict(scaler, model, frame: pl.DataFrame) -> np.ndarray:
    return model.predict_proba(scaler.transform(frame.select(FEATURES).to_numpy()))[:, 1]


def _rss_mb() -> float:
    """Live resident memory in MB. Diagnostic only; 0.0 if unavailable."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024.0
    except OSError:                                    # noqa: BLE001 - diagnostic
        pass
    return 0.0


def run(*, manifest_path: str, canonical_path: str, outdir: str,
        seed: int = 42, calibrate: bool = False, control_repeats: int = 3,
        control_folds: int | None = None, windows_dir: str | None = None,
        rebuild_windows: bool = False, folds_limit: int | None = None,
        max_chunk_rows: int = WINDOW_MAX_CHUNK_ROWS,
        window: str = RULE_SAFE) -> dict:
    """
    Score every fold of a manifest and write the hand-in table.

    `window` picks the history rule: `safe` (default) or `published`. See the
    module docstring for what they mean and why both exist.

    Returns a metadata dict (timings, counts, control AUCs, fold AUCs).
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score

    if window not in WINDOW_RULES:
        raise SystemExit(f"unknown window rule {window!r}; expected one of {WINDOW_RULES}")

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    run_id = f"A-{dataset}-{window}-{'iso' if calibrate else 'raw'}"

    # `row_id` is attached before any filter, so the same canonical row gets the
    # same id here and in the evaluator, whatever population each goes on to
    # select. `job_id` cannot serve as that key (see `common.with_row_id`).
    lf = with_row_id(pl.scan_parquet(canonical_path))

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
    # each of 45 folds, is what made an earlier run stall the machine. The cache
    # is per window rule, so the two rules coexist without rebuilding either.
    eligible = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    cache_dir = Path(windows_dir) if windows_dir else Path(outdir).parent / "windows"
    cache = ensure_windows(eligible, canonical_path, cache_dir,
                           rebuild=rebuild_windows, max_chunk_rows=max_chunk_rows,
                           rule=window)
    windows = pl.scan_parquet(cache)

    fold_aucs, control_aucs = [], []
    # Recorded so the evaluator can pair the arm against its own shuffle fold by
    # fold. `control_aucs` is fold-major (all repeats for a fold, then the next
    # fold), so `control_folds_used` runs alongside it one entry per draw.
    control_folds_used: list[int] = []
    fold_auc_by_fold: dict[str, float] = {}
    skipped: list[dict] = []
    fit_seconds = inference_seconds = 0.0
    n_test = n_unscorable = 0
    folds = manifest["folds"]
    if folds_limit:
        # The FIRST folds, not the last: their training sets are the smallest,
        # so this is the cheapest way to check that a run works end to end.
        folds = folds[:folds_limit]
    # The control folds are spread evenly across the run, not taken from the
    # front. The earliest folds are the cheapest (their training sets are the
    # smallest) and that is what chose them, but it is the worst choice
    # statistically and it was costing more than it saved. Those folds carry the
    # fewest independent users — fold 3 has 84 and a per-draw shuffled sd of
    # 0.199, against 253 users and sd 0.080 by fold 30 — and the arm is erratic
    # on them: on Eagle's first two folds Arm A scores *below* chance (AUC 0.38
    # and 0.47). A gate that pairs the arm against its own null has nothing to
    # measure on a fold where the arm has no signal. Spreading the control costs
    # a few minutes of refits and buys a null tight enough to test against;
    # on Eagle's evenly-spread folds the arm scores 0.69-0.81.
    # See `control_gate_finding.md` section 6.
    control_set = _control_folds(folds, control_folds)
    control_done = 0

    # Predictions are streamed, one Parquet shard per fold, and merged at the
    # end: holding all ~7M rows in a Python list and `pl.concat`-ing them, then
    # re-materialising them in `write_predictions`, cost several GB. The loop
    # must stay flat instead.
    run_dir = Path(outdir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="preds_"))
    shards: list[Path] = []

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
            .select(["job_id", "row_id", "month", "scoreable", "label",
                     "energy_j", *FEATURES])
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
            shards.append(_write_shard(
                _blank_rows(test, run_id, fold).with_columns([
                    pl.lit(None, dtype=pl.Float64).alias("score"),
                    pl.lit(None, dtype=pl.Float64).alias("probability"),
                    pl.lit(None, dtype=pl.Float64).alias("latency_ms"),
                ]), scratch, fold))
            continue

        n_train = train.height
        started = time.time()
        # The frame is released the moment its NumPy form exists. On the last
        # folds it holds 9.1M rows, and keeping it alongside the feature matrix
        # and the standardised copy that matrix becomes is pure overhead.
        train_x = train.select(FEATURES).to_numpy()
        train_labels = train["label"].to_numpy()
        del train
        scaler, x_train, model = _fit_matrix(train_x, train_labels, seed)
        del train_x
        fit_seconds += time.time() - started

        started = time.time()
        # The test matrix is standardised once and reused by the main model and
        # every control fit below, instead of being re-transformed for each.
        x_test = (scaler.transform(scoreable.select(FEATURES).to_numpy(),
                                   copy=False)
                  if scoreable.height else None)
        scores = (model.predict_proba(x_test)[:, 1] if x_test is not None
                  else np.array([]))
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
        # Scores are attached positionally, not by a `job_id` join. `scoreable`
        # is `test` filtered in order, so its scores line up with the scoreable
        # rows of `test` one for one. `job_id` is *not* a unique key —
        # `canonical_table.md` §2 keys it only "stable within the dataset", and
        # §5 lists a duplicate `job_id` as a flagged, deliberately-unfixed edge
        # case (a requeued job keeps its id across attempts). Joining on it
        # therefore multiplies rows: a test row whose id appears k times matches
        # k rows and yields k². Measured on the kestrel manifest, one fold alone
        # turns 374,436 rows into 63.5M and exhausts memory.
        if scoreable.height:
            mask = test["scoreable"].to_numpy()
            score_col = np.full(test.height, np.nan, dtype=np.float64)
            prob_col = np.full(test.height, np.nan, dtype=np.float64)
            score_col[mask] = scores
            prob_col[mask] = probabilities
            frame = frame.with_columns([
                pl.Series("score", score_col).fill_nan(None),
                pl.Series("probability", prob_col).fill_nan(None),
            ])
        else:
            frame = frame.with_columns([
                pl.lit(None, dtype=pl.Float64).alias("score"),
                pl.lit(None, dtype=pl.Float64).alias("probability"),
            ])
        frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("latency_ms"))
        shards.append(_write_shard(frame, scratch, fold))

        if scoreable.height and 0 < int(scoreable["label"].sum()) < scoreable.height:
            fold_auc = float(roc_auc_score(scoreable["label"].to_numpy(), scores))
            fold_aucs.append(fold_auc)
            fold_auc_by_fold[str(fold["fold_id"])] = fold_auc

        # Label-shuffle control: scrambled labels must score like a coin flip.
        # Shuffling the labels leaves the features untouched, so the fold's
        # already-standardised training and test matrices are reused as-is rather
        # than rebuilt and re-transformed for every repeat.
        if n_train and fold["fold_id"] in control_set:
            labels = train_labels
            for rep in range(control_repeats):
                rng = np.random.default_rng(seed + rep)
                shuffled = labels.copy()
                rng.shuffle(shuffled)
                m3 = _fit_scaled(x_train, shuffled, seed)
                if x_test is not None:
                    auc = float(roc_auc_score(scoreable["label"].to_numpy(),
                                              m3.predict_proba(x_test)[:, 1]))
                    if not np.isnan(auc):
                        control_aucs.append(auc)
                        control_folds_used.append(int(fold["fold_id"]))
            control_done += 1

        # `rss` is what is live now; `peak` is the high-water mark. A `peak`
        # that only ever climbs while `rss` falls back to a floor means the
        # allocator is not returning freed pages, not that a fold needs more.
        print(f"  fold {fold['fold_id']:>2}: train {n_train:>9,} "
              f"test {test.height:>8,} | rss {_rss_mb():>6,.0f} MB "
              f"| peak {_peak_rss_mb():>6,.0f} MB", flush=True)

    # Merge the per-fold shards by streaming straight into the final file, so
    # the whole hand-in table is never resident. `sink_parquet` reads the shards
    # lazily; an empty shard list means no test rows at all (fold_limit=0, say).
    path = run_dir / "predictions.parquet"
    try:
        if shards:
            pl.scan_parquet([str(p) for p in shards]).sink_parquet(path)
        else:
            pl.DataFrame(schema=PREDICTION_SCHEMA).write_parquet(path)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    meta = {
        "run_id": run_id,
        "arm": ARM,
        "task": TASK,
        "dataset": dataset,
        "manifest": str(manifest_path),
        "canonical": str(canonical_path),
        "seed": seed,
        "calibrated": calibrate,
        "window_rule": window,
        # The one thing a reader must know to interpret the AUC. `safe` windows
        # use only jobs that had ended before the target was submitted; the
        # published rule does not, and its features are not knowable at submit
        # time. The evaluator's shuffle gate does *not* check this, so it is
        # recorded here and surfaced in the write-up rather than left to the gate.
        "submit_time_safe": window == RULE_SAFE,
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
        # The same AUCs keyed by fold id. The control gate needs the arm's AUC on
        # exactly the folds the control ran on, and a positional list cannot say
        # which those are once a fold is skipped.
        "fold_auc_by_fold": fold_auc_by_fold,
        "control_aucs": control_aucs,
        "control_folds_used": control_folds_used,
        "control_mean": float(np.mean(control_aucs)) if control_aucs else None,
        "control_repeats": control_repeats,
        "control_fold_ids": sorted(control_set),
        "control_folds": control_done,
        "skipped_folds": skipped,
        "predictions": str(path),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _control_folds(folds: list[dict], count: int | None) -> set[int]:
    """
    Which folds the label-shuffle control runs on.

    `None` means every fold. Otherwise the folds are spread evenly across the
    run — centred, so a one-fold control lands mid-run rather than on the first
    fold. See the call site for why taking the cheapest folds was the wrong
    choice.
    """
    if count is None:
        return {fold["fold_id"] for fold in folds}
    n = max(0, min(count, len(folds)))
    if not n:
        return set()
    stride = len(folds) / n
    return {folds[min(len(folds) - 1, int((i + 0.5) * stride))]["fold_id"]
            for i in range(n)}


def _write_shard(frame: pl.DataFrame, scratch: Path, fold: dict) -> Path:
    """
    Write one fold's predictions to its own Parquet file and return the path.

    Each shard is normalised to the contract schema as it is written, so the
    shards share one schema and can be merged by streaming at the end of the
    run — the arm never holds more than a fold's worth of rows at once.
    """
    shard = scratch / f"fold-{fold['fold_id']:05d}.parquet"
    normalise_predictions(frame).write_parquet(shard)
    return shard


def _blank_rows(test: pl.DataFrame, run_id: str, fold: dict) -> pl.DataFrame:
    """
    One row per test job with null scores.

    Used for jobs the arm cannot score (insufficient history) and for whole
    folds it must skip. Emitting the rows rather than omitting them keeps the
    "every test row appears exactly once" contract intact, and makes every
    exclusion visible.
    """
    return (
        test.select(["job_id", "row_id", "month"])
        .with_columns([
            pl.lit(run_id).alias("run_id"),
            pl.lit(ARM).alias("arm"),
            pl.lit(TASK).alias("task"),
            pl.lit(fold["fold_id"], dtype=pl.Int64).alias("fold_id"),
            pl.col("month").alias("test_month"),
        ])
    )
