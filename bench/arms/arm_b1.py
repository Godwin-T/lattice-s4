"""
Arm B1 — boosted trees over the shared Arm B feature layer.

`arm-b-planning.md` §6. LightGBM on `features.FEATURE_NAMES`, with native
categorical handling for the three string columns, early stopping on the fold's
inner-validation month, and the label-shuffle control on the same evenly-spread
folds Arm A uses.

**One sweep, two tasks.** T1 (TIMEOUT) and T2 (any failure) differ only in the
label; the features and the folds are identical. So a fold is read and cast to
its matrix once and both models fit from it. Each task still gets its own run
directory, because the evaluator scores one task per hand-in table — a file
carrying both would have every `row_id` twice and fail the contract.

**What early stopping does and does not do.** It chooses the *iteration count*
and nothing else: the hyperparameters are the frozen ones from `params/`,
identical for all 45 folds (decision B-D1). Validation is the last of
`fold["train_months"]` — the month Arm A already carves out for its calibrator —
so the test month is never touched. The first fold has a single training month
and so no inner month; there the model trains for the tuned fallback count
instead, and that is the only fold where the rule differs (§8.1).

**The control refits rather than re-binning.** Each shuffle draw builds its own
`lightgbm.Dataset` from the same matrix. Swapping labels on an already-binned
dataset would be faster, but a dataset that silently kept its old labels would
report a "control" that is really the arm — the worst possible failure, and a
quiet one. Re-binning costs a few minutes across the whole run and cannot fail
that way.
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
    PREDICTION_SCHEMA, RULE_SAFE, WINDOW_MAX_CHUNK_ROWS, _peak_rss_mb,
    load_manifest, normalise_predictions, with_row_id,
)
from .features import (
    CATEGORICAL_FEATURES, FEATURE_NAMES, FEATURE_VERSION, ensure_features,
)

ARM = "B1"
TASKS = ("T1", "T2")
PARAMS_DIR = "params"

# Fold guards, as Arm A's: a fold too small to fit or score is skipped and
# reported rather than crashed on.
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 50

# Rounds. With an inner month the model trains up to `NUM_BOOST_ROUND` and stops
# early; without one (fold 1) it trains for the tuned fallback exactly.
NUM_BOOST_ROUND = 2000
N_FALLBACK_ROUNDS = 300
EARLY_STOPPING_ROUNDS = 100

# Untuned defaults, used until `params/b1-<task>.json` exists. Deliberately
# modest: the tuned values from the Optuna pass replace every one of these.
DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_to_onehot": 8,
    "verbosity": -1,
}

LABEL_COLUMN = {"T1": "label", "T2": "label_t2"}
CATEGORY_INDICES = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]


def load_params(task: str, params_dir: str | Path = PARAMS_DIR) -> tuple[dict, int, str | None]:
    """
    The frozen hyperparameters for a task, and the fallback iteration count.

    Returns `(params, n_estimators, source)`. `source` is None when no tuned file
    exists, which the run records so an untuned result can never be read as tuned.
    """
    path = Path(params_dir) / f"{ARM.lower()}-{task.lower()}.json"
    if not path.exists():
        return dict(DEFAULT_PARAMS), N_FALLBACK_ROUNDS, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ({**DEFAULT_PARAMS, **payload["params"]},
            int(payload.get("n_estimators", N_FALLBACK_ROUNDS)), str(path))


def _categories(cache: str | Path) -> dict[str, list[str]]:
    """
    The category vocabulary of each string column, taken from the whole cache.

    Fixed once per run and sorted, so the integer codes a model sees are the same
    in every fold. A per-fold vocabulary would renumber the categories between
    folds and make "partition 3" mean something different in each model.
    """
    lf = pl.scan_parquet(cache)
    return {
        c: sorted(lf.select(c).unique().collect()[c].drop_nulls().to_list())
        for c in CATEGORICAL_FEATURES
    }


def _feature_exprs(categories: dict[str, list[str]]) -> list[pl.Expr]:
    """
    Feature columns as `Float32`, with the categoricals as integer codes.

    Casting inside the `select` means Polars never materialises the float64 form
    of a nine-million-row column, which halves the peak on the largest folds.
    """
    exprs = []
    for name in FEATURE_NAMES:
        if name in CATEGORICAL_FEATURES and categories[name]:
            exprs.append(pl.col(name).cast(pl.Enum(categories[name]))
                         .to_physical().cast(pl.Float32).alias(name))
        else:
            exprs.append(pl.col(name).cast(pl.Float32).alias(name))
    return exprs


def _matrix(frame: pl.DataFrame, exprs: list[pl.Expr]) -> np.ndarray:
    """The feature matrix of a frame as a contiguous `float32` array."""
    return frame.select(exprs).to_numpy().astype(np.float32, copy=False)


def _dataset(x: np.ndarray, y: np.ndarray):
    import lightgbm as lgb

    return lgb.Dataset(x, label=y, feature_name=list(FEATURE_NAMES),
                       categorical_feature=CATEGORY_INDICES, free_raw_data=True)


def _train(dataset, params, rounds, *, num_threads, seed, valid_sets=None,
           callbacks=None):
    import lightgbm as lgb

    p = {**params, "num_threads": num_threads, "seed": seed, "verbosity": -1}
    return lgb.train(p, dataset, num_boost_round=rounds,
                     valid_sets=valid_sets or [], callbacks=callbacks or [])


def _fold_model(x_train, y_train, x_val, y_val, *, params, fallback, num_threads, seed):
    """
    Fit one fold's model, returning `(booster, iterations)`.

    Early stopping only when there is an inner month with both classes present;
    otherwise the tuned fallback count is used exactly.
    """
    import lightgbm as lgb

    dataset = _dataset(x_train, y_train)
    if x_val is not None and y_val is not None and 0 < int(y_val.sum()) < len(y_val):
        valid = [lgb.Dataset(x_val, label=y_val, reference=dataset,
                             feature_name=list(FEATURE_NAMES),
                             categorical_feature=CATEGORY_INDICES,
                             free_raw_data=False)]
        booster = _train(dataset, params, NUM_BOOST_ROUND, num_threads=num_threads,
                         seed=seed, valid_sets=valid,
                         callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                       verbose=False)])
        best = booster.best_iteration
        # `best_iteration == 0` is a real answer (the first round won) and must
        # not be mistaken for "unset" by a truthiness check.
        return booster, int(best if best is not None else NUM_BOOST_ROUND)
    booster = _train(dataset, params, fallback, num_threads=num_threads, seed=seed)
    return booster, int(fallback)


def run(*, manifest_path: str, canonical_path: str, outdir: str,
        tasks: tuple[str, ...] = TASKS, seed: int = 42, control_repeats: int = 3,
        control_folds: int | None = None, features_dir: str | None = None,
        rebuild_features: bool = False, folds_limit: int | None = None,
        max_chunk_rows: int = WINDOW_MAX_CHUNK_ROWS,
        params_dir: str = PARAMS_DIR, num_threads: int = 2,
        window: str = RULE_SAFE) -> dict:
    """
    Score every fold of a manifest for each task, and write the hand-in tables.

    Returns the primary task's metadata, with every task's under `"tasks"`.
    """
    from sklearn.metrics import roc_auc_score

    if window != RULE_SAFE:
        raise SystemExit(f"Arm B1 runs only under the {RULE_SAFE!r} rule")
    tasks = tuple(tasks)
    if not tasks:
        raise SystemExit("no tasks requested")

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]

    # The arm must reproduce the manifest's eligible population exactly, or it
    # is scoring a different set of rows than every other arm.
    lf = with_row_id(pl.scan_parquet(canonical_path))
    users = eligible_users(lf)
    got = eligible_summary(users)
    if got != manifest.get("eligible_users"):
        raise SystemExit(
            f"eligible users do not match the manifest: {got} vs "
            f"{manifest.get('eligible_users')}")

    eligible = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    cache_dir = Path(features_dir) if features_dir else Path(outdir).parent / "windows"
    cache = ensure_features(eligible, canonical_path, cache_dir,
                            rebuild=rebuild_features, max_chunk_rows=max_chunk_rows,
                            rule=window)
    feats = pl.scan_parquet(cache)
    categories = _categories(cache)
    exprs = _feature_exprs(categories)
    print(f"  features       {cache} | {len(FEATURE_NAMES)} columns "
          f"({len(CATEGORICAL_FEATURES)} categorical) | v{FEATURE_VERSION}",
          flush=True)

    folds = manifest["folds"]
    if folds_limit:
        folds = folds[:folds_limit]
    control_set = _control_folds(folds, control_folds)

    run_ids = {t: f"{ARM}-{dataset}-{window}-{t.lower()}" for t in tasks}
    run_dirs = {t: Path(outdir) / run_ids[t] for t in tasks}
    for d in run_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    params_by_task, fallback_by_task, source_by_task = {}, {}, {}
    for t in tasks:
        p, n, src = load_params(t, params_dir)
        params_by_task[t], fallback_by_task[t], source_by_task[t] = p, n, src
        print(f"  params {t:<8} {src or 'defaults (untuned)'}", flush=True)

    # One accumulator per task, filled in a single sweep over the folds.
    state = {
        t: {"shards": [], "fold_aucs": [], "fold_auc_by_fold": {},
            "control_aucs": [], "control_folds_used": [], "skipped": [],
            "fit_seconds": 0.0, "inference_seconds": 0.0, "iterations": [],
            "n_control_folds_done": 0}
        for t in tasks
    }
    n_test = n_unscorable = 0
    scratch = Path(tempfile.mkdtemp(prefix="preds_b1_"))

    try:
        for fold in folds:
            inner_months = fold["train_months"][:-1]
            inner_month = fold["train_months"][-1]

            # `inner_months` is empty only on the first fold; the filter then
            # selects no training rows, which is why the fallback below exists.
            train = (feats
                     .filter(pl.col("scoreable")
                             & pl.col("month").is_in(inner_months))
                     .select([*exprs, "label", "label_t2"]).collect())
            val = (feats
                   .filter(pl.col("scoreable") & (pl.col("month") == inner_month))
                   .select([*exprs, "label", "label_t2"]).collect()
                   if inner_months else None)
            test = (feats
                    .filter(pl.col("month") == fold["test_month"])
                    .select(["job_id", "row_id", "month", "scoreable", "label",
                             "label_t2", "energy_j", *exprs]).collect())
            scoreable = test.filter(pl.col("scoreable"))
            n_test += test.height
            n_unscorable += test.height - scoreable.height

            if train.height < MIN_TRAIN_ROWS or scoreable.height < MIN_TEST_ROWS:
                for t in tasks:
                    state[t]["skipped"].append({
                        "fold_id": fold["fold_id"], "test_month": fold["test_month"],
                        "train_rows": train.height, "test_rows": scoreable.height})
                    state[t]["shards"].append(_write_blank(
                        test, scratch, t, fold, run_ids[t]))
                del train, val, test
                continue

            # One matrix for both tasks: the features do not depend on the label.
            x_train = _matrix(train, exprs)
            y_train = {t: train[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                       for t in tasks}
            x_val = (_matrix(val, exprs)
                     if val is not None and val.height else None)
            y_val = ({t: val[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                      for t in tasks} if x_val is not None else None)
            del train, val
            x_test = _matrix(scoreable, exprs) if scoreable.height else None
            test_labels = ({
                t: scoreable[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                for t in tasks} if x_test is not None else None)
            del scoreable

            for t in tasks:
                started = time.time()
                booster, iterations = _fold_model(
                    x_train, y_train[t], x_val, (y_val[t] if y_val else None),
                    params=params_by_task[t], fallback=fallback_by_task[t],
                    num_threads=num_threads, seed=seed)
                state[t]["fit_seconds"] += time.time() - started
                state[t]["iterations"].append(iterations)

                started = time.time()
                scores = (booster.predict(x_test, num_iteration=iterations)
                          if x_test is not None else np.array([]))
                predict_seconds = time.time() - started
                state[t]["inference_seconds"] += predict_seconds
                latency_ms = (1000.0 * predict_seconds / x_test.shape[0]
                              if x_test is not None and x_test.shape[0] else None)
                del booster

                frame = _blank_rows(test, run_ids[t], t, fold)
                if x_test is not None:
                    mask = test["scoreable"].to_numpy()
                    col = np.full(test.height, np.nan, dtype=np.float64)
                    col[mask] = scores
                    frame = frame.with_columns(
                        pl.Series("score", col).fill_nan(None),
                        pl.Series("probability", col).fill_nan(None))
                else:
                    frame = frame.with_columns(
                        pl.lit(None, dtype=pl.Float64).alias("score"),
                        pl.lit(None, dtype=pl.Float64).alias("probability"))
                frame = frame.with_columns(pl.lit(latency_ms, dtype=pl.Float64)
                                           .alias("latency_ms"))
                state[t]["shards"].append(_write_shard(frame, scratch, t, fold))

                if (test_labels is not None and 0 < int(test_labels[t].sum())
                        < len(test_labels[t])):
                    auc = float(roc_auc_score(test_labels[t], scores))
                    state[t]["fold_aucs"].append(auc)
                    state[t]["fold_auc_by_fold"][str(fold["fold_id"])] = auc

                if x_test is not None and fold["fold_id"] in control_set:
                    for rep in range(control_repeats):
                        rng = np.random.default_rng(seed + rep)
                        shuffled = y_train[t].copy()
                        rng.shuffle(shuffled)
                        c_booster, _ = _fold_model(
                            x_train, shuffled, None, None,
                            params=params_by_task[t], fallback=iterations,
                            num_threads=num_threads, seed=seed)
                        control_auc = float(roc_auc_score(
                            test_labels[t],
                            c_booster.predict(x_test, num_iteration=iterations)))
                        del c_booster
                        if not np.isnan(control_auc):
                            state[t]["control_aucs"].append(control_auc)
                            state[t]["control_folds_used"].append(int(fold["fold_id"]))
                    state[t]["n_control_folds_done"] += 1

            print(f"  fold {fold['fold_id']:>2}: train {x_train.shape[0]:>9,} "
                  f"test {test.height:>8,} "
                  f"| rounds {'/'.join(str(state[t]['iterations'][-1]) for t in tasks)} "
                  f"| rss {_peak_rss_mb():>6,.0f} MB", flush=True)
            del x_train, y_train, x_val, y_val, x_test, test_labels

        for t in tasks:
            _finish_task(t, state[t], run_dirs[t], run_ids[t], manifest,
                         manifest_path, canonical_path, tasks, seed,
                         control_repeats, control_set, n_test, n_unscorable,
                         params_by_task[t], fallback_by_task[t], source_by_task[t],
                         len(folds), cache, num_threads, window)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    primary = tasks[0]
    meta = _read_meta(run_dirs[primary])
    meta["tasks"] = {t: _read_meta(run_dirs[t]) for t in tasks}
    return meta


def _control_folds(folds: list[dict], count: int | None) -> set[int]:
    """
    Which folds the label-shuffle control runs on.

    `None` means every fold. Otherwise the folds are spread evenly across the
    run — centred, so a one-fold control lands mid-run rather than on the first
    fold. Arm A's rationale applies unchanged: the earliest folds have the fewest
    users and no signal, so a null drawn there is wide and tests nothing
    (`control_gate_finding.md` §6).
    """
    if count is None:
        return {fold["fold_id"] for fold in folds}
    n = max(0, min(count, len(folds)))
    if not n:
        return set()
    stride = len(folds) / n
    return {folds[min(len(folds) - 1, int((i + 0.5) * stride))]["fold_id"]
            for i in range(n)}


def _blank_rows(test: pl.DataFrame, run_id: str, task: str, fold: dict,
                arm: str = ARM) -> pl.DataFrame:
    """
    One row per test job, with the identity columns of the contract filled in.

    `arm` defaults to this module's own and is overridden by the arms that reuse
    this helper (B2, B3): the label is the contract's, and stamping another
    arm's name would make one arm's hand-in read as another's.
    """
    return (
        test.select(["job_id", "row_id", "month"])
        .with_columns([
            pl.lit(run_id).alias("run_id"),
            pl.lit(arm).alias("arm"),
            pl.lit(task).alias("task"),
            pl.lit(fold["fold_id"], dtype=pl.Int64).alias("fold_id"),
            pl.col("month").alias("test_month"),
        ])
    )


def _write_shard(frame: pl.DataFrame, scratch: Path, task: str, fold: dict) -> Path:
    """Normalise one fold's predictions to the contract and write the shard."""
    shard = scratch / f"{task}-fold-{fold['fold_id']:05d}.parquet"
    normalise_predictions(frame).write_parquet(shard)
    return shard


def _write_blank(test: pl.DataFrame, scratch: Path, task: str, fold: dict,
                 run_id: str, arm: str = ARM) -> Path:
    """A skipped fold still hands in every test row, with null scores."""
    frame = _blank_rows(test, run_id, task, fold, arm=arm).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("score"),
        pl.lit(None, dtype=pl.Float64).alias("probability"),
        pl.lit(None, dtype=pl.Float64).alias("latency_ms"))
    return _write_shard(frame, scratch, task, fold)


def _finish_task(task, acc, run_dir: Path, run_id: str, manifest, manifest_path,
                 canonical_path, tasks, seed, control_repeats, control_set,
                 n_test, n_unscorable, params, fallback, source, n_folds, cache,
                 num_threads, window):
    """Merge a task's shards and write its metadata."""
    path = run_dir / "predictions.parquet"
    if acc["shards"]:
        pl.scan_parquet([str(p) for p in acc["shards"]]).sink_parquet(path)
    else:                                      # pragma: no cover - no folds ran
        pl.DataFrame(schema=PREDICTION_SCHEMA).write_parquet(path)

    aucs = acc["fold_aucs"]
    meta = {
        "run_id": run_id,
        "arm": ARM,
        "task": task,
        "tasks_run": list(tasks),
        "dataset": manifest["dataset"],
        "manifest": str(manifest_path),
        "canonical": str(canonical_path),
        "seed": seed,
        "calibrated": False,
        "window_rule": window,
        "submit_time_safe": True,
        "folds": n_folds,
        "test_rows": n_test,
        "unscorable_rows": n_unscorable,
        "unscorable_fraction": (n_unscorable / n_test) if n_test else None,
        "fit_seconds_total": round(acc["fit_seconds"], 2),
        "inference_seconds_total": round(acc["inference_seconds"], 2),
        "inference_seconds_per_1m": (acc["inference_seconds"] / (n_test / 1e6)
                                     if n_test else None),
        "cost_usd_per_1m": 0.0,
        "runs_offline": True,
        "data_leaves_site": False,
        # The feature layer this was trained on, so a result can be tied to the
        # cache shape that produced it.
        "features": str(cache),
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "learner": "lightgbm",
        "num_threads": num_threads,
        # Tuning provenance (decision B-D1). A run that was not tuned says so.
        "params": params,
        "params_source": source,
        "tuned": source is not None,
        "n_estimators_fallback": fallback,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "iterations": acc["iterations"],
        "iterations_median": float(np.median(acc["iterations"]))
                             if acc["iterations"] else None,
        "fold_auc": aucs,
        "fold_auc_median": float(np.median(aucs)) if aucs else None,
        # The gate pairs the arm against its own shuffle fold by fold, so both
        # are recorded per fold and in the same order.
        "fold_auc_by_fold": acc["fold_auc_by_fold"],
        "control_aucs": acc["control_aucs"],
        "control_folds_used": acc["control_folds_used"],
        "control_mean": (float(np.mean(acc["control_aucs"]))
                         if acc["control_aucs"] else None),
        "control_repeats": control_repeats,
        "control_fold_ids": sorted(control_set),
        "control_folds": acc["n_control_folds_done"],
        "skipped_folds": acc["skipped"],
        "predictions": str(path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2),
                                           encoding="utf-8")
    return meta


def _read_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
