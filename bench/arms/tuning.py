"""
One-shot hyperparameter tuning for Arm B — decision B-D1.

    python3 -m bench.arms.tuning --arm B1 --task T1 \
        --manifest data/manifests/eagle_parquet.folds.json \
        --canonical data/canonical/eagle_parquet.parquet \
        --trials 50 --out params --threads 4

50 Optuna trials, once per model per task, on an **early validation block**: the
inner-validation months of the earliest folds (2018-12 → 2019-07 on Eagle). The
best parameters are frozen to `params/<arm>-<task>.json` and reused by every one
of the 45 folds, so all folds are scored with identical hyperparameters.

**B3 gets four of those months, not eight (decision B-D1a).** The 50-trial budget
was set before the cost was measurable. A boosted trial over the eight-month block
costs ~104 s, B2's MLP ~3–5 min, but B3 walks a GRU over 32 *sequential* timesteps
that do not parallelise across time: the same 50 trials would have taken 11–33 h of
exclusive CPU on an eight-core box that also has to run the arms and their
evaluations. The trial *count* — the part of B-D1 that governs how well the space
is searched — is kept; the *block* is cut to its four most recent months
(2019-04 → 2019-07), which is ~3× cheaper and still leaves four independent
held-out months. The months actually used are written into `params/b3-*.json` under
`validation_months`, and `folds_sharing_a_tuned_month` is recomputed from the
reduced block, so the deviation is legible rather than smoothed over. B1 and B2
keep the full eight.

**Why one-shot, and what it costs.** Per-fold tuning would take about a day per
model on this eight-core box and would tune later folds on months adjacent to
their own test month — a temporal leak inside the tuning loop. Tuning once on the
earliest months keeps the full 50-trial budget and touches only months that are
already long past by the time the later folds are scored.

**The disclosure that must travel with the params.** Every month belongs to some
fold's test set, so a block drawn from real months is the test month of the
earliest few folds. Tuned parameters therefore *do* see the test months of the
first folds (2018-12 … 2019-07 — eight of forty-five on Eagle, four for B3), and
those folds are not independent of tuning. The alternative — tuning on nothing but
the single month before the first fold — is not tuning at all. This is written into
the params file under `leak_note`, recorded in every run's metadata, and belongs in
the write-up; it is not something a reader should have to discover.

Within the block, each validation month is scored by a model trained only on
months strictly before it, and the objective is the mean of those per-month AUCs
— the same quantity the benchmark reports, measured the same way.

**Nothing early-stops inside a trial, for any of the three learners.** B1 trains a
fixed `TUNE_ROUNDS`; B2 and B3 train a fixed `TUNE_EPOCHS` and keep the last
weights, with no inner month and no checkpoint. Letting each trial pick its own
iteration count would make two trials differ in two ways at once — the
hyperparameters *and* how long they were allowed to run — and the objective would
be a mix of the two. The run that consumes these params does early-stop; the trial
that chose them does not.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from optuna.trial import TrialState

from .arm_b1 import LABEL_COLUMN, _categories, _feature_exprs, _matrix
from .common import RULE_SAFE, load_manifest, with_row_id
from ..splits.eligibility import eligible_users

# How many of the earliest folds define the block. Nine gives 2018-11 → 2019-07
# on Eagle; the first month is dropped because nothing precedes it to train on.
BLOCK_FOLDS = 9

# How many of the block's *most recent* months a model is tuned on. `None` keeps
# the whole block. B3 is the only entry, and B-D1a is the reason: a sequential
# learner is ~10-20x an MLP per epoch and the full block is a day and a half of
# this box for one model's hyperparameters.
BLOCK_MONTHS_CAP = {"B1": None, "B2": None, "B3": 4}

# Rounds used inside a B1 trial. No early stopping here: the trial is measuring
# the *hyperparameters*, and letting each trial pick its own iteration count
# would make two trials differ in two ways at once.
TUNE_ROUNDS = 400

# The same rule for B2/B3, in epochs. 12 rather than a smaller number because it
# is the arms' own `MIN_EPOCHS` — the point past which their inner-validation
# month is allowed to mean anything. A shorter budget would tune for a model that
# has not yet left its initialisation.
TUNE_EPOCHS = 12

MODELS = ("B1", "B2", "B3")


def block_months(manifest: dict, n_folds: int = BLOCK_FOLDS) -> list[str]:
    """
    The months tuning is allowed to look at, earliest first.

    The inner-validation month of each of the earliest folds, minus any month
    with nothing before it to train on.
    """
    inner = sorted({f["train_months"][-1] for f in manifest["folds"][:n_folds]
                    if f["train_months"]})
    if not inner:
        return []
    first = min(m for f in manifest["folds"] for m in f["train_months"])
    return [m for m in inner if m > first]


def tuning_block(manifest: dict, arm: str, n_folds: int = BLOCK_FOLDS) -> list[str]:
    """
    The block *this arm* is tuned on — `block_months`, then B-D1a's reduction.

    A function rather than three lines inside `tune()` so the reduction can be
    asserted without running a study: which months a model's hyperparameters saw
    is the whole content of the disclosure, and a disclosure that is only
    reachable by fitting for three hours is one nobody checks.
    """
    block = block_months(manifest, n_folds)
    cap = BLOCK_MONTHS_CAP.get(arm)
    if cap and len(block) > cap:
        return block[-cap:]
    return block


def folds_sharing_a_tuned_month(manifest: dict, block: list[str]) -> list:
    """
    The folds whose test month is inside the block, and so are not independent.

    Computed from the block actually used, so a reduced block shortens the
    disclosure rather than leaving it claiming months that were never seen.
    """
    months = set(block)
    return sorted(f["fold_id"] for f in manifest["folds"] if f["test_month"] in months)


def _search_space(trial, arm: str) -> dict:
    """
    The Optuna search space for one model, as arm-shaped parameter overrides.

    The values are the names the arms themselves read (`hidden`, `learning_rate`,
    …), not Optuna's flat trial names, so the dict returned here is the same one
    that lands in `params/<arm>-<task>.json` and is merged over `DEFAULT_PARAMS`.
    A shape the arm cannot use — a two-layer MLP written as two separate keys —
    would be found by the run, not by the tuner, and only after the hours it took
    to tune.
    """
    if arm == "B1":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 1000, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 4, 32),
        }
    if arm == "B2":
        # `hidden_2 = 0` means "no second hidden layer", the standard way to put
        # depth itself in the space rather than fixing it. The second key is
        # always suggested, so a resumed study sees one consistent space.
        hidden = [trial.suggest_categorical("hidden_1", [64, 128, 256, 512])]
        second = trial.suggest_categorical("hidden_2", [0, 64, 128, 256])
        if second:
            hidden.append(int(second))
        return {
            "hidden": hidden,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
        }
    if arm == "B3":
        # `num_layers` is in the space *and* gates the GRU's own inter-layer
        # dropout, so the two are not independent — which is the point: a
        # single-layer GRU cannot use dropout between layers at all.
        return {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
        }
    raise SystemExit(f"no search space defined for arm {arm!r}")


def _arm_params(flat: dict, arm: str) -> dict:
    """
    Optuna's flat trial parameters, as the dict the arm's `load_params` merges.

    **Not a formality, and the silent failure this module is most exposed to.**
    `study.best_params` is keyed by the name passed to `suggest_*`, which for B2
    is `hidden_1`/`hidden_2` — a flat pair, because depth has to be *suggested*
    as two independent choices. The arm reads `params["hidden"]`, a list. Writing
    the flat form to `params/b2-*.json` would leave `load_params`'s merge with no
    `hidden` key at all, so every fold would train the *default* two-layer MLP
    while the params file claimed a tuned architecture, and the metadata's
    `params_source` would point at it as though it were real. The conversion is a
    function of its own so it can be tested without a study.
    """
    if arm == "B2":
        hidden = [int(flat["hidden_1"])]
        if int(flat.get("hidden_2") or 0):
            hidden.append(int(flat["hidden_2"]))
        return {k: v for k, v in flat.items()
                if k not in ("hidden_1", "hidden_2")} | {"hidden": hidden}
    # B1 and B3 name their parameters the way their arms read them, so the flat
    # form *is* the arm form.
    return dict(flat)


def tune(*, arm: str, task: str, manifest_path: str, canonical_path: str,
         out: str, trials: int = 50, seed: int = 42, features_dir: str | None = None,
         num_threads: int = 4, window: str = RULE_SAFE,
         studies_dir: str | None = None, epochs: int = TUNE_EPOCHS,
         device: str = "auto") -> dict:
    """Run the study and write `params/<arm>-<task>.json`."""
    import lightgbm as lgb
    import optuna

    from sklearn.metrics import roc_auc_score

    from .features import FEATURE_NAMES, ensure_features

    if arm not in MODELS:
        raise SystemExit(f"no search space defined for arm {arm!r}")

    if arm == "B2":
        from . import arm_b2 as learner
        from .arm_b2 import DEFAULT_PARAMS, EARLY_STOPPING_PATIENCE
    elif arm == "B3":
        from . import arm_b3 as learner
        from .arm_b3 import DEFAULT_PARAMS, EARLY_STOPPING_PATIENCE
        from .sequences import SequenceSource, ensure_sequences
    else:
        from . import arm_b1 as learner
        from .arm_b1 import CATEGORY_INDICES, DEFAULT_PARAMS, EARLY_STOPPING_ROUNDS

    if arm in ("B2", "B3"):
        device = learner.resolve_device(device)
    else:
        device = "cpu"

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]
    # B-D1a: an arm tuned on fewer months is tuned on the block's *latest*, so
    # the training set behind each validation month is as large as it can be.
    block = tuning_block(manifest, arm)
    if not block:
        raise SystemExit("no validation block: the manifest has no usable early months")

    # The cap the run will apply. Read off the arm so tuning and the run it feeds
    # cannot disagree about how many rows the model sees.
    train_row_cap = getattr(learner, "TRAIN_ROW_CAP", None)

    lf = with_row_id(pl.scan_parquet(canonical_path))
    users = eligible_users(lf)
    eligible = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    # The same cache directory the arms use by default (`Path(outdir).parent /
    # "windows"` with `--out runs`), so tuning and the run it feeds share one
    # table instead of each building its own copy of eleven million rows.
    cache_dir = Path(features_dir) if features_dir else Path("windows")
    cache = ensure_features(eligible, canonical_path, cache_dir, rule=window)

    # One read of the whole early region. Every trial then filters it in memory
    # instead of re-reading the Parquet eight times.
    last_block_month = block[-1]
    early = (pl.scan_parquet(cache)
             .filter(pl.col("scoreable") & (pl.col("month") <= last_block_month)
                     & pl.col("month").is_not_null())
             .collect())
    label = LABEL_COLUMN[task]

    src = None
    if arm == "B3":
        # The second cache. B3 still reads the feature cache — for `scoreable`,
        # `month`, `hist_n` and the label — but its windows come from here.
        from .common import WINDOW_MAX_CHUNK_ROWS
        seq_path = ensure_sequences(eligible, canonical_path, cache_dir,
                                    max_chunk_rows=WINDOW_MAX_CHUNK_ROWS, rule=window)
        src = SequenceSource.load(seq_path)
        print(f"  sequences      {src.n_jobs:,} jobs over {src.n_users:,} users",
              flush=True)
        import torch
        torch.set_num_threads(max(1, num_threads))

    exprs = None
    categories = None
    if arm in ("B1", "B2"):
        categories = _categories(cache)
        exprs = _feature_exprs(categories)

    print(f"  block          {block[0]} .. {block[-1]} ({len(block)} months)", flush=True)
    print(f"  early rows     {early.height:,} (months <= {last_block_month})", flush=True)
    if train_row_cap:
        print(f"  train row cap  {train_row_cap:,} (a trial sees no more)", flush=True)

    def _score(train: pl.DataFrame, val: pl.DataFrame, params: dict) -> float | None:
        """One validation month's AUC, or None when it cannot be scored."""
        y_val = val[label].to_numpy().astype(np.float32)
        if train.height < 100 or not (0 < int(y_val.sum()) < len(y_val)):
            return None

        if arm == "B1":
            booster = lgb.train(
                params, lgb.Dataset(
                    _matrix(train, exprs),
                    label=train[label].to_numpy().astype(np.float32),
                    feature_name=list(FEATURE_NAMES),
                    categorical_feature=CATEGORY_INDICES),
                num_boost_round=TUNE_ROUNDS)
            return float(roc_auc_score(y_val, booster.predict(_matrix(val, exprs))))

        if arm == "B2":
            y = train[label].to_numpy().astype(np.float32)
            x_raw, _, kept = learner._cap_rows(_matrix(train, exprs), train_row_cap, seed)
            if kept is not None:
                y = y[kept]
            expanded = learner._onehot(x_raw, categories)
            mu, sd = learner._standardiser(expanded)
            x = learner._standardise(expanded, mu, sd)
            del expanded, x_raw
            x_val = learner._standardise(
                learner._onehot(_matrix(val, exprs), categories), mu, sd)
            model, _, _ = learner._fit(x, y, None, None, params, seed=seed,
                                       epochs=epochs, device=device)
            return float(roc_auc_score(y_val, learner._predict(model, x_val)))

        # B3. The cap goes on the targets, before the windows exist — the same
        # order the run uses, and for the same reason (`_cap_index`).
        y = train[label].to_numpy().astype(np.float32)
        hashes = train["user_hash"].to_numpy()
        hist_n = train["hist_n"].to_numpy()
        keep, _ = learner._cap_index(train.height, train_row_cap, seed)
        if keep is not None:
            hashes, hist_n, y = hashes[keep], hist_n[keep], y[keep]
        x, m = learner._raw_windows(src, hashes, hist_n)
        del hashes, hist_n
        mu, sd = learner._fit_scale(x, m)
        x = learner._scaled(x, m, mu, sd)
        x_val, m_val = learner._raw_windows(src, val["user_hash"].to_numpy(),
                                            val["hist_n"].to_numpy())
        x_val = learner._scaled(x_val, m_val, mu, sd)
        model, _, _ = learner._fit(x, m, y, None, None, None, params, seed=seed,
                                   epochs=epochs, device=device)
        return float(roc_auc_score(y_val, learner._predict(model, x_val, m_val)))

    def objective(trial) -> float:
        override = _search_space(trial, arm)
        params = {**DEFAULT_PARAMS, **override,
                  "num_threads": num_threads, "seed": seed}
        if arm == "B1":
            params["verbosity"] = -1
        aucs = []
        for month in block:
            auc = _score(early.filter(pl.col("month") < month),
                         early.filter(pl.col("month") == month), params)
            if auc is not None:
                aucs.append(auc)
        if not aucs:
            return float("nan")
        return float(np.mean(aucs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # A study that lives only in memory loses every trial if the process dies,
    # and this one runs for hours. Persisting to SQLite makes a restart resume
    # the TPE sampler instead of starting over — and makes "how far along is it"
    # a question with an answer.
    study_name = f"{arm}-{task}-{dataset}"
    studies = Path(studies_dir) if studies_dir else Path(out).parent / "studies"
    studies.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name,
        storage=f"sqlite:///{studies / (study_name + '.db')}",
        load_if_exists=True)
    # A process killed during a long objective leaves its Optuna trial in
    # RUNNING state.  Counting that row toward the budget makes a resumed
    # study either skip the missing trial or crash at ``best_params`` because
    # no completed trial exists.  Mark stale rows failed and spend the budget
    # on genuinely completed trials instead.  This also makes the documented
    # SQLite resume behaviour true after a terminal/session interruption.
    stale = [t for t in study.trials if t.state == TrialState.RUNNING]
    for trial in stale:
        study._storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    remaining = max(0, trials - len(completed))
    if len(study.trials):
        print(f"  resuming       {len(completed)} completed trials, "
              f"{len(stale)} stale trial(s) marked failed, "
              f"{remaining} to go", flush=True)
    finished = [t.value for t in completed
                if t.value is not None and math.isfinite(t.value)]
    # -inf, not nan: every comparison against nan is False, so `max(nan, x)` is
    # nan and a nan seed would make the reported best unreadable for the whole
    # study.
    best_so_far = [max(finished) if finished else float("-inf")]

    def _progress(study, trial) -> None:
        if trial.value is not None and math.isfinite(trial.value):
            best_so_far[0] = max(best_so_far[0], trial.value)
        if (trial.number + 1) % 5 == 0 or trial.number == 0:
            value = "n/a" if trial.value is None else format(trial.value, "+.4f")
            best = ("n/a" if best_so_far[0] == float("-inf")
                    else format(best_so_far[0], ".4f"))
            print(f"  trial {trial.number + 1:>3}/{trials} value {value} "
                  f"best {best}", flush=True)

    study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                   callbacks=[_progress])

    # `_arm_params`, not `study.best_params` directly: for B2 the two differ, and
    # writing the flat form would produce a params file the arm reads as untuned.
    best = _arm_params(study.best_params, arm)
    payload = {
        "arm": arm,
        "task": task,
        "dataset": dataset,
        "learner": {"B1": "lightgbm", "B2": "pytorch-mlp",
                    "B3": "pytorch-gru"}[arm],
        "device": device,
        "n_trials": len(study.trials),
        "n_trials_target": trials,
        "study_name": study_name,
        "seed": seed,
        "tuned_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "objective": "mean per-month AUC over the validation block",
        "validation_months": block,
        "block_months_used": len(block),
        "train_rule": "for each validation month, months strictly before it",
        "best_value": study.best_value,
        "params": best,
        "defaults": {k: v for k, v in DEFAULT_PARAMS.items()
                     if k not in ("objective", "metric", "verbosity")},
        # The folds whose test month is inside the block are not independent of
        # tuning; a reader must be able to see which, without recomputing. For
        # B3 this is computed from the *reduced* block, so the shorter disclosure
        # is what the file actually claims.
        "leak_note": (
            "Tuning looked at these months, which are the test months of the "
            "earliest folds; those folds are not independent of the tuned "
            "parameters. Recorded so the write-up can say which."),
        "folds_sharing_a_tuned_month": folds_sharing_a_tuned_month(manifest, block),
        "top_trials": [
            {"value": t.value, "params": _arm_params(t.params, arm)}
            for t in sorted((t for t in study.trials if t.value is not None),
                            key=lambda t: t.value, reverse=True)[:5]],
    }
    if arm == "B1":
        payload["tune_rounds"] = TUNE_ROUNDS
        payload["n_estimators"] = TUNE_ROUNDS
        payload["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    else:
        # The trial budget is written down because it is *not* the run's rule:
        # the run early-stops on its inner month, the trial trains a fixed
        # `epochs` and keeps the last weights.
        payload["tune_epochs"] = epochs
        payload["early_stopping_patience"] = EARLY_STOPPING_PATIENCE
        payload["train_row_cap"] = train_row_cap
        payload["no_early_stopping_in_trial"] = (
            f"each trial trained exactly {epochs} epochs and kept the last "
            "weights, so two trials cannot differ in iteration count as well as "
            "in hyperparameters")
        if arm == "B3":
            payload["reduced_block_note"] = (
                f"B3 is tuned on the {len(block)} most recent block months, not "
                f"the full {BLOCK_FOLDS - 1} (decision B-D1a): a GRU over 32 "
                "sequential timesteps cannot parallelise across time, so the full "
                "block would cost a day and a half of this eight-core box for one "
                "model's hyperparameters.")

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{arm.lower()}-{task.lower()}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  best AUC       {study.best_value:.4f}")
    print(f"  params         {best}")
    print(f"  wrote          {path}")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.arms.tuning",
        description="One-shot Optuna tuning for Arm B (decision B-D1).")
    ap.add_argument("--arm", default="B1", choices=MODELS)
    ap.add_argument("--task", default="T1", choices=("T1", "T2", "T3"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out", default="params")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=TUNE_EPOCHS,
                    help="epochs a B2/B3 trial trains for, with no early stopping "
                         "(ignored by B1, which uses TUNE_ROUNDS)")
    ap.add_argument("--features-dir", default=None)
    ap.add_argument("--studies-dir", default=None,
                    help="where the resumable Optuna study lives "
                         "(default: <out>/../studies)")
    ap.add_argument("--window", default=RULE_SAFE, choices=(RULE_SAFE,))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="torch device for B2/B3; auto selects CUDA when available")
    args = ap.parse_args(argv)

    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    try:
        os.sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                  # pragma: no cover
        pass

    print(f"tuning {args.arm} {args.task} <- {args.manifest}")
    tune(arm=args.arm, task=args.task, manifest_path=args.manifest,
         canonical_path=args.canonical, out=args.out, trials=args.trials,
         seed=args.seed, features_dir=args.features_dir, num_threads=args.threads,
         window=args.window, studies_dir=args.studies_dir, epochs=args.epochs,
         device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
