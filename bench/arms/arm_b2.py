"""
Arm B2 — a dense MLP over the shared Arm B feature layer.

`arm-b-planning.md` §7. Same folds, same cache, same two tasks as B1; the
learner is the only thing that changes, because the point of a second Arm B
learner is to see whether the feature layer's signal survives a model that
cannot lean on LightGBM's native categorical handling.

**The categoricals are one-hot, not ordinal.** B1 passes the three string
columns to LightGBM as native categoricals; a dense layer has no such facility
and an integer code would assert an order the levels do not have (is partition
12 "between" 11 and 13?). So the codes are expanded to indicators here — 42 + 9
+ 5 = 56 of them over 34 numeric columns, 90 inputs, against the "~35" figure
in the plan, which was written before the vocabulary was fixed. A null code
lights no indicator, so an all-zero block means "unknown"; that is the encoding
B1 gets for free, and it is why the vocabulary is taken from the whole cache
rather than per fold.

**Scale is handled explicitly, because a dense layer is not scale-blind.** Two
problems a tree never has. First, nulls: the numeric features may be null
(Eagle's `hist_queue_*` are null on every row), and `sklearn`'s
`StandardScaler` would turn a null column into a null mean and then a null column
of inputs, so `_standardiser` uses `nanmean`/`nanstd` and fills what is left with
0 — the column mean *after* standardising — leaving an all-null column as a
constant zero, which carries no information and is honestly nothing. Second, the
tails: `hist_ratio_min`'s training sd on Eagle's 2018-12 is ~4e-6, so a
next-month value standardises to ~3.6e+03, and a weight times that saturates the
logit (the first smoke run diverged this way). `_standardise` clips every input
to ±`INPUT_CLIP` sd.

**The training-row cap (decision in `arm-b-planning.md` §7).** 45 folds × full
epochs is not survivable on 8 cores, so each fold's training set is subsampled
to `train_row_cap` rows — deterministically, and applied so the control sees the
same rows. B1 is never capped. The cap and the rows actually used are recorded
in every metadata file, because a capped run is not directly comparable to an
uncapped one and a reader must be able to see which they are holding.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import polars as pl

# B1 is the de-facto home of the tabular plumbing — `tuning.py` and
# `xgb_check.py` already import its privates — so the cache readers, the shard
# writers, and the control-fold rule are taken from there rather than copied.
# Only the learner and its two consequences (one-hot, the row cap) live here.
from ..splits.eligibility import eligible_summary, eligible_users
from .arm_b1 import (
    CATEGORY_INDICES, _blank_rows, _categories, _control_folds, _feature_exprs,
    _matrix, _read_meta, _write_blank, _write_shard,
)
from .common import (
    PREDICTION_SCHEMA, RULE_SAFE, WINDOW_MAX_CHUNK_ROWS, _peak_rss_mb,
    load_manifest, with_row_id,
)
from .features import (
    CATEGORICAL_FEATURES, FEATURE_NAMES, FEATURE_VERSION, ensure_features,
)

ARM = "B2"
TASKS = ("T1", "T2")
PARAMS_DIR = "params"

MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 50


def resolve_device(requested: str = "auto") -> str:
    """Resolve the torch device and fail clearly for an unavailable CUDA request."""
    import torch
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"

# Architecture, `arm-b-planning.md` §7. Untuned until `params/b2-<task>.json`
# exists; the Optuna pass replaces the numbers, not the shape.
HIDDEN = (256, 128)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
# 1024, not a round 4096: `steps per epoch = rows / batch`, and a cap of 250k
# rows at 4096 is only 61 updates per epoch — few enough that a first epoch can
# win the validation AUC on noise alone, at which point early stopping keeps a
# model that has barely left its initialisation. The smoke run showed exactly
# that (it kept epoch 0). The batch is a hyperparameter and the tuning pass owns
# it, but the default has to be one that trains.
BATCH_SIZE = 1024
MAX_EPOCHS = 40
# Early stopping may not fire before this many epochs. A single validation month
# is a noisy estimate — Eagle's folds carry 110–283 independent users
# (`control_gate_finding.md` §9.1) — so a "best" epoch drawn from the first
# handful is often a coin landing, not a model peaking.
MIN_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 5

# The most a single standardised input may contribute, in standard deviations
# (see `_standardise`). The bulk of every column is far inside it.
INPUT_CLIP = 10.0

# B2/B3 only. The largest cap that keeps 45 folds × epochs inside the box's
# memory and time; B1 is uncapped, and the asymmetry is recorded, not hidden.
TRAIN_ROW_CAP = 250_000

DEFAULT_PARAMS = {
    "hidden": list(HIDDEN),
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "min_epochs": MIN_EPOCHS,
    "patience": EARLY_STOPPING_PATIENCE,
}

LABEL_COLUMN = {"T1": "label", "T2": "label_t2"}


def load_params(task: str, params_dir: str | Path = PARAMS_DIR) -> tuple[dict, str | None]:
    """
    The frozen hyperparameters for a task, and where they came from.

    Returns `(params, source)`. `source` is None when no tuned file exists, which
    the run records so an untuned result can never be read as tuned (B-D1).
    """
    path = Path(params_dir) / f"{ARM.lower()}-{task.lower()}.json"
    if not path.exists():
        return dict(DEFAULT_PARAMS), None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {**DEFAULT_PARAMS, **payload["params"]}, str(path)


def _onehot(x: np.ndarray, categories: dict[str, list[str]]) -> np.ndarray:
    """
    Expand the integer-coded categorical columns to one-hot indicators.

    A code of `k` lights indicator `k`; a null code lights none of them, so an
    all-zero block is the encoding of "unknown". The numeric columns keep their
    order and come first, so the input layout is a function of `FEATURE_NAMES`
    and the vocabulary alone — the same for every fold.
    """
    blocks = [np.delete(x, CATEGORY_INDICES, axis=1)]
    for j in CATEGORY_INDICES:
        n_levels = len(categories[FEATURE_NAMES[j]])
        codes = x[:, j]
        block = np.zeros((x.shape[0], n_levels), dtype=np.float32)
        valid = ~np.isnan(codes)
        if valid.any():
            rows = np.nonzero(valid)[0]
            block[rows, codes[valid].astype(np.int64)] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _standardiser(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-column mean and scale over the **non-null** values.

    Not `sklearn`'s: that one propagates NaN into the mean and then into every
    row of the column. A column with no variation (or no values at all) gets a
    scale of 1, so it standardises to a constant rather than to 0/0.
    """
    # An all-null column makes `nanmean`/`nanstd` warn about an empty slice
    # before returning NaN — which the next two lines then handle. Silenced
    # because it is the expected input here, not a surprise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0).astype(np.float32)
    return mu, sd


def _standardise(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Standardise, fill what is still null with 0, then clip.

    The clip is not cosmetic. Several features are heavy-tailed or near-degenerate
    on a training month — `hist_ratio_min` has a training sd of ~4e-6 on Eagle's
    2018-12, so a next-month value standardises to ~3.6e+03. A tree is blind to
    that (it splits on rank), but a dense layer multiplies it by a weight and the
    logit saturates: the first smoke run diverged this way, predicting
    probabilities of exactly 0 with `exp` overflowing. Bounding every standardised
    input to ±`INPUT_CLIP` bounds what any one feature can contribute.
    """
    out = (x - mu) / sd
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -INPUT_CLIP, INPUT_CLIP).astype(np.float32, copy=False)


def _cap_rows(x: np.ndarray, cap: int | None, seed: int):
    """
    Cap the training rows, deterministically and once.

    Returns `(x_capped, rows_used, index_or_None)` — the index so the *labels*
    can be cut to the same rows without redrawing. The draw is seeded rather
    than taken from the global state, and the control refits on the same rows:
    a cap that differed between the arm and its control would put a second
    difference into the comparison and quietly stop it being a shuffle test.
    """
    n = x.shape[0]
    if not cap or n <= cap:
        return x, n, None
    idx = np.random.default_rng(seed).choice(n, size=cap, replace=False)
    idx.sort()
    return x[idx], cap, idx


def _model(n_inputs: int, params: dict):
    """
    The fully-connected stack, emitting one logit per row.

    The final `squeeze(-1)` is not cosmetic: `BCEWithLogitsLoss` wants a target
    of the same shape as its input, so a `(n, 1)` output against an `(n,)` label
    is a shape error, not a broadcast.
    """
    import torch.nn as nn

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers: list = []
            width = n_inputs
            for h in params["hidden"]:
                layers += [nn.Linear(width, int(h)), nn.ReLU(),
                           nn.Dropout(params["dropout"])]
                width = int(h)
            layers += [nn.Linear(width, 1)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return _MLP()


def _val_auc(model, x_val, y_val) -> float:
    from sklearn.metrics import roc_auc_score

    if x_val is None or y_val is None or not (0 < int(y_val.sum()) < len(y_val)):
        return float("nan")
    return float(roc_auc_score(y_val, _predict(model, x_val)))


def _fit(x_train, y_train, x_val, y_val, params, *, seed: int,
         epochs: int | None = None, device: str = "cpu"):
    """
    Fit one fold's MLP, returning `(model, epochs_run, best_val_auc)`.

    With an inner-validation month the model trains up to `max_epochs` and keeps
    the weights that scored best on it — the same rule B1 uses, on the same
    month. Without one (the first fold) it trains `max_epochs` and keeps the
    last; the control always trains exactly `epochs` rounds, so the shuffle is
    compared against the arm's own procedure rather than a longer one.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = _model(x_train.shape[1], params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                           weight_decay=params.get("weight_decay", 0.0))
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)).to(device)
    n = xt.shape[0]
    batch = int(params["batch_size"])
    patience = int(params.get("patience", EARLY_STOPPING_PATIENCE))
    min_epochs = int(params.get("min_epochs", MIN_EPOCHS))
    budget = int(epochs if epochs is not None else params["max_epochs"])
    score_val = epochs is None and x_val is not None and y_val is not None \
        and 0 < int(y_val.sum()) < len(y_val)

    best_state, best_auc, bad = None, -np.inf, 0
    ran = 0
    for epoch in range(budget):
        model.train()
        order = torch.randperm(n, device=device)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            opt.zero_grad()
            loss_fn(model(xt[idx]), yt[idx]).backward()
            opt.step()
        ran = epoch + 1
        # The first `min_epochs` are warmup: no scoring, no stopping, and — the
        # point of it — no checkpoint. Gating only the patience counter would
        # still let a noisy validation month pick epoch 0 as the best epoch and
        # hand back an untrained model, which is the failure the smoke run hit.
        if not score_val or ran < min_epochs:
            continue
        auc = _val_auc(model, x_val, y_val)
        if auc > best_auc + 1e-4:
            best_state, best_auc, bad = \
                {k: v.detach().clone() for k, v in model.state_dict().items()}, auc, 0
        else:
            bad += 1
            if bad >= patience:
                break

    if score_val and best_state is not None:
        model.load_state_dict(best_state)
    return model, ran, best_auc if best_state is not None else float("nan")


def _predict(model, x: np.ndarray) -> np.ndarray:
    """
    The model's probability of the positive class, as `float64`.

    `torch.sigmoid`, not the textbook expression: `1/(1+exp(-z))` overflows for
    `z < -745` and returns a silent 0 with a `RuntimeWarning`, which is how the
    diverging first smoke run announced itself.
    """
    import torch

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(torch.from_numpy(
            np.ascontiguousarray(x, dtype=np.float32)).to(device))
        return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype(np.float64)


def run(*, manifest_path: str, canonical_path: str, outdir: str,
        tasks: tuple[str, ...] = TASKS, seed: int = 42, control_repeats: int = 3,
        control_folds: int | None = None, features_dir: str | None = None,
        rebuild_features: bool = False, folds_limit: int | None = None,
        max_chunk_rows: int = WINDOW_MAX_CHUNK_ROWS,
        params_dir: str = PARAMS_DIR, num_threads: int = 2,
        train_row_cap: int | None = TRAIN_ROW_CAP,
        window: str = RULE_SAFE, device: str = "auto") -> dict:
    """
    Score every fold of a manifest for each task, and write the hand-in tables.

    Returns the primary task's metadata, with every task's under `"tasks"`.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    if window != RULE_SAFE:
        raise SystemExit(f"Arm B2 runs only under the {RULE_SAFE!r} rule")
    tasks = tuple(tasks)
    if not tasks:
        raise SystemExit("no tasks requested")
    torch.set_num_threads(max(1, num_threads))
    device = resolve_device(device)

    manifest = load_manifest(manifest_path)
    dataset = manifest["dataset"]

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
    n_inputs = len(FEATURE_NAMES) - len(CATEGORICAL_FEATURES) \
        + sum(len(categories[c]) for c in CATEGORICAL_FEATURES)
    print(f"  features       {cache} | {len(FEATURE_NAMES)} columns "
          f"({len(CATEGORICAL_FEATURES)} categorical) -> {n_inputs} inputs "
          f"| v{FEATURE_VERSION}", flush=True)
    print(f"  train row cap  {train_row_cap:,}" if train_row_cap else
          "  train row cap  none", flush=True)

    folds = manifest["folds"]
    if folds_limit:
        folds = folds[:folds_limit]
    control_set = _control_folds(folds, control_folds)

    run_ids = {t: f"{ARM}-{dataset}-{window}-{t.lower()}" for t in tasks}
    run_dirs = {t: Path(outdir) / run_ids[t] for t in tasks}
    for d in run_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    params_by_task, source_by_task = {}, {}
    for t in tasks:
        p, src = load_params(t, params_dir)
        params_by_task[t], source_by_task[t] = p, src
        print(f"  params {t:<8} {src or 'defaults (untuned)'}", flush=True)

    state = {
        t: {"shards": [], "fold_aucs": [], "fold_auc_by_fold": {},
            "control_aucs": [], "control_folds_used": [], "skipped": [],
            "fit_seconds": 0.0, "inference_seconds": 0.0, "epochs": [],
            "train_rows_used": [], "val_aucs": [], "n_control_folds_done": 0}
        for t in tasks
    }
    n_test = n_unscorable = 0
    scratch = Path(tempfile.mkdtemp(prefix="preds_b2_"))

    try:
        for fold in folds:
            inner_months = fold["train_months"][:-1]
            inner_month = fold["train_months"][-1]

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
                        test, scratch, t, fold, run_ids[t], arm=ARM))
                del train, val, test
                continue

            # Cap, then one-hot, then standardise — in that order, so the cap is
            # a cap on what the model sees and the standardiser is fit on
            # exactly that. The vocabulary comes from the whole cache either way.
            labels_raw = {t: train[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                          for t in tasks}
            x_raw, used, kept = _cap_rows(_matrix(train, exprs), train_row_cap, seed)
            y_train = {t: (y[kept] if kept is not None else y)
                       for t, y in labels_raw.items()}
            del labels_raw
            x_expanded = _onehot(x_raw, categories)
            mu, sd = _standardiser(x_expanded)
            x_train = _standardise(x_expanded, mu, sd)
            del x_raw, x_expanded
            x_val = (_standardise(_onehot(_matrix(val, exprs), categories), mu, sd)
                     if val is not None and val.height else None)
            y_val = ({t: val[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                      for t in tasks} if x_val is not None else None)
            del train, val
            x_test = (_standardise(_onehot(_matrix(scoreable, exprs), categories),
                                   mu, sd) if scoreable.height else None)
            test_labels = ({
                t: scoreable[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                for t in tasks} if x_test is not None else None)
            del scoreable

            for t in tasks:
                started = time.time()
                model, epochs_ran, val_auc = _fit(
                    x_train, y_train[t], x_val, (y_val[t] if y_val else None),
                    params_by_task[t], seed=seed, device=device)
                state[t]["fit_seconds"] += time.time() - started
                state[t]["epochs"].append(epochs_ran)
                state[t]["train_rows_used"].append(used)
                if np.isfinite(val_auc):
                    state[t]["val_aucs"].append(val_auc)

                started = time.time()
                scores = _predict(model, x_test) if x_test is not None else np.array([])
                predict_seconds = time.time() - started
                state[t]["inference_seconds"] += predict_seconds
                latency_ms = (1000.0 * predict_seconds / x_test.shape[0]
                              if x_test is not None and x_test.shape[0] else None)
                del model

                frame = _blank_rows(test, run_ids[t], t, fold, arm=ARM)
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
                        c_model, _, _ = _fit(x_train, shuffled, None, None,
                                             params_by_task[t], seed=seed,
                                             epochs=epochs_ran, device=device)
                        c_scores = _predict(c_model, x_test)
                        del c_model
                        control_auc = float(roc_auc_score(test_labels[t], c_scores))
                        if not np.isnan(control_auc):
                            state[t]["control_aucs"].append(control_auc)
                            state[t]["control_folds_used"].append(int(fold["fold_id"]))
                    state[t]["n_control_folds_done"] += 1

            print(f"  fold {fold['fold_id']:>2}: train {used:>9,} "
                  f"test {test.height:>8,} "
                  f"| epochs {'/'.join(str(state[t]['epochs'][-1]) for t in tasks)} "
                  f"| rss {_peak_rss_mb():>6,.0f} MB", flush=True)
            del x_train, y_train, x_val, y_val, x_test, test_labels

        for t in tasks:
            _finish_task(t, state[t], run_dirs[t], run_ids[t], manifest,
                         manifest_path, canonical_path, tasks, seed,
                         control_repeats, control_set, n_test, n_unscorable,
                         params_by_task[t], source_by_task[t], len(folds), cache,
                         num_threads, window, train_row_cap, n_inputs, categories,
                         device)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    primary = tasks[0]
    meta = _read_meta(run_dirs[primary])
    meta["tasks"] = {t: _read_meta(run_dirs[t]) for t in tasks}
    return meta


def _finish_task(task, acc, run_dir: Path, run_id: str, manifest, manifest_path,
                 canonical_path, tasks, seed, control_repeats, control_set,
                 n_test, n_unscorable, params, source, n_folds, cache,
                 num_threads, window, train_row_cap, n_inputs, categories,
                 device):
    """Merge a task's shards and write its metadata."""
    path = run_dir / "predictions.parquet"
    if acc["shards"]:
        pl.scan_parquet([str(p) for p in acc["shards"]]).sink_parquet(path)
    else:                                      # pragma: no cover - no folds ran
        pl.DataFrame(schema=PREDICTION_SCHEMA).write_parquet(path)

    aucs = acc["fold_aucs"]
    used = acc["train_rows_used"]
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
        "features": str(cache),
        "feature_version": FEATURE_VERSION,
        "n_features": len(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        # The vocabulary sizes are what make the input width auditable: one-hot
        # turns 42 partition levels into 42 columns, and a reader cannot check
        # that from `inputs` alone.
        "categorical_levels": {c: len(categories[c]) for c in CATEGORICAL_FEATURES},
        "learner": "pytorch-mlp",
        "architecture": {"inputs": n_inputs, "hidden": list(params["hidden"]),
                         "activation": "relu", "dropout": params["dropout"],
                         "outputs": 1},
        "num_threads": num_threads,
        "device": device,
        "params": params,
        "params_source": source,
        "tuned": source is not None,
        # The cap is B2/B3-only and visible, so a capped run is never silently
        # read as comparable to B1's uncapped one (`arm-b-planning.md` §7).
        "train_row_cap": train_row_cap,
        "train_rows_used": used,
        "train_rows_used_median": float(np.median(used)) if used else None,
        "epochs": acc["epochs"],
        "epochs_median": float(np.median(acc["epochs"])) if acc["epochs"] else None,
        "val_auc_median": (float(np.median(acc["val_aucs"]))
                           if acc["val_aucs"] else None),
        "fold_auc": aucs,
        "fold_auc_median": float(np.median(aucs)) if aucs else None,
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


if __name__ == "__main__":                   # pragma: no cover
    from .cli import main

    raise SystemExit(main())
