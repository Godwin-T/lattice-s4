"""
Arm B3 — a GRU over the user's last 32 legal jobs.

`arm-b-planning.md` §7 and §7.1. B3 is the one Arm B learner that reads the
history rather than a summary of it, which changes two things and nothing else.

**It needs a second cache, and that is the whole reason `sequences.py` exists.**
Every other arm reads a *row*: Arm A's four statistics, B1's and B2's 37-column
feature vector. Those are aggregates of a user's prior jobs, and the per-job
`ratio`, `state` and `end_time` they were computed from are dropped by
`build_features`. So the sequence cannot be recovered by reading the feature
cache more carefully (B3-D1); it is built once, per job, into its own artifact
under the same `_ensure_cached` staleness key.

**The window is a slice, not a search** (B3-D2). A target's last 32 legal jobs
are the slice `[idx_end-31, idx_end]` of its user's array, because `idx_end` —
the index of the last job that ended strictly before the target was submitted —
is already known: `common._rolling` computes it for Arm A, and the feature cache
carries it as `hist_n - 1`. That is only true because the sequence source is
sorted by the *same* `(user_hash, end_time, job_id)` key, so "position k in the
array" and "the k-th job in the join's order" mean the same thing. Nothing here
recomputes the ordering, and nothing here restates the leak rule (B3-D6): both
are inherited from the feature layer so they cannot drift from it.

**Padding goes at the front** (B3-D4), so column 31 is always the most recent
job and the last step the GRU sees is never padding. `nn.GRU` gets the sequence
through `pack_padded_sequence`, which means the real jobs have to be moved to the
front of each row first (`_collapse`) — packing reads the valid entries from the
front, so without that it would drop the recent jobs and keep the padding, which
is the exact opposite of the intent and would not raise.

**Scale is handled the way B2 handles it** (B3-D7): a GRU gates on `W x + b` and
is no more scale-blind than a dense layer, so the per-job vectors go through
B2's nan-aware standardiser and its `INPUT_CLIP`. The difference is *what* the
statistics are taken over — B3 fits them on the real jobs of the training
windows, since padding must not drag a mean toward zero — and that the cap is
applied **to targets before the windows are assembled**, because B3 has no
materialised matrix to cap afterwards. 250,000 targets is 256 MB of window
against 4 GB for the uncapped training set, and the uncapped path is the one
that does not fit.

**T1 and T2 only, for now.** §7 gives B3 a T1–T3 ceiling — it is N/A for T4 —
but T3's target does not exist: there is no `label_t3` in the canonical table or
anywhere in the repository, because the PRD builds T3 in stage 3 ("T3 targets;
arm B on T3"), after stage 2's "Arm B (B1–B3) on T1 and T2". So B3's first run is
the same two tasks as B1's and B2's, and `--task T3` would have no column to
read. Adding it is one line in `LABEL_COLUMN` once the target lands.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl

# B1 is the de-facto home of the tabular plumbing (see the same note in
# arm_b2.py), so the shard writers and the control-fold rule come from there.
# The scale treatment comes from B2, which is where it was worked out.
from ..splits.eligibility import eligible_summary, eligible_users
from .arm_b1 import (
    _blank_rows, _control_folds, _read_meta, _write_blank, _write_shard,
)
from .arm_b2 import _standardise, _standardiser
from .common import (
    PREDICTION_SCHEMA, RULE_SAFE, WINDOW_MAX_CHUNK_ROWS, _peak_rss_mb,
    load_manifest, with_row_id,
)
from .features import FEATURE_NAMES, FEATURE_VERSION, ensure_features
from .sequences import (
    N_INPUTS, SEQUENCE_VERSION, SEQ_WINDOW, STATE_CLASSES, SequenceSource,
    ensure_sequences,
)

ARM = "B3"
TASKS = ("T1", "T2")
PARAMS_DIR = "params"


def resolve_device(requested: str = "auto") -> str:
    """Resolve the torch device and fail clearly for an unavailable CUDA request."""
    import torch
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is unavailable")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"

MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 50

# Architecture, `arm-b-planning.md` §7. Untuned until `params/b3-<task>.json`
# exists; the Optuna pass replaces the numbers, not the shape.
HIDDEN = 64
NUM_LAYERS = 1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
# The same reasons as B2's: rows/batch at a 250k cap is 244 updates per epoch at
# 1024, enough that a first epoch cannot win the validation AUC on noise; and
# `min_epochs` keeps a single noisy validation month from being read as a peak
# (`control_gate_finding.md` §9.1).
BATCH_SIZE = 1024
MAX_EPOCHS = 40
MIN_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 5

INPUT_CLIP = 10.0
TRAIN_ROW_CAP = 250_000

DEFAULT_PARAMS = {
    "hidden": HIDDEN,
    "num_layers": NUM_LAYERS,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "min_epochs": MIN_EPOCHS,
    "patience": EARLY_STOPPING_PATIENCE,
}

LABEL_COLUMN = {"T1": "label", "T2": "label_t2"}

# The columns read from the feature cache for a target row. `hist_n` is the only
# one B3 needs that B1/B2 do not: it *is* the window's end index.
_TARGET_COLUMNS = ["user_hash", "hist_n", "label", "label_t2"]


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


def _cap_index(n: int, cap: int | None, seed: int) -> tuple[np.ndarray | None, int]:
    """
    The training targets to keep, deterministically.

    Returns `(index_or_None, rows_used)`. Different in kind from B2's `_cap_rows`,
    not just in shape: B2 caps a matrix it has already built, while B3 must choose
    the targets *before* their windows exist, or the uncapped assembly is the 4 GB
    tensor the plan forbids. `None` means uncapped, so the caller can skip the
    gather rather than copy an arange over four million rows.
    """
    if not cap or n <= cap:
        return None, n
    idx = np.random.default_rng(seed).choice(n, size=cap, replace=False)
    idx.sort()
    return idx, cap


def _raw_windows(src: SequenceSource, user_hashes: np.ndarray,
                 hist_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The assembled, unstandardised windows for a batch of targets.

    Takes the two columns as arrays rather than the frame, because the caller has
    to subset them by `keep` *before* the windows exist, and the labels have
    already been subset the same way. Indexing the frame instead would mean
    relying on `DataFrame[array]`, which for a list of integers selects columns,
    not rows — and a window built from one ordering against a label from another
    would not raise, it would just be wrong.
    """
    idx_end = (np.asarray(hist_n) - 1).astype(np.int64)
    return src.windows(src.codes_for(user_hashes), idx_end)


def _fit_scale(x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The per-component mean and scale, over the *real* jobs only.

    `x[mask]` is every non-padding slot flattened, so the statistics describe the
    jobs the model actually sees. Including the padded zeros would bias each mean
    toward zero by the padding fraction, and B3-D4 guarantees that fraction is
    not zero — up to 8 of 32 slots for the shortest legal histories.
    """
    return _standardiser(x[mask])


def _scaled(x: np.ndarray, mask: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Standardise, clip, re-zero the padding.

    The re-zeroing is not for the GRU — `pack_padded_sequence` never reads a
    padded slot — but so the array keeps the invariant `sequences.windows` and
    its tests assert: a padded slot is zero, and `mask` is the only thing that
    says which slots are real.
    """
    return _standardise(x, mu, sd) * mask[..., None]


def _collapse(x, mask):
    """
    Left-align each window's real jobs, preserving their order.

    `pack_padded_sequence` reads the valid entries from the *front* of each row,
    and B3 pads at the front (B3-D4), so the two disagree by construction. Left
    unaligned, packing would keep the padding and drop the recent jobs — a
    silently wrong model, not an error. `stable=True` keeps each user's jobs in
    chronological order, so the final hidden state is still the one that
    summarised the job that just ended.
    """
    import torch

    lengths = mask.sum(dim=1)
    order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
    return torch.gather(x, 1, order.unsqueeze(-1).expand_as(x)), lengths


def _gru(n_inputs: int, params: dict):
    """
    The recurrent stack, emitting one logit per target.

    `dropout` is passed to the GRU only when it has more than one layer, because
    PyTorch warns — and the warning is right — that inter-layer dropout on a
    single layer applies to nothing.
    """
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    layers = int(params["num_layers"])

    class _GRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_inputs, int(params["hidden"]), num_layers=layers,
                              batch_first=True,
                              dropout=params["dropout"] if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(params["dropout"]),
                                      nn.Linear(int(params["hidden"]), 1))

        def forward(self, x, mask):
            gathered, lengths = _collapse(x, mask)
            # Clamped to 1 because `pack_padded_sequence` rejects a zero length,
            # and a target with no legal history (`idx_end < 0`) is real: it is
            # unscoreable and never scored, but the forward is still called on it
            # and must not raise.
            packed = pack_padded_sequence(gathered, lengths.clamp(min=1),
                                          batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            emb = h[-1]
            # A zero-length row's one-step pass over zero padding would otherwise
            # emit a bias-driven embedding indistinguishable from a real one.
            emb = emb * (lengths > 0).unsqueeze(1).to(emb.dtype)
            return self.head(emb).squeeze(-1)

    return _GRU()


def _val_auc(model, x_val, m_val, y_val) -> float:
    from sklearn.metrics import roc_auc_score

    if x_val is None or y_val is None or not (0 < int(y_val.sum()) < len(y_val)):
        return float("nan")
    return float(roc_auc_score(y_val, _predict(model, x_val, m_val)))


def _fit(x_train, m_train, y_train, x_val, m_val, y_val, params, *, seed: int,
         epochs: int | None = None, device: str = "cpu"):
    """
    Fit one fold's GRU, returning `(model, epochs_run, best_val_auc)`.

    Same rule as B1 and B2 on the same inner-validation month: train up to
    `max_epochs`, keep the weights that scored best on it, and let the first
    `min_epochs` of those epochs be warmup with no checkpoint at all. Without an
    inner month (the first fold) it trains `max_epochs` and keeps the last; the
    control always trains exactly `epochs` rounds, so the shuffle is compared
    against the arm's own procedure rather than a longer one.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = _gru(x_train.shape[-1], params).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["learning_rate"],
                           weight_decay=params.get("weight_decay", 0.0))
    loss_fn = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)).to(device)
    mt = torch.from_numpy(np.ascontiguousarray(m_train)).to(device)
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
            loss_fn(model(xt[idx], mt[idx]), yt[idx]).backward()
            opt.step()
        ran = epoch + 1
        if not score_val or ran < min_epochs:
            continue
        auc = _val_auc(model, x_val, m_val, y_val)
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


def _predict(model, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The model's probability of the positive class, as `float64`."""
    import torch

    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        logits = model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device),
                       torch.from_numpy(np.ascontiguousarray(mask)).to(device))
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
        raise SystemExit(f"Arm B3 runs only under the {RULE_SAFE!r} rule")
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

    # The sequence source, from the same eligible frame under the same rule. B3
    # still needs the feature cache — for `scoreable`, `month`, `hist_n` and the
    # labels — but the windows come from here.
    seq_path = ensure_sequences(eligible, canonical_path, cache_dir,
                                rebuild=rebuild_features,
                                max_chunk_rows=max_chunk_rows, rule=window)
    src = SequenceSource.load(seq_path)
    print(f"  features       {cache} | {len(FEATURE_NAMES)} columns | "
          f"v{FEATURE_VERSION}", flush=True)
    print(f"  sequences      {seq_path} | {src.n_jobs:,} jobs over "
          f"{src.n_users:,} users | window {SEQ_WINDOW} x {N_INPUTS} | "
          f"v{SEQUENCE_VERSION}", flush=True)
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
        p, src_path = load_params(t, params_dir)
        params_by_task[t], source_by_task[t] = p, src_path
        print(f"  params {t:<8} {src_path or 'defaults (untuned)'}", flush=True)

    state = {
        t: {"shards": [], "fold_aucs": [], "fold_auc_by_fold": {},
            "control_aucs": [], "control_folds_used": [], "skipped": [],
            "fit_seconds": 0.0, "inference_seconds": 0.0, "epochs": [],
            "train_rows_used": [], "val_aucs": [], "n_control_folds_done": 0}
        for t in tasks
    }
    n_test = n_unscorable = n_blank = 0
    # The most front-padding any scored window needed, tracked across the run so
    # a reader can see that the mask is doing more than B3-D4's worst case of 8.
    padding_max = 0
    scratch = Path(tempfile.mkdtemp(prefix="preds_b3_"))

    try:
        for fold in folds:
            inner_months = fold["train_months"][:-1]
            inner_month = fold["train_months"][-1]

            train = (feats
                     .filter(pl.col("scoreable")
                             & pl.col("month").is_in(inner_months))
                     .select([*_TARGET_COLUMNS]).collect())
            val = (feats
                   .filter(pl.col("scoreable") & (pl.col("month") == inner_month))
                   .select([*_TARGET_COLUMNS]).collect()
                   if inner_months else None)
            test = (feats
                    .filter(pl.col("month") == fold["test_month"])
                    .select(["job_id", "row_id", "month", "scoreable",
                             *_TARGET_COLUMNS, "energy_j"]).collect())
            scoreable = test.filter(pl.col("scoreable"))
            n_test += test.height
            n_unscorable += test.height - scoreable.height

            if train.height < MIN_TRAIN_ROWS or scoreable.height < MIN_TEST_ROWS:
                # A skipped fold writes every row blank, including the rows that
                # *are* scoreable — so it adds to `blank_rows` beyond
                # `unscorable_rows`, which counts only the rows the split cannot
                # score. Keeping the two apart is what makes "how much of this
                # hand-in table is empty, and why" answerable from metadata.
                n_blank += test.height
                for t in tasks:
                    state[t]["skipped"].append({
                        "fold_id": fold["fold_id"], "test_month": fold["test_month"],
                        "train_rows": train.height, "test_rows": scoreable.height})
                    state[t]["shards"].append(_write_blank(
                        test, scratch, t, fold, run_ids[t], arm=ARM))
                del train, val, test, scoreable
                continue
            n_blank += test.height - scoreable.height

            # Cap the targets, then assemble only their windows. The order
            # matters: assembling first and subsetting after would build the
            # uncapped tensor, which is the 4 GB case `_cap_index` exists to
            # avoid.
            keep, used = _cap_index(train.height, train_row_cap, seed)
            hashes = train["user_hash"].to_numpy()
            hist_n = train["hist_n"].to_numpy()
            y_train = {t: train[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                       for t in tasks}
            del train
            if keep is not None:
                hashes, hist_n = hashes[keep], hist_n[keep]
                y_train = {t: y[keep] for t, y in y_train.items()}

            x_train, m_train = _raw_windows(src, hashes, hist_n)
            del hashes, hist_n
            # Fit the scale on exactly the jobs the model will train on, and not
            # on the padding: `_fit_scale` reads the masked slots only.
            mu, sd = _fit_scale(x_train, m_train)
            x_train = _scaled(x_train, m_train, mu, sd)

            if val is not None and val.height:
                x_val, m_val = _raw_windows(src, val["user_hash"].to_numpy(),
                                            val["hist_n"].to_numpy())
                x_val = _scaled(x_val, m_val, mu, sd)
                y_val = {t: val[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                         for t in tasks}
            else:
                x_val = m_val = y_val = None
            del val

            if scoreable.height:
                # B3-D4's worst case is 8 padded slots (`scoreable` needs 24 prior
                # jobs of the 32); tracked so a reader can see the real figure.
                padding_max = max(padding_max,
                                  SEQ_WINDOW - int(scoreable["hist_n"].min()))
                x_test, m_test = _raw_windows(src, scoreable["user_hash"].to_numpy(),
                                              scoreable["hist_n"].to_numpy())
                x_test = _scaled(x_test, m_test, mu, sd)
                test_labels = {t: scoreable[LABEL_COLUMN[t]].to_numpy().astype(np.float32)
                               for t in tasks}
            else:
                x_test = m_test = test_labels = None
            del scoreable

            for t in tasks:
                started = time.time()
                model, epochs_ran, val_auc = _fit(
                    x_train, m_train, y_train[t], x_val, m_val,
                    (y_val[t] if y_val else None), params_by_task[t], seed=seed,
                    device=device)
                state[t]["fit_seconds"] += time.time() - started
                state[t]["epochs"].append(epochs_ran)
                state[t]["train_rows_used"].append(used)
                if np.isfinite(val_auc):
                    state[t]["val_aucs"].append(val_auc)

                started = time.time()
                scores = (_predict(model, x_test, m_test)
                          if x_test is not None else np.array([]))
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
                        c_model, _, _ = _fit(x_train, m_train, shuffled, None, None,
                                             None, params_by_task[t], seed=seed,
                                             epochs=epochs_ran, device=device)
                        c_scores = _predict(c_model, x_test, m_test)
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
            del x_train, m_train, y_train, x_val, m_val, y_val, x_test, m_test, \
                test_labels, test

        for t in tasks:
            _finish_task(t, state[t], run_dirs[t], run_ids[t], manifest,
                         manifest_path, canonical_path, tasks, seed,
                         control_repeats, control_set, n_test, n_unscorable,
                         n_blank, params_by_task[t], source_by_task[t], len(folds),
                         cache, seq_path, num_threads, window, train_row_cap, src,
                         padding_max, device)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    primary = tasks[0]
    meta = _read_meta(run_dirs[primary])
    meta["tasks"] = {t: _read_meta(run_dirs[t]) for t in tasks}
    return meta


def _finish_task(task, acc, run_dir: Path, run_id: str, manifest, manifest_path,
                 canonical_path, tasks, seed, control_repeats, control_set,
                 n_test, n_unscorable, n_blank, params, source, n_folds, cache,
                 seq_path, num_threads, window, train_row_cap, src, padding_max,
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
        # Rows written with a null score: the unscoreable ones, plus every row of
        # any fold that was skipped for too little training or test data. The two
        # differ, and only this one is "how much of the table is empty".
        "blank_rows": n_blank,
        "blank_fraction": (n_blank / n_test) if n_test else None,
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
        "learner": "pytorch-gru",
        "device": device,
        "architecture": {"inputs": N_INPUTS, "hidden": int(params["hidden"]),
                         "layers": int(params["num_layers"]),
                         "dropout": params["dropout"],
                         "window": SEQ_WINDOW, "outputs": 1},
        # The sequence contract, recorded where the other arms record the feature
        # vocabulary: a reader must be able to tell which per-job vector and which
        # window width produced these numbers without reading the cache.
        "sequence_source": str(seq_path),
        "sequence_version": SEQUENCE_VERSION,
        "sequence_jobs": src.n_jobs,
        "sequence_users": src.n_users,
        "seq_window": SEQ_WINDOW,
        "n_inputs": N_INPUTS,
        "state_classes": list(STATE_CLASSES),
        "window_padding_slots_max": padding_max,
        "num_threads": num_threads,
        "params": params,
        "params_source": source,
        "tuned": source is not None,
        # B2/B3-only and visible, so a capped run is never silently read as
        # comparable to B1's uncapped one (`arm-b-planning.md` §7).
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
