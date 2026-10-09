"""
Arm B3's sequence source — the per-job history B2 and B1 aggregate away.

`features.py` keeps only *aggregates* of a user's history (`hist_ratio_mean`,
`hist_ratio_std`, …). A per-job `ratio`, a per-job `state` and a per-job
`end_time` are computed inside `build_features` and then dropped, so the
37-feature cache cannot be un-aggregated back into a sequence. B3 needs the
sequence, so this module builds it: one row per canonical job, carrying that
job's own four-vector, ordered by the *same* key the feature layer orders by.

Why a second cache rather than a join at run time is `arm-b-planning.md` §7.1
(B3-D1..D7); the two decisions worth restating here, because they are the ones
that make this cheap and correct:

**The window is a slice, not a search** (B3-D2). Every target already knows its
position in its user's end-ordered history — `idx_end`, which `common._rolling`
computes, and which the feature cache already carries as `hist_n - 1`. So the
32-job window is the slice `[idx_end-31, idx_end]` of its user's array. There is
no "last K ended-before" join anywhere in this module, and there must not be one:
a windowed join over nine million rows is the ~9 GB hazard the plan names.

**Sorted by `(user_hash, end_time, job_id)` is what makes that true.** It is the
ordering `common._rolling` uses, and because `end_time` is non-decreasing in it,
`idx_end` — the last row ending strictly before the target's submit — is the
*last* such index, so the legal history is exactly the prefix `[0, idx_end]`.
That is the whole reason a contiguous slice can stand in for an as-of join, and
it is asserted rather than assumed in `test_sequences.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import common
from .common import RULE_SAFE
from .features import FAIL_STATES

# Bump when a per-job vector component or its meaning changes, so a stale cache
# is rebuilt rather than silently mis-read — the same discipline as
# `FEATURE_VERSION`, and independent of it (the two caches change for different
# reasons).
SEQUENCE_VERSION = 1

# `folds_manifest.md` §2.1 makes a 32-job window explicitly legal.
SEQ_WINDOW = 32

# The five classes `hist_last_state` collapses to, in the same order and by the
# same rule, so a sequence's "most recent job" and B1's `hist_last_state` describe
# the same partition of the same data (B3-D5). `CANCELLED` and `TIMEOUT` are
# tested *before* `FAIL_STATES` for the reason `build_features` tests them first:
# the ordering is what stops a state that is in both sets being swallowed.
STATE_CLASSES = ("COMPLETED", "TIMEOUT", "CANCELLED", "FAILED", "OTHER")

# Per-job vector: three numeric components, then the one-hot state block.
NUMERIC_FIELDS = ("ratio", "timelimit_log", "cpus")
N_INPUTS = len(NUMERIC_FIELDS) + len(STATE_CLASSES)      # 8

# The columns read from the canonical table. As in `features.py`, `name`,
# `work_dir` and `submit_line` are never named, so they are never decoded.
_SOURCE_COLUMNS = ["job_id", "row_id", "user_hash", "submit_time", "end_time",
                   "elapsed_s", "timelimit_s", "state", "cpus_req"]

# One row per job, in the ordering that defines `pos`. `user_hash` is kept so a
# reader can group; the loader turns it into a code.
_ORDER_COLUMNS = ["user_hash", "end_time", "job_id"]
_OUTPUT_COLUMNS = [*_ORDER_COLUMNS, "row_id", "pos", "state_class", *NUMERIC_FIELDS]


def _state_class(expr: pl.Expr) -> pl.Expr:
    """A job's state collapsed to `STATE_CLASSES`, by `build_features`' own rule."""
    return (pl.when(expr == "COMPLETED").then(pl.lit("COMPLETED"))
            .when(expr == "TIMEOUT").then(pl.lit("TIMEOUT"))
            .when(expr == "CANCELLED").then(pl.lit("CANCELLED"))
            .when(expr.is_in(FAIL_STATES)).then(pl.lit("FAILED"))
            .otherwise(pl.lit("OTHER")))


def build_sequences(lf: pl.LazyFrame, rule: str = RULE_SAFE) -> pl.LazyFrame:
    """
    Canonical rows -> one row per job, ordered for the prefix-slice guarantee.

    One row per *input* row, not a row per target: this table is the history, and
    the windows are cut out of it at run time. `pos` is the job's position within
    its user under `(user_hash, end_time, job_id)` — the same quantity
    `common._rolling` calls `idx`, so a target's `hist_n - 1` indexes straight
    into this table's per-user block.

    `ratio` is byte-for-byte the expression the window features use, for the same
    reason `features.build_features` uses it: so the sequence and Arm A's four
    features are the same quantity, not merely similar ones.
    """
    if rule != RULE_SAFE:
        raise ValueError(
            f"Arm B builds only under {RULE_SAFE!r}, got {rule!r} "
            "(see arm-b-planning.md §5.1)")

    ordered = (lf.select(_SOURCE_COLUMNS)
               .with_columns((pl.col("elapsed_s") / pl.col("timelimit_s")).alias("ratio"))
               .sort(_ORDER_COLUMNS))

    return ordered.with_columns([
        pl.int_range(pl.len()).over("user_hash").cast(pl.Int32).alias("pos"),
        _state_class(pl.col("state")).alias("state_class"),
        pl.col("timelimit_s").log1p().alias("timelimit_log"),
        pl.col("cpus_req").cast(pl.Float64).alias("cpus"),
    ]).select(_OUTPUT_COLUMNS)


def ensure_sequences(lf: pl.LazyFrame, canonical_path: str | Path,
                     cache_dir: str | Path, rebuild: bool = False,
                     max_chunk_rows: int = common.WINDOW_MAX_CHUNK_ROWS,
                     rule: str = RULE_SAFE) -> Path:
    """
    Build the sequence table once and keep it as Parquet.

    **Chunking is safe here, and for a stronger reason than for the features.**
    `_ensure_cached` packs whole users per chunk, and this table's only cross-row
    quantity is `pos`, which is a per-user `int_range` — so a chunk's output
    depends on that chunk alone, exactly as the aggregates do. The sort is
    global, but it orders *rows*, not users, and users are already whole.

    Both of `_ensure_cached`'s defaults are overridden, and each for its own
    reason: there is no `scoreable` on a per-job row, and — the one that would
    bite — month-clustering would partition each user's jobs across month files
    and interleave the users when they are concatenated back, so the contiguity
    `SequenceSource.load` slices on would be gone. Skipping the clustering also
    costs nothing: B3 reads this table whole, once per run, and never issues the
    per-fold month filter the clustering exists to speed up.
    """
    if rule != RULE_SAFE:
        raise ValueError(f"Arm B builds only under {RULE_SAFE!r}, got {rule!r}")
    return common._ensure_cached(
        lf, canonical_path, cache_dir,
        filename=f"{Path(canonical_path).stem}.{rule}.sequences.parquet",
        version=SEQUENCE_VERSION, rule=rule,
        builder=lambda frame: build_sequences(frame, rule=rule),
        max_chunk_rows=max_chunk_rows, rebuild=rebuild, what="sequences",
        cluster_by=None, count_column=None,
        extra_meta={"sequence_version": SEQUENCE_VERSION, "seq_window": SEQ_WINDOW,
                    "n_inputs": N_INPUTS, "state_classes": list(STATE_CLASSES)})


class SequenceSource:
    """
    The cached sequence table, held as arrays and sliced into windows.

    The table is ~9M narrow rows, so it is read once per run and kept as raw
    numpy arrays: users are contiguous after the sort, so `offset[u]` is the
    global index where user `u`'s block starts and a target's window is a
    contiguous run of 32 indices ending at `offset[u] + idx_end`.
    """

    def __init__(self, vectors: np.ndarray, offsets: np.ndarray,
                 user_codes: np.ndarray, pos: np.ndarray,
                 user_order: list[str]):
        self.vectors = vectors          # (n_jobs, N_INPUTS) float32
        self.offsets = offsets          # (n_users,) int64 — first row of each user
        self.user_codes = user_codes    # (n_jobs,) int64 — each row's user code
        self.pos = pos                  # (n_jobs,) int64 — position within the user
        # Which user each code stands for, in code order. An arm reads targets
        # out of the *feature* cache, which carries `user_hash` but no code, so
        # this is what turns a target's user into an offset index without a
        # per-job lookup table (9M entries) or a second sort at run time.
        self.user_order = user_order
        self._code_of = {u: i for i, u in enumerate(user_order)}

    @property
    def n_jobs(self) -> int:
        return self.vectors.shape[0]

    @property
    def n_users(self) -> int:
        return self.offsets.shape[0]

    @classmethod
    def load(cls, path: str | Path) -> "SequenceSource":
        """
        Read the cache and derive the per-user offsets.

        The offset is derived, then *checked*: every row of a user must sit at
        `offset[user] + pos[row]`. If the table were ever written unsorted, the
        derivation would still return something and the slices would silently cut
        the wrong 32 jobs — so the invariant is verified here, once, at load.
        """
        frame = pl.read_parquet(path, columns=[*_ORDER_COLUMNS, "pos",
                                               "state_class", *NUMERIC_FIELDS])
        if frame.height == 0:
            raise ValueError(f"{path}: the sequence table is empty")

        # `rle_id` numbers the runs of an already-sorted column 0..k-1 in order,
        # which is exactly "which user, left to right" for a table sorted by
        # `user_hash`. Cheaper than a dense rank and no dependency on how Polars
        # happens to number a `Categorical`.
        user_codes = frame["user_hash"].rle_id().to_numpy().astype(np.int64)
        pos = frame["pos"].to_numpy().astype(np.int64)
        row_index = np.arange(frame.height, dtype=np.int64)

        # Two invariants, both of which the slice depends on, checked once here
        # because both fail *silently*: a window would still be cut, just from
        # the wrong rows.
        n_blocks = int(user_codes[-1]) + 1
        n_distinct = frame["user_hash"].n_unique()
        if n_blocks != n_distinct:
            raise ValueError(
                f"{path}: {n_distinct} distinct users but {n_blocks} contiguous "
                "runs, so at least one user's rows are not together. The table "
                "must be sorted by (user_hash, end_time, job_id).")

        candidate = row_index - pos
        first = np.ones(frame.height, dtype=bool)
        first[1:] = user_codes[1:] != user_codes[:-1]
        offsets = np.full(n_blocks, -1, dtype=np.int64)
        offsets[user_codes[first]] = candidate[first]
        if not np.array_equal(candidate, offsets[user_codes]):
            bad = int((candidate != offsets[user_codes]).sum())
            raise ValueError(
                f"{path}: {bad} row(s) do not sit at `offset + pos` within their "
                "user, so `pos` is not the user's own 0..n-1 ordering and a "
                "32-job slice would not be the user's real history.")

        # `STATE_CLASSES` order, not alphabetical: the one-hot columns are part
        # of the per-job vector every B3 run is trained on, so their order is
        # part of the data contract rather than an accident of the strings.
        state_code = (frame["state_class"]
                      .replace_strict(list(STATE_CLASSES),
                                      list(range(len(STATE_CLASSES))),
                                      return_dtype=pl.Int64).to_numpy())

        numeric = np.column_stack([
            frame[f].to_numpy().astype(np.float32)
            for f in NUMERIC_FIELDS]).astype(np.float32)
        onehot = np.zeros((frame.height, len(STATE_CLASSES)), dtype=np.float32)
        onehot[np.arange(frame.height), state_code] = 1.0

        # First appearance order, which for a grouped column *is* the code order
        # `rle_id` assigned — so `user_order[k]` is the user `user_codes == k`
        # refers to. Re-checked against `n_blocks` rather than trusted: if the
        # two ever disagreed, `codes_for` would map a target to another user's
        # block and the window would be silently cut from the wrong history.
        user_order = frame["user_hash"].unique(maintain_order=True).to_list()
        if len(user_order) != n_blocks:
            raise ValueError(
                f"{path}: {len(user_order)} distinct users but {n_blocks} runs; "
                "the code order and the run order disagree.")
        return cls(np.ascontiguousarray(np.hstack([numeric, onehot])),
                   offsets, user_codes, pos, user_order)

    def codes_for(self, user_hashes) -> np.ndarray:
        """
        The offset index for each of `user_hashes`, in the order given.

        Strict on purpose: a user absent from the source raises rather than
        resolving to a default. A default here would be a silent wrong-window
        error — the arm would cut a valid-looking 32-job slice out of a
        *different* user's history — which is the failure the load-time
        invariants exist to make impossible rather than the one to add here.
        """
        return (pl.Series("user_hash", list(user_hashes))
                .replace_strict(self._code_of, return_dtype=pl.Int64)
                .to_numpy())

    def windows(self, user_codes: np.ndarray, idx_end: np.ndarray,
                width: int = SEQ_WINDOW):
        """
        The `width`-job windows ending at each target's last legal job.

        Returns `(x, mask)`: `x` is `(n_targets, width, N_INPUTS)` float32 and
        `mask` is `(n_targets, width)` bool, true where a real job sits. Padding
        goes at the **front** (B3-D4), so column `width-1` is always the most
        recent job and the last step the GRU sees is never padding.

        A target with `idx_end < 0` has no legal history and gets an all-padding
        window with a cleared mask. Such a row is unscoreable and the arm writes
        a blank for it, so this is a total function rather than a special case.
        """
        user_codes = np.asarray(user_codes, dtype=np.int64)
        idx_end = np.asarray(idx_end, dtype=np.int64)
        if user_codes.shape != idx_end.shape:
            raise ValueError("user_codes and idx_end must be the same length")

        # Where the window ends, how far back it reaches into the user's block,
        # and the first index of that block (the pad boundary).
        end = self.offsets[user_codes] + idx_end                      # (n,)
        start = self.offsets[user_codes]                              # (n,)
        back = np.arange(width - 1, -1, -1, dtype=np.int64)
        gather = end[:, None] - back                                  # oldest first
        mask = (idx_end[:, None] >= 0) & (gather >= start[:, None])

        # Clipped so the fancy index stays in range: out-of-window slots read the
        # block's first row and are then zeroed by the mask. `np.maximum` also
        # covers the no-history case, where `end` is `start - 1`.
        clipped = np.maximum(gather, start[:, None])
        x = self.vectors[clipped]                                     # (n, width, k)
        return x * mask[..., None], mask
