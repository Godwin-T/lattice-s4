"""
Shared machinery for the arms.

**Windowing.** Every arm that predicts a job's outcome looks back over that
user's history. What differs between arms is which features they take from it;
what must not differ is *which history is legal*. So the history rule lives here,
once, and every arm reads the same window table.

Two rules are implemented, and a run picks one (`--window`):

* `safe` — **the default, and the rule every arm is scored under.** The window is
  the `WINDOW` jobs that had most recently **ended before the target was
  submitted**. This is `folds_manifest.md`'s
  `history_jobs_must_end_before_target_submit` and `plan.md` §14.4. It is
  deliberately conservative: if one of the previous 24 jobs was still running,
  the target is marked unscorable rather than reaching further back. It can only
  lose rows, never leak. How many it loses is reported.

  "**before**" is strict, and strictness is load-bearing. The match is on
  full-resolution timestamps and excludes a job that ends in the same
  microsecond as the target's submit — in particular the target itself, since a
  job that never ran has `end_time == submit_time` (28,638 such rows on
  Kestrel). An inclusive match on second-truncated timestamps would let those
  rows take their features from their own final ratio, which is the label
  leaking back in through the one door this rule exists to shut.

* `published` — the window is the previous `WINDOW` jobs in **submit order**,
  with no requirement that they had ended. This is what the published
  replication actually does (`21913139/kestrel_replicate.py:47,65-66`), and it
  exists here solely so the benchmark can report the published number and its own
  side by side under one evaluator. It is **not** submit-time safe: measured on
  Kestrel, 97.9% of its windows include a job that had not finished when the
  target was submitted, and the feature then uses that job's final
  `elapsed / timelimit` — a value that does not exist at decision time. The
  write-up's claim that *"the window closes before the predicted job is
  submitted"* (`21913139/WRITEUP.md:132-133`) does not hold as coded.

The two rules differ in exactly one step — how the rolling statistics are
attached to a target — so a difference between their scores is attributable to
the window and nothing else. The features, the label, the month and the row ids
are shared.

**The hand-in table.** `normalise_predictions` enforces the contract: one row
per test row per fold, keyed by `row_id` (not `job_id`, which is not unique),
`score`/`probability` null for rows an arm cannot score, and never a row for a
canonical row that is not in that fold's test set. `write_predictions` applies it
to a whole table at once; an arm that streams its output applies it to each shard
instead, so the merged table is never resident.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import polars as pl

WINDOW = 24
FEATURES = ("f_mean", "f_std", "f_range", "f_madiff")

# The two history-window rules (see the module docstring). `safe` is what every
# arm is scored under; `published` exists only for the side-by-side reproduction.
WINDOW_RULES = ("safe", "published")
RULE_SAFE, RULE_PUBLISHED = "safe", "published"

# How many *rows* of windows to build before spilling to disk. Windowing is a
# per-user operation, so chunking cannot change the result; it only bounds
# memory. The cap is on rows rather than users because user sizes are wildly
# unequal -- one user can hold hundreds of thousands of jobs, so "N users" says
# almost nothing about how much memory a chunk will need.
WINDOW_MAX_CHUNK_ROWS = 250_000

# The on-disk window cache is clustered by month (see `ensure_windows`), so a
# per-fold `month` filter can skip row groups instead of scanning every row.
# The cache is keyed on the canonical table's hash, this version, and the window
# rule; bump CACHE_VERSION whenever the cache's shape *or the meaning of its
# columns* changes, so a stale cache is rebuilt rather than silently mis-read.
# v5: both rules now order and match on full-resolution timestamps. v4 and
# earlier matched the safe rule on second-truncated keys, which let a target's
# window include a job that had not ended -- and, for a zero-elapsed job, the
# target itself.
CACHE_VERSION = 5
WINDOW_ROW_GROUP_ROWS = 262_144

PREDICTION_SCHEMA = {
    "run_id": pl.String,
    "arm": pl.String,
    "task": pl.String,
    "fold_id": pl.Int64,
    "test_month": pl.String,
    "row_id": pl.Int64,
    "job_id": pl.String,
    "score": pl.Float64,
    "probability": pl.Float64,
    "latency_ms": pl.Float64,
}


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def with_row_id(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Attach a stable, row-unique `row_id` to the canonical table.

    `job_id` is *not* a unique key. `canonical_table.md` §2 asks only that it be
    "stable within the dataset", and §5 lists a duplicate `job_id` as a flagged,
    deliberately-unfixed edge case: a requeued job keeps its id across attempts.
    Kestrel is the worked example — 9,320,707 rows under 8,562,264 ids, one id
    covering 10,000 attempts with a single submit time and 10,000 different end
    times. Anything that must identify a *row* — attaching a label, checking a
    fold was scored exactly once — therefore needs a key of its own; joining on
    `job_id` multiplies rows by the square of the duplicate count.

    The id is the row's position in the canonical file. Parquet preserves row
    order, so it is dense, deterministic and cheap, and it is assigned *before*
    any filter, so it does not depend on which population a caller later selects.
    The arm and the evaluator derive it independently from the same file and must
    agree; that was verified stable across 1, 2 and 8 threads.

    Cast to `Int64` to match `PREDICTION_SCHEMA`: `with_row_index` defaults to
    `UInt32`, and a join key whose type depends on which side built it would rely
    on an implicit upcast.
    """
    return lf.with_row_index("row_id", offset=0).with_columns(
        pl.col("row_id").cast(pl.Int64))


# Every rule emits the same columns; only `scoreable` and the features differ.
_OUTPUT_COLUMNS = ["job_id", "row_id", "user_hash", "month", "scoreable", "label",
                   "energy_j", *FEATURES]


def _rolling(ordered: pl.LazyFrame, order_cols: list[str]) -> pl.LazyFrame:
    """
    The four published features as a rolling window over one ordering.

    Shared by both rules, so the features cannot drift between them — the only
    thing a rule changes is *which* target inherits which window's statistics.
    `idx` is the row's position within its user under this ordering, which is
    what each rule's `scoreable` guard reads.
    """
    return ordered.sort(["user_hash", *order_cols]).with_columns([
        pl.int_range(pl.len()).over("user_hash").cast(pl.Int32).alias("idx"),
        pl.col("ratio").rolling_mean(WINDOW).over("user_hash").alias("f_mean"),
        # ddof=0 to match the published method (numpy's default); Polars uses 1.
        pl.col("ratio").rolling_std(WINDOW, ddof=0).over("user_hash").alias("f_std"),
        (pl.col("ratio").rolling_max(WINDOW).over("user_hash")
         - pl.col("ratio").rolling_min(WINDOW).over("user_hash")).alias("f_range"),
        pl.col("ratio").diff().abs()
          .rolling_mean(WINDOW - 1).over("user_hash").alias("f_madiff"),
    ])


def build_windows(lf: pl.LazyFrame, rule: str = RULE_SAFE) -> pl.LazyFrame:
    """
    Turn the canonical table into one row per (target job -> its 24-job history).

    The four features are the published Lattice24 set: mean, standard deviation,
    range, and mean absolute successive difference of `elapsed / timelimit` over
    the window. Nothing about the target itself enters its own features.

    **Which 24 jobs?** Two answers, chosen by `rule` (the module docstring says
    why both exist):

    * `RULE_SAFE` — the 24 that had most recently **ended before the target was
      submitted**. Jobs are ordered by end time, the rolling statistics are the
      window ending at each job, and an as-of join maps each target to the last
      job that had ended before its submit time; the target inherits that row's
      statistics. Ordering by end time is safe: anything that ended before the
      target was submitted was necessarily submitted before it too. The naive
      reading — "the previous 24 jobs by submit order, and require that they all
      ended first" — is far too strict for real clusters, where array jobs submit
      thousands of jobs in the same second and none has ended yet: it discards
      almost the whole dataset (measured: ~1.5% of rows survive on Eagle),
      whereas this keeps the history a submit-time decision could really use.

    * `RULE_PUBLISHED` — the previous 24 in **submit order**, whether or not they
      had ended. This is the published replication verbatim
      (`21913139/kestrel_replicate.py`: `sort_values('submit_time')`, then
      `idxs[k:k+WINDOW]`); row *i* takes the rolling value at *i-1*, i.e. the
      window `[i-24, i-1]`. It produced the published 0.954 and is reproduced
      here only to be reported beside the safe rule. It is not submit-time safe.

    The two rules share everything but that one step, so their outputs agree on
    `row_id`, `job_id`, `month`, `label` and `energy_j` by construction, and can
    differ only in `scoreable` and the features. A difference between their
    scores is therefore attributable to the window and nothing else.

    `row_id` (see `with_row_id`) is carried through to the output; it is the
    only column here that identifies a canonical *row*, since `job_id` does not.
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"unknown window rule {rule!r}; expected one of {WINDOW_RULES}")

    ordered = (
        lf.select(["job_id", "row_id", "user_hash", "submit_time", "end_time",
                   "elapsed_s", "timelimit_s", "state", "energy_j"])
        .with_columns([
            (pl.col("elapsed_s") / pl.col("timelimit_s")).alias("ratio"),
            pl.col("submit_time").dt.strftime("%Y-%m").alias("month"),
        ])
    )

    if rule == RULE_PUBLISHED:
        # Row `i` is the target; shifting the submit-ordered rolling value by one
        # row is exactly the published window [i-24, i-1]. No ended-before-submit
        # requirement -- that is the whole difference from the safe rule.
        #
        # Ordered on full-resolution `submit_time`, as the published script's
        # `sort_values('submit_time')` is. Ordering on a second-truncated key
        # instead would break ties by `job_id` and reorder the window for array
        # jobs that submit thousands of jobs in one second -- a second, silent
        # difference from the safe rule, on top of the one this branch exists to
        # isolate.
        return (
            _rolling(ordered, ["submit_time", "job_id"])
            .rename({"idx": "idx_submit"})
            .with_columns([pl.col(f).shift(1).over("user_hash").alias(f)
                           for f in FEATURES])
            .with_columns([
                (pl.col("idx_submit") >= WINDOW).alias("scoreable"),
                (pl.col("state") == "TIMEOUT").alias("label"),
            ])
            .select(_OUTPUT_COLUMNS)
        )

    # Each target takes the statistics of the last job that ended **strictly
    # before** it was submitted. A target whose match has fewer than 24 jobs
    # behind it cannot be scored.
    #
    # The join is on full-resolution `end_time`/`submit_time`, and the left key
    # is pulled back by one microsecond, because `strategy="backward"` is `<=`
    # and the rule is `<`. Joining on the second-truncated `end_s`/`submit_s`
    # instead is wrong twice over: it treats a job that ended later in the same
    # second as already finished, and — because a job that never ran has
    # `end_time == submit_time` (28,638 such rows on Kestrel) — it can match a
    # target to *itself* and build the target's features from the target's own
    # final ratio. That is exactly the leak this rule exists to prevent.
    # `end <= submit - 1us` is `end < submit` on the microsecond grid the
    # canonical table is stored on.
    by_end = _rolling(ordered, ["end_time", "job_id"]).rename({"idx": "idx_end"})
    targets = ordered.sort(["submit_time", "job_id"]).with_columns(
        (pl.col("submit_time") - pl.duration(microseconds=1)).alias("submit_prev")
    ).select([
        "job_id", "row_id", "user_hash", "submit_prev", "month", "state", "energy_j",
    ])
    joined = targets.join_asof(
        by_end.select(["user_hash", "end_time", "idx_end", *FEATURES]),
        left_on="submit_prev", right_on="end_time", by="user_hash",
        strategy="backward",
    )

    return joined.with_columns([
        ((pl.col("idx_end").is_not_null()) & (pl.col("idx_end") >= WINDOW - 1))
        .alias("scoreable"),
        (pl.col("state") == "TIMEOUT").alias("label"),
    ]).select(_OUTPUT_COLUMNS)


def normalise_predictions(rows: pl.DataFrame) -> pl.DataFrame:
    """
    Coerce a hand-in frame to the contract schema.

    Columns the arm did not produce become explicit nulls; working columns it
    did produce (such as `month`) are dropped. An arm applies this to each
    streamed shard, so the shards share one schema and can be merged without
    ever holding the whole table.
    """
    exprs = [
        (pl.col(name).cast(dtype, strict=False) if name in rows.columns
         else pl.lit(None, dtype=dtype)).alias(name)
        for name, dtype in PREDICTION_SCHEMA.items()
    ]
    return rows.select(exprs)


def write_predictions(rows: pl.DataFrame, outdir: str | Path) -> Path:
    """Write the hand-in table in the contract shape."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "predictions.parquet"
    normalise_predictions(rows).write_parquet(path)
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


def _ensure_cached(lf: pl.LazyFrame, canonical_path: str | Path,
                   cache_dir: str | Path, *, filename: str, version: int,
                   rule: str, builder, max_chunk_rows: int, rebuild: bool,
                   what: str, extra_meta: dict | None = None,
                   cluster_by: str | None = "month",
                   count_column: str | None = "scoreable") -> Path:
    """
    Build a per-target table from the canonical rows once, and keep it as Parquet.

    Shared by the window cache (Arm A) and the feature cache (Arm B) so the two
    cannot drift: the chunking, the month-clustering and the staleness key are
    one implementation. `builder` turns a chunk's `LazyFrame` into that chunk's
    output rows; a chunk holds whole users, so the result cannot depend on how
    the users were packed.

    `lf` is the already-filtered frame to build from (the arms pre-filter to the
    manifest's eligible population). This is the memory-heavy step — one row per
    job, eleven million of them on Eagle — so writing it out means a run can then
    read a fold at a time instead of holding everything in RAM. The cache is
    keyed to the canonical table's hash, `version` and the window `rule`, so a
    rebuilt table, a change in the cache's own shape, or a switch of window rule
    each invalidates it automatically.

    The key deliberately omits the eligible population and `WINDOW`: correct
    today only because every arm builds from the same manifest's population, and
    `WINDOW` is a module constant. If either becomes per-run, it belongs in the
    key too — the failure mode is a silently wrong cache, not a crash.

    `cluster_by` and `count_column` are the two things the window and feature
    tables happen to share rather than properties of the machinery.
    `count_column` is `"scoreable"` for both and is written into the metadata;
    a per-*job* table has no notion of scoreability, so B3 passes `None` and the
    field is simply absent. `cluster_by` is subtler and B3 must pass `None`:
    clustering reorders rows by month so a per-fold month filter can skip row
    groups, and that is free for a table with one row per *target* — but B3's
    table is one row per *job* and its loader requires each user's jobs to be
    contiguous, so partitioning by month would interleave users and silently
    invalidate every window. Defaults reproduce the existing two caches exactly.
    """
    canonical_path = Path(canonical_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / filename
    meta = cache.with_suffix(".meta.json")

    fingerprint = file_sha256(canonical_path)
    if not rebuild and cache.exists() and meta.exists():
        try:
            stored = json.loads(meta.read_text())
            if (stored.get("version") == version
                    and stored.get("rule") == rule
                    and stored["canonical_sha256"] == fingerprint):
                return cache
        except (OSError, ValueError, KeyError):
            pass

    # Build in row-bounded chunks. Every arm's per-target build is a per-user
    # operation — neither the rolling statistics nor the as-of join crosses a
    # user boundary — so the result is identical, but peak memory is one chunk
    # instead of the whole table. The bound is on rows because users differ
    # enormously in size.
    counts = lf.group_by("user_hash").len().collect()
    if counts.height == 0:
        raise SystemExit(f"no users to build {what} for")
    chunks = _pack_users(counts, max_chunk_rows)

    scratch = Path(tempfile.mkdtemp(prefix=f"{what}_"))
    shards: list[Path] = []
    rows = scoreable = 0
    try:
        for i, (batch, expected) in enumerate(chunks):
            frame = builder(lf.filter(pl.col("user_hash").is_in(batch))).collect()
            shard = scratch / f"part-{i:05d}.parquet"
            frame.write_parquet(shard)
            shards.append(shard)
            rows += frame.height
            if count_column is not None:
                scoreable += int(frame[count_column].sum())
            print(f"    chunk {i + 1}: {frame.height:,} rows "
                  f"({len(batch)} user(s), ~{expected:,} expected) "
                  f"| running total {rows:,} | peak RSS {_peak_rss_mb():,.0f} MB",
                  flush=True)
        # Merge the shards into the single cached file, clustered by month. The
        # shards are packed by user, so months are interleaved and a per-fold
        # `month` filter cannot skip any row group — it scans the whole table.
        # Clustered, each row group spans about one month and a fold reads only
        # the months it needs. Rows are unchanged; only their order is.
        #
        # Done by partitioning into one file per month and concatenating those
        # in month order, rather than by sorting the whole table in memory:
        # `sort("month").collect()` and even `.sink_parquet()` after a sort both
        # materialise all nine million rows (measured: ~3.0 GB peak), which is
        # exactly the cost the cache exists to avoid. Partitioning keeps the
        # peak at the chunk-build level (~1.0 GB) for the same output.
        src = [str(p) for p in shards]
        if cluster_by is None:
            # Concatenate the shards as they are. Whole users were packed per
            # chunk, so every user's rows stay contiguous in this order — which
            # is the property the caller passed `cluster_by=None` to keep.
            pl.scan_parquet(src).sink_parquet(
                cache, row_group_size=WINDOW_ROW_GROUP_ROWS)
        else:
            months = sorted(
                pl.scan_parquet(src).select(cluster_by).unique()
                .collect()[cluster_by].to_list()
            )
            parts = []
            for month in months:
                part = scratch / f"month-{month}.parquet"
                (pl.scan_parquet(src).filter(pl.col(cluster_by) == month)
                 .sink_parquet(part))
                parts.append(str(part))
            pl.scan_parquet(parts).sink_parquet(
                cache, row_group_size=WINDOW_ROW_GROUP_ROWS)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    payload = {
        "version": version,
        "rule": rule,
        "canonical": str(canonical_path),
        "canonical_sha256": fingerprint,
        "rows": rows,
        "max_chunk_rows": max_chunk_rows,
    }
    if count_column is not None:
        payload["scoreable"] = scoreable
    payload.update(extra_meta or {})
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return cache


def ensure_windows(lf: pl.LazyFrame, canonical_path: str | Path,
                   cache_dir: str | Path, rebuild: bool = False,
                   max_chunk_rows: int = WINDOW_MAX_CHUNK_ROWS,
                   rule: str = RULE_SAFE) -> Path:
    """
    Build the window table once and keep it as Parquet.

    See `_ensure_cached` for the chunking and staleness discipline, which this
    shares with Arm B's feature cache. The four published features are all this
    table carries; Arm B reads `features.ensure_features` instead.
    """
    if rule not in WINDOW_RULES:
        raise ValueError(f"unknown window rule {rule!r}; expected one of {WINDOW_RULES}")
    return _ensure_cached(
        lf, canonical_path, cache_dir,
        filename=f"{Path(canonical_path).stem}.{rule}.windows.parquet",
        version=CACHE_VERSION, rule=rule,
        builder=lambda frame: build_windows(frame, rule=rule),
        max_chunk_rows=max_chunk_rows, rebuild=rebuild, what="windows")


def _pack_users(counts: pl.DataFrame, max_rows: int):
    """
    Group users into batches whose total job count stays near `max_rows`.

    Biggest users first, so a giant lands in a batch of its own rather than
    dragging a full complement of ordinary users along with it.
    """
    ordered = counts.sort("len", descending=True)
    batch: list[str] = []
    size = 0
    for user, n in ordered.iter_rows():
        n = int(n)
        if batch and size + n > max_rows:
            yield batch, size
            batch, size = [], 0
        batch.append(user)
        size += n
    if batch:
        yield batch, size


def _peak_rss_mb() -> float:
    """Peak resident memory so far, in MB. Diagnostic only; 0.0 if unavailable."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                                  # noqa: BLE001 - diagnostic
        return 0.0
