"""
Adapter: Eagle 11M Parquet (OEDI 5860) -> canonical table.

Source: `data/eagle_data.parquet`, 11,030,377 rows, 52 consecutive months
(2018-11 … 2023-02), 936 users. See canonical_table.md §4.1.

Notes specific to this source:

* **No energy column at all.** Every row gets `energy_j = null` and
  `energy_tier = "none"`. On this dataset only the prediction metrics (AUC,
  recall, false-flag) are computable — never "energy captured".
* **Identifiers are hashed** with a fixed per-dataset salt, for uniformity with
  the other sources: the value in the table is `sha256(salt + id)[:16]`. The
  raw ids here are synthetic pseudonyms (`user0001`, `account0001`), so the hash
  adds no privacy over the source — but it guarantees the canonical table never
  carries a source identifier verbatim, and it makes all three datasets agree in
  shape. The salt is supplied at run time and is never written to disk.
* **Durations are plain seconds** (`wallclock_req`, `run_time`) — not strings.
* **Timestamps are naive** (no timezone); they are declared UTC.
* **Sensitive columns are never read**: `name`, `work_dir`, `submit_line` are
  excluded from the projection, so they are never even decoded.
* **11M rows will not fit comfortably in memory**, so this adapter streams: see
  `iter_canonical` and `build.py`.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from .. import energy
from ..durations import duration_expr
from ..hashing import hash_column
from ..schema import conform
from ..states import state_expr

CLUSTER = "eagle"
CHUNK_ROWS = 2_000_000
# This is the only current source that hashes: it must be given a salt.
REQUIRES_SALT = True

# Only these columns are ever decoded. Everything else -- including the three
# sensitive text fields -- is left on disk.
SRC_COLUMNS = [
    "job_id", "user", "account", "partition", "qos", "state",
    "submit_time", "start_time", "end_time",
    "wallclock_req", "run_time",
    "processors_req", "nodes_req", "gpus_req", "mem_req",
]


def _timestamp_exprs(df: pl.DataFrame, cols: list[str]) -> list[pl.Expr]:
    """
    Convert timestamp columns to UTC, handling both naive and tz-aware inputs.
    A naive value is *declared* UTC (no shift); an aware value is *converted*.
    """
    exprs = []
    for col in cols:
        dtype = df.schema[col]
        expr = pl.col(col)
        expr = (expr.dt.convert_time_zone("UTC")
                if getattr(dtype, "time_zone", None)
                else expr.dt.replace_time_zone("UTC"))
        exprs.append(expr.cast(pl.Datetime("us", "UTC")).alias(col))
    return exprs


def map_frame(df: pl.DataFrame, source_file: str, salt: str) -> pl.DataFrame:
    """Map one chunk (or a whole file) of Eagle 11M rows into canonical shape."""
    if not salt:
        raise ValueError(
            "eagle_parquet hashes identifiers and requires a salt: pass --salt "
            "or set BENCH_SALT_EAGLE (or BENCH_SALT)"
        )
    # Hashed before anything else, from the original columns; the raw ids are
    # dropped by conform() and never reach the output.
    user_hash = hash_column(df["user"], salt)
    account_hash = hash_column(df["account"], salt)

    df = df.with_columns([
        pl.col("job_id").cast(pl.String, strict=False).alias("job_id"),
        pl.col("partition").cast(pl.String),
        pl.col("qos").cast(pl.String),
        state_expr("state").alias("state"),
    ] + _timestamp_exprs(df, ["submit_time", "start_time", "end_time"]) + [
        duration_expr("wallclock_req", "seconds").alias("timelimit_s"),
        duration_expr("run_time", "seconds").alias("elapsed_s"),
        pl.col("processors_req").cast(pl.Int64, strict=False).alias("cpus_req"),
        pl.col("nodes_req").cast(pl.Int64, strict=False).alias("nodes_req"),
        pl.col("gpus_req").cast(pl.Int64, strict=False).alias("gpus_req"),
        pl.col("mem_req").cast(pl.Float64, strict=False).alias("mem_req"),
        energy.tier("none").alias("energy_tier"),   # this source has no energy
        pl.lit(str(source_file)).alias("source_file"),
    ])
    df = df.with_columns([
        user_hash.alias("user_hash"),
        account_hash.alias("account_hash"),
    ])
    return conform(df, CLUSTER)


def load(paths: list[str], salt: str | None = None) -> pl.DataFrame:
    """In-memory variant (fine for tests and small slices)."""
    frames = [map_frame(pl.read_parquet(p), str(p), salt) for p in sorted(paths)]
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")


def iter_canonical(paths: list[str], salt: str | None = None,
                   chunk_rows: int = CHUNK_ROWS):
    """
    Yield canonical chunks, one pyarrow batch at a time.

    This is what keeps an 11M-row ingest inside memory: only `chunk_rows` rows
    are ever materialised at once, and `build.py` spills each filtered chunk to
    disk before moving on.
    """
    for path in sorted(paths):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_rows, columns=SRC_COLUMNS):
            yield map_frame(pl.from_arrow(batch), str(path), salt)


def month_of(path: str) -> str:
    """Filename stem, handy for logging."""
    return Path(path).stem
