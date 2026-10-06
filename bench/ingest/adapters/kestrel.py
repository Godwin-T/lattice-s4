"""
Adapter: Kestrel (NLR submission 302) -> canonical table.

Source: `21913139/kestrel/kestrel_jobs_<YYYYMM>_0.parquet`, 29 monthly files,
2023-08 … 2025-12, 10,559,977 rows. See canonical_table.md §4.3.

Things this adapter has to get right, all verified against the real archive:

* **Timestamps are not uniformly UTC.** 24 files are UTC; 3 are -07:00 and 2 are
  -06:00. Every timestamp column is converted to UTC per file *before* anything
  is combined (a naive concat crashes on the dtype mismatch).
* **Durations are Arrow `duration[ns]`**, not strings or seconds.
* **Identifiers are already hashed** (`user_hash`, `account_hash`,
  `submit_script_hash`) -> passed through, never re-hashed.
* **`state_simple` is pre-normalised**; the raw `state` carries 'CANCELLED by N'.
* **Energy is measured** (`consumed_energy_raw_joules`). The decoys --
  `consumed_energy_joules` (string), `consumed_energy_raw_watt_hours`, and
  `cpu_energy_tdp_estimated_*` -- are deliberately never read.
* **`memory_req` is a string with a unit suffix** ('500000G', '250000M'), but the
  values look like partition defaults rather than real requests (see the doc);
  they are parsed to MB for completeness and flagged as unreliable.
* **Seven columns are entirely null** (`array_pos`, `array_range`,
  `gpus_requested`, `gpu_nodes_occupied`, the three `*_mem_eff`) -> left null;
  they must not be imputed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import energy
from ..durations import duration_expr
from ..schema import conform
from ..states import state_expr

CLUSTER = "kestrel"

# Only these source columns are ever read. Everything else -- including the
# energy decoys and every hash of a sensitive field we do not need -- is left
# on disk, unread.
SRC_COLUMNS = [
    "job_id", "user_hash", "account_hash", "submit_script_hash",
    "partition", "qos", "state_simple",
    "submit_time", "start_time", "end_time",
    "wallclock_req", "wallclock_used", "queue_wait",
    "processors_req", "processors_used", "nodes_req", "nodes_used",
    "memory_req", "consumed_energy_raw_joules",
]

# memory_req suffix -> multiplier into MB (the canonical unit for this source).
_MB = {"K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024}


def _to_utc(col: str) -> pl.Expr:
    """Convert any tz-aware timestamp column to UTC, us precision."""
    return (
        pl.col(col)
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime("us", "UTC"))
    )


def _mem_mb(col: str) -> pl.Expr:
    """'256G' -> 262144.0 (MB); '250000M' -> 250000.0. Null if unparseable."""
    value = pl.col(col).cast(pl.String).str.extract(r"^\s*([0-9.]+)", 1).cast(pl.Float64, strict=False)
    unit = pl.col(col).cast(pl.String).str.extract(r"([A-Za-z])\s*$", 1).str.to_uppercase()
    factor = (
        pl.when(unit == "K").then(_MB["K"])
        .when(unit == "M").then(_MB["M"])
        .when(unit == "G").then(_MB["G"])
        .when(unit == "T").then(_MB["T"])
        .otherwise(None)
    )
    return value * factor


def map_file(path: str) -> pl.DataFrame:
    """Read one monthly file and return it in canonical shape (unfiltered)."""
    df = pl.read_parquet(path, columns=SRC_COLUMNS)

    joules = energy.measured_j("consumed_energy_raw_joules")
    df = df.select([
        pl.col("job_id").cast(pl.String, strict=False).alias("job_id"),
        pl.col("user_hash").alias("user_hash"),              # already hashed
        pl.col("account_hash").alias("account_hash"),        # already hashed
        pl.col("submit_script_hash").alias("script_hash"),   # already hashed
        pl.col("partition").cast(pl.String),
        pl.col("qos").cast(pl.String),
        state_expr("state_simple").alias("state"),
        _to_utc("submit_time").alias("submit_time"),
        _to_utc("start_time").alias("start_time"),
        _to_utc("end_time").alias("end_time"),
        duration_expr("wallclock_req", "arrow_duration_ns").alias("timelimit_s"),
        duration_expr("wallclock_used", "arrow_duration_ns").alias("elapsed_s"),
        duration_expr("queue_wait", "arrow_duration_ns").alias("queue_wait_s"),
        pl.col("processors_req").cast(pl.Int64, strict=False).alias("cpus_req"),
        pl.col("processors_used").cast(pl.Int64, strict=False).alias("cpus_used"),
        pl.col("nodes_req").cast(pl.Int64, strict=False).alias("nodes_req"),
        pl.col("nodes_used").cast(pl.Int64, strict=False).alias("nodes_used"),
        _mem_mb("memory_req").alias("mem_req"),
        joules.alias("energy_j"),
        energy.tier_from(joules, "measured").alias("energy_tier"),
        pl.lit(str(path)).alias("source_file"),
    ])
    return conform(df, CLUSTER)


def load(paths: list[str], salt: str | None = None) -> pl.DataFrame:
    """
    Map every monthly file and stack them into one canonical frame.

    Each file is mapped and cast to the canonical dtypes *before* being
    combined, which is what makes the mixed-timezone files compatible.

    `salt` is accepted for interface uniformity and ignored: Kestrel already
    ships hashed identifiers, so there is nothing to hash (§3.1).
    """
    frames = [map_file(p) for p in sorted(paths)]
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")


def month_of(path: str) -> str:
    """'.../kestrel_jobs_202308_0.parquet' -> '2023-08'."""
    stem = Path(path).stem                     # kestrel_jobs_202308_0
    ym = stem.split("_")[-2]                   # 202308
    return f"{ym[:4]}-{ym[4:]}"
