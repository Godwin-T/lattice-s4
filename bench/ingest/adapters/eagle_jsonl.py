"""
Adapter: Eagle 3-month JSONL (NLR submission 152) -> canonical table.

Source: `21913139/anon_jobs_<YYYY-MM>.json` — JSON Lines, one job per line.
Only three non-consecutive months exist (2019-12, 2020-04, 2020-08), so this
dataset is used for reproduction, not for fold-based evaluation.

Notable properties (see canonical_table.md §4.2):
* `user` / `account` / `script` are ALREADY hashed -> pass through, no re-hash.
* durations are ISO-8601 strings -> kind="iso8601".
* energy is not measured: `avg_power x nodes_used x elapsed_s` -> "modelled".
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from .. import energy
from ..durations import duration_expr
from ..schema import conform
from ..states import state_expr

CLUSTER = "eagle"


def _ts(col: str) -> pl.Expr:
    """Parse '2019-12-04T16:17:00.000Z' -> Datetime[us, UTC]."""
    return (
        pl.col(col)
        .cast(pl.String, strict=False)
        .str.strip_suffix("Z")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%S%.f", time_zone="UTC", strict=False)
    )


def load(paths: list[str], salt: str | None = None) -> pl.DataFrame:
    """
    Read one or more monthly JSONL files into the canonical shape.

    `salt` is accepted for interface uniformity and ignored: this source already
    ships hashed identifiers, so there is nothing to hash (§3.1).
    """
    frames = []
    for p in paths:
        frame = pl.read_ndjson(p, infer_schema_length=None)
        frames.append(frame.with_columns(pl.lit(str(p)).alias("source_file")))
    df = frames[0] if len(frames) == 1 else pl.concat(frames, how="vertical_relaxed")

    # Durations first: the modelled-energy formula depends on elapsed_s.
    elapsed = duration_expr("wallclock_used", "iso8601")

    df = df.with_columns([
        pl.col("job_id").cast(pl.String, strict=False).alias("job_id"),
        pl.col("user").cast(pl.String).alias("user_hash"),        # already hashed
        pl.col("account").cast(pl.String).alias("account_hash"),  # already hashed
        pl.col("script").cast(pl.String).alias("script_hash"),    # already hashed
        pl.col("partition").cast(pl.String),
        pl.col("qos").cast(pl.String),
        state_expr("state").alias("state"),
        _ts("submit_time").alias("submit_time"),
        _ts("start_time").alias("start_time"),
        _ts("end_time").alias("end_time"),
        duration_expr("wallclock_req", "iso8601").alias("timelimit_s"),
        elapsed.alias("elapsed_s"),
        duration_expr("queue_wait", "iso8601").alias("queue_wait_s"),
        pl.col("processors_req").cast(pl.Int64, strict=False).alias("cpus_req"),
        pl.col("processors_used").cast(pl.Int64, strict=False).alias("cpus_used"),
        pl.col("nodes_req").cast(pl.Int64, strict=False).alias("nodes_req"),
        pl.col("nodes_used").cast(pl.Int64, strict=False).alias("nodes_used"),
        pl.col("array_pos").cast(pl.Int64, strict=False).alias("array_pos"),
    ])

    # Energy: modelled from avg_power x nodes_used x elapsed_s.
    joules = energy.modelled_j("avg_power", "nodes_used", "elapsed_s")
    df = df.with_columns([
        joules.alias("energy_j"),
        energy.tier_from(joules, "modelled").alias("energy_tier"),
    ])

    # `cluster` is set by conform(); provenance columns already present except
    # cluster/ingest_version, which conform fills in.
    return conform(df, CLUSTER)


def file_month(path: str) -> str:
    """'anon_jobs_2019-12.json' -> '2019-12' (handy for audit output)."""
    stem = Path(path).stem            # anon_jobs_2019-12
    return stem.split("_")[-1]
