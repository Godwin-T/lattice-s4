"""
The canonical job table schema (canonical_table.md §2) and shared constants.

Every adapter produces exactly these columns, in this order, with these types.
Columns a source lacks are added as nulls rather than omitted, so downstream
code can rely on the shape.
"""
from __future__ import annotations

import polars as pl

INGEST_VERSION = "1"

# Terminal states are kept; anything else has no outcome to learn from and is
# dropped. DEADLINE was added after scanning Kestrel (canonical_table.md §4.3.2).
TERMINAL_STATES: tuple[str, ...] = (
    "COMPLETED", "TIMEOUT", "FAILED", "CANCELLED",
    "OUT_OF_MEMORY", "NODE_FAIL", "DEADLINE",
)

ENERGY_TIERS: tuple[str, ...] = ("measured", "modelled", "estimated", "none")

# Fields that must never reach the canonical table (canonical_table.md §3.7).
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({
    "name", "jobname", "work_dir", "workdir", "submit_line", "submitline",
    "command", "comment", "user", "account", "wckey", "reservation",
})

# Authoritative column order and types.
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "job_id": pl.String(),
    "cluster": pl.String(),
    "user_hash": pl.String(),
    "account_hash": pl.String(),
    "partition": pl.String(),
    "qos": pl.String(),
    "state": pl.String(),
    "script_hash": pl.String(),
    "submit_time": pl.Datetime("us", "UTC"),
    "start_time": pl.Datetime("us", "UTC"),
    "end_time": pl.Datetime("us", "UTC"),
    "timelimit_s": pl.Float64(),
    "elapsed_s": pl.Float64(),
    "queue_wait_s": pl.Float64(),
    "cpus_req": pl.Int64(),
    "cpus_used": pl.Int64(),
    "mem_req": pl.Float64(),
    "mem_used": pl.Float64(),
    "gpus_req": pl.Int64(),
    "nodes_req": pl.Int64(),
    "nodes_used": pl.Int64(),
    "array_pos": pl.Int64(),
    "dependency": pl.String(),
    "energy_j": pl.Float64(),
    "energy_tier": pl.String(),
    "source_dataset": pl.String(),
    "source_file": pl.String(),
    "ingest_version": pl.String(),
}

CANONICAL_COLUMNS: list[str] = list(CANONICAL_SCHEMA)


def conform(df: pl.DataFrame, cluster: str) -> pl.DataFrame:
    """
    Force a frame into the canonical shape: add missing columns as nulls, cast
    the rest, set the provenance columns, and return columns in canonical order.
    """
    exprs = []
    for name, dtype in CANONICAL_SCHEMA.items():
        if name in df.columns:
            exprs.append(pl.col(name).cast(dtype, strict=False).alias(name))
        elif name == "cluster":
            exprs.append(pl.lit(cluster).cast(dtype).alias(name))
        elif name == "ingest_version":
            exprs.append(pl.lit(INGEST_VERSION).cast(dtype).alias(name))
        else:
            exprs.append(pl.lit(None).cast(dtype).alias(name))
    return df.select(exprs)
