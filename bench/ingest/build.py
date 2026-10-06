"""
Streaming build of a canonical table from a chunked adapter.

`adapters.kestrel` and `adapters.eagle_jsonl` are small enough to map in one go.
The Eagle 11M table is not: mapping it whole peaks well above available RAM.
This module runs the same mapping in chunks, filters each chunk, spills clean
shards to disk, and streams them back into one output file -- so peak memory
stays proportional to the chunk, not the dataset.

The output is the same table the in-memory path would produce; only the memory
profile differs.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import polars as pl

from . import validate
from .filters import apply_filters


def build_from_chunks(chunks, outdir, dataset: str) -> tuple[dict, dict]:
    """
    Consume an iterable of canonical (unfiltered) DataFrames.

    Returns (stats, summary):
      stats   -- rows in/out and per-reason drop counts (they reconcile)
      summary -- state counts, energy-tier counts, coverage
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"canon_{dataset}_"))
    shards: list[Path] = []
    stats: dict[str, int] = {"rows_in": 0}

    try:
        for i, chunk in enumerate(chunks):
            stats["rows_in"] += chunk.height
            kept, chunk_stats = apply_filters(chunk)
            for key, value in chunk_stats.items():
                if key in ("rows_in", "rows_out"):
                    continue
                stats[key] = stats.get(key, 0) + value
            shard = scratch / f"part-{i:05d}.parquet"
            kept.write_parquet(shard)
            shards.append(shard)

        if not shards:
            raise SystemExit("no chunks produced -- nothing to write")

        table = outdir / f"{dataset}.parquet"
        # One file out, streamed, so nothing large is held in memory.
        pl.scan_parquet([str(p) for p in shards]).sink_parquet(table)

        lf = pl.scan_parquet(table)
        stats["rows_out"] = int(lf.select(pl.len()).collect().item())
        problems = validate.lazy_checks(lf)
        if problems:
            raise ValueError("canonical table failed validation:\n  - "
                             + "\n  - ".join(problems))

        state_counts = dict(lf.group_by("state").len().collect().iter_rows())
        tier_counts = dict(lf.group_by("energy_tier").len().collect().iter_rows())
        coverage = float(
            lf.select(pl.col("energy_j").is_not_null().mean()).collect().item() or 0.0
        )
        return stats, {
            "state_counts": state_counts,
            "energy_tier_counts": tier_counts,
            "energy_coverage": coverage,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
