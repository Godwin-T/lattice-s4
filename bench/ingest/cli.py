"""
Ingest CLI.

    python -m bench.ingest.cli --dataset <name> --raw <files...> --out <dir>

Writes `<out>/<dataset>.parquet` and `<out>/<dataset>.audit.json`.

Two build paths, chosen automatically:

* **In-memory** — the adapter exposes `load(paths)`. Used by the small sources
  (eagle_jsonl, kestrel): map everything, filter, validate, write.
* **Streaming** — the adapter exposes `iter_canonical(paths)`. Used by the large
  Eagle 11M source: map and filter one chunk at a time, spill shards, stream
  them into one file. Peak memory stays proportional to the chunk.

Both paths produce the same table and the same audit file.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from . import validate
from .filters import apply_filters
from .hashing import resolve_salt
from .schema import INGEST_VERSION

DATASETS = ("eagle_jsonl", "kestrel", "eagle_parquet")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _adapter(dataset: str):
    """Load the adapter module named after the dataset."""
    try:
        return importlib.import_module(f".adapters.{dataset}", package=__package__)
    except ModuleNotFoundError as exc:               # unknown dataset
        raise SystemExit(f"unknown dataset {dataset!r}") from exc


def _audit(stats: dict, summary: dict, sources: list[str]) -> dict:
    """Aggregate, non-identifying provenance for the canonical table."""
    return {
        "ingest_version": INGEST_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
        "rows": stats,
        "state_counts": summary["state_counts"],
        "energy_tier_counts": summary["energy_tier_counts"],
        "energy_coverage": summary["energy_coverage"],
    }


def _summarise_df(df: pl.DataFrame) -> dict:
    return {
        "state_counts": dict(df.group_by("state").len().iter_rows()),
        "energy_tier_counts": dict(df.group_by("energy_tier").len().iter_rows()),
        "energy_coverage": (float(df["energy_j"].is_not_null().mean())
                            if df.height else None),
    }


def _report_drops(stats: dict) -> str:
    drops = ", ".join(f"{k}={v:,}" for k, v in stats.items()
                      if k.startswith("dropped_") and v)
    return f"({drops})" if drops else "(no rows dropped)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.ingest",
        description="Raw scheduler source -> canonical job table.")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--raw", nargs="+", required=True, help="raw source file(s)")
    ap.add_argument("--out", default="data/canonical", help="output directory")
    ap.add_argument("--salt", default=None,
                    help="hash salt for sources that hash identifiers "
                         "(currently eagle_parquet). May also come from "
                         "BENCH_SALT_<DATASET> or BENCH_SALT.")
    args = ap.parse_args(argv)

    print(f"ingest: {args.dataset} -> {args.out}")
    adapter = _adapter(args.dataset)

    salt = resolve_salt(args.dataset, args.salt)
    if getattr(adapter, "REQUIRES_SALT", False) and not salt:
        raise SystemExit(
            f"--dataset {args.dataset} hashes identifiers and therefore needs a "
            f"salt: pass --salt, or set BENCH_SALT_{args.dataset.upper()} "
            f"(or BENCH_SALT). The same salt must be used for every run, "
            f"otherwise the identifiers change and tables stop matching."
        )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    table = outdir / f"{args.dataset}.parquet"

    if hasattr(adapter, "iter_canonical"):
        # Large source: stream chunk by chunk and spill shards.
        from .build import build_from_chunks
        print("  streaming build (chunked)")
        stats, summary = build_from_chunks(
            adapter.iter_canonical(args.raw, salt=salt), outdir, args.dataset)
        print(f"  mapped {stats['rows_in']:,} rows")
        print(f"  kept {stats['rows_out']:,} rows {_report_drops(stats)}")
        print("  validation: OK")
    else:
        df = adapter.load(args.raw, salt=salt)
        print(f"  mapped {df.height:,} rows")
        df, stats = apply_filters(df)
        print(f"  kept {stats['rows_out']:,} rows {_report_drops(stats)}")

        problems = validate.run_checks(df)
        if problems:
            print("\nVALIDATION FAILED:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("  validation: OK")
        df.write_parquet(table)
        summary = _summarise_df(df)

    audit = _audit(stats, summary, args.raw)
    audit["output"] = {"path": str(table), "sha256": _sha256(table),
                       "rows": stats["rows_out"]}
    (outdir / f"{args.dataset}.audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    print(f"  wrote {table}")
    print(f"  wrote {outdir / (args.dataset + '.audit.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
