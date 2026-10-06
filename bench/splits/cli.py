"""
Split-generator CLI (splits.md T8).

    python -m bench.splits.cli --dataset eagle_parquet \
        --canonical data/canonical/eagle_parquet.parquet \
        --out data/manifests

Reads one canonical table, writes `<out>/<dataset>.folds.json`.

One note on ordering: the month floor (S1) is applied to the **eligible**
population — the rows that are actually split — not to the raw table. A month
that passes the floor overall but falls below the guardrails once ineligible
users are removed would otherwise occupy a holdout slot while contributing
nothing to any fold.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

from . import manifest as manifest_mod
from .config import DEFAULT_SEED
from .eligibility import eligible_summary, eligible_users
from .folds import build_folds, month_stats
from .months import month_counts, timeline
from .sample import build_sample, with_activity_quartile
from .validate import check_manifest

DATASETS = ("eagle_parquet", "kestrel", "eagle_jsonl")


def _counts_by_month(lf: pl.LazyFrame) -> dict[str, int]:
    df = month_counts(lf)
    return {m: int(n) for m, n in zip(df["month"].to_list(), df["len"].to_list())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.splits",
        description="Canonical job table -> frozen folds manifest.")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--canonical", default=None,
                    help="canonical table (default: data/canonical/<dataset>.parquet)")
    ap.add_argument("--out", default="data/manifests", help="output directory")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true",
                    help="report the split without writing anything")
    args = ap.parse_args(argv)

    canonical = args.canonical or f"data/canonical/{args.dataset}.parquet"
    if not Path(canonical).exists():
        raise SystemExit(f"canonical table not found: {canonical}")

    print(f"splits: {args.dataset} <- {canonical}")
    lf = pl.scan_parquet(canonical)
    rows = int(lf.select(pl.len()).collect().item())

    users = eligible_users(lf)
    eligible = eligible_summary(users)
    print(f"  rows {rows:,} | eligible users {eligible['count']:,}")

    lf_e = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    months, dropped = timeline(lf_e)
    for month, n in dropped:
        print(f"  dropped month {month} ({n:,} eligible rows, below the floor)")
    print(f"  timeline: {len(months)} months, {months[0]} .. {months[-1]}")

    stats = month_stats(lf_e)
    try:
        folds, locked, open_months = build_folds(months, stats)
    except ValueError as exc:
        print("\nREFUSED:", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 2

    print(f"  open months {len(open_months)} | folds {len(folds)} "
          f"({folds[0]['test_month']} .. {folds[-1]['test_month']})")
    print(f"  locked holdout {locked[0]} .. {locked[-1]}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    sample = build_sample(lf_e, open_months[1:], with_activity_quartile(users),
                          seed=args.seed)
    print(f"  sample {sample['n']:,} jobs over {len(sample['strata'])} strata")

    built = manifest_mod.build_manifest(
        dataset=args.dataset, canonical_path=canonical, canonical_rows=rows,
        months=months, dropped_months=dropped, eligible=eligible, folds=folds,
        locked=locked, open_months=open_months, sample=sample, seed=args.seed)

    problems = check_manifest(built, month_counts=_counts_by_month(lf_e), lf=lf_e)
    if problems:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("  validation: OK")

    path, digest = manifest_mod.write_manifest(built, args.out, args.dataset)
    print(f"  wrote {path}")
    print(f"  manifest sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
