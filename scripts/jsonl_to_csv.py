#!/usr/bin/env python3
"""
jsonl_to_csv.py — turn newline-delimited JSON job records into a plain CSV.

WHY THIS EXISTS
---------------
The Eagle files fetched by `fetch_data.sh` are JSON Lines: one JSON object per
line, NOT a JSON array. (A normal `json.load()` fails on them with "Extra
data" — you must iterate the lines.) They are also awkward to read by eye, so
this converts each one to CSV for viewing in any spreadsheet.

TWO THINGS WORTH KNOWING
------------------------
1. Fields differ slightly from record to record, so the CSV header is the union
   of every key seen in the file (first pass), then rows are written (second
   pass). Missing keys become empty cells -- no data is silently dropped.
2. Some values are lists (e.g. `nodelist`) or nested objects. Lists are joined
   with `--list-sep` (default ";"), and dicts are written as compact JSON, so
   every cell stays a single CSV field.

USAGE
-----
    python3 scripts/jsonl_to_csv.py 21913139/anon_jobs_2019-12.json
    python3 scripts/jsonl_to_csv.py 21913139/anon_jobs_*.json --rows 5000
    python3 scripts/jsonl_to_csv.py FILE.json --out-dir data/csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten(value, sep: str):
    """Reduce any JSON value to a single CSV-safe string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return sep.join(flatten(v, sep) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return value


def collect_keys(path: Path, limit: int | None) -> list[str]:
    """
    First pass: gather the union of keys, in first-seen order, so the header is
    stable and no field is missed. Reads line by line to stay memory-light.
    """
    keys: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines rather than abort the whole file
            for key in record:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    return keys


def convert(src: Path, dst: Path, limit: int | None, sep: str) -> tuple[int, int]:
    """Second pass: write one CSV row per JSON line. Returns (rows, columns)."""
    keys = collect_keys(src, limit)
    written = 0
    with src.open(encoding="utf-8") as fh, dst.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        for line in fh:
            if limit is not None and written >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            writer.writerow({k: flatten(record.get(k), sep) for k in keys})
            written += 1
    return written, len(keys)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Convert JSON Lines job records to CSV for viewing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="one or more .json (JSON Lines) files")
    ap.add_argument("--out-dir", default=None,
                    help="directory for the CSVs (default: alongside each input)")
    ap.add_argument("--rows", type=int, default=None,
                    help="only convert the first N records per file (default: all)")
    ap.add_argument("--list-sep", default=";",
                    help="separator for list values such as nodelist (default ';')")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.inputs:
        src = Path(name)
        if not src.exists():
            print(f"skip (not found): {src}", file=sys.stderr)
            continue
        dst = (out_dir / (src.stem + ".csv")) if out_dir else src.with_suffix(".csv")
        rows, cols = convert(src, dst, args.rows, args.list_sep)
        size_mb = dst.stat().st_size / 1_048_576
        print(f"{src} -> {dst}  ({rows:,} rows, {cols} columns, {size_mb:.1f} MB)",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
