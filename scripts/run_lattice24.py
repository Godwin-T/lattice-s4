#!/usr/bin/env python3
"""
run_lattice24.py — end-to-end runner for the Lattice24 timeout-exposure method
on the NREL Eagle job archive (or any comparable scheduler export).

WHAT THIS SCRIPT DOES, IN ONE SENTENCE
--------------------------------------
It reads a raw Slurm job archive (Parquet), converts it into the exact
sacct-style columns the Lattice24 method needs, then runs the *published*
Lattice24 pipeline — windowing, training, forward-chained evaluation and the
label-shuffle control — and writes the same report/summary the `lattice24-assess`
CLI would produce, plus a scorecard-shaped metrics file.

WHY A SEPARATE SCRIPT?
----------------------
The installed package (`lattice24_assess`) only reads sacct/CSV text files. The
Eagle data is a Parquet file with different column names. Rather than modify the
reference implementation (which must stay frozen to remain a fair benchmark
baseline), this script does the translation and then calls the package's own
functions, so the *method* is exactly the published one.

THE PREDICTION TARGET (read this before interpreting anything)
--------------------------------------------------------------
Lattice24 predicts a single binary label for each job:

        label = 1  if  State == "TIMEOUT"   (the job hit its wall-clock limit)
        label = 0  otherwise

and it predicts it from the *same user's previous 24 jobs only*. Nothing about
the job being predicted enters its own features. The label lives in the `state`
column of the input data — it is a recorded outcome, not something invented.
IMPORTANT: only TIMEOUT is positive. COMPLETED, FAILED, CANCELLED,
OUT_OF_MEMORY and NODE_FAIL are all *negatives* (they are not the target here).

HOW TO RUN
----------
    # Full dataset (this can take a long time — see note below):
    python scripts/run_lattice24.py --out ./run_lattice24

    # A 12-month slice (much faster; the README's own recommended window):
    python scripts/run_lattice24.py --start 2022-03-01 --out ./run_lattice24

RUNTIME NOTE
------------
Cost grows with the number of months (each month = one model fit, and the
control runs every fit 3 more times). The full 2018-2023 archive has ~52 months
and ~11M windows, which is hours of CPU. A 12-month slice is minutes. Use
`--start`/`--end` and/or `--max-users` while iterating.

This script does NOT modify or depend on changing the package source. It only
imports and calls it.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the local package importable when this script is run from a clone,
# without requiring `pip install -e .`.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# The five columns Lattice24 requires, mapped to the names the method expects.
# The package lowercases headers and accepts several aliases, but we emit these
# canonical headers so the intent is obvious.
CANONICAL_HEADERS = ["User", "End", "Timelimit", "Elapsed", "State"]


# ---------------------------------------------------------------------------
# 1. Loading and conversion
# ---------------------------------------------------------------------------

def seconds_to_slurm(seconds: float) -> str:
    """
    Convert a number of seconds into the Slurm duration string the package
    parses: "[DD-]HH:MM:SS".

    WHY THIS MATTERS: the package's duration parser treats a *bare integer* as
    minutes. If we wrote raw seconds (e.g. 3600) it would be read as 3600
    minutes and every time limit would be 60x too large. Always emit strings.
    """
    total = int(max(0.0, float(seconds)))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clean_text(series) -> "object":
    """
    Neutralise anything that could break the pipe-delimited file we write:
    drop the delimiter and collapse newlines. Only applied to identifier/state
    columns, never to the numeric ones.
    """
    return (
        series.astype("string")
        .str.replace("|", "/", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .fillna("")
    )


def load_and_convert(args) -> tuple[str, dict]:
    """
    Read the Parquet archive and write a sacct-style text file the package can
    consume. Returns (path_to_text_file, metadata_dict).

    Steps:
      1. Read only the columns we need (memory-friendly on an 11M-row file).
      2. Apply optional date and user filters.
      3. Rename/reshape Eagle columns into User/End/Timelimit/Elapsed/State.
      4. Reformat the two duration columns from seconds to Slurm strings.
      5. Write a '|'-delimited file (the same format `sacct --parsable2` emits).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "pyarrow is required to read Parquet. Install it with "
            "`pip install pyarrow` (pandas alone is not enough for large files)."
        ) from exc

    src = Path(args.parquet)
    if not src.exists():
        raise SystemExit(f"Input Parquet not found: {src}")

    # -- 1. Read only what we need ------------------------------------------
    wanted = ["user", "end_time", "wallclock_req", "run_time", "state"]
    if args.energy_col:
        wanted.append(args.energy_col)

    pf = pq.ParquetFile(src)
    available = set(pf.schema_arrow.names)
    missing = [c for c in wanted if c not in available]
    if missing:
        raise SystemExit(
            f"{src} is missing required column(s): {missing}. "
            f"Columns present: {sorted(available)}"
        )
    # job_id is not needed for the method, but keeping it lets a viewer trace a
    # row back to the source record (the PRD's "traceability" idea).
    if "job_id" in available and "job_id" not in wanted:
        wanted = wanted + ["job_id"]

    print(f"  reading {src} (columns: {', '.join(wanted)}) ...", flush=True)
    table = pf.read(columns=wanted)
    df = table.to_pandas()
    if args.limit_rows:
        df = df.iloc[: args.limit_rows]
    rows_start = len(df)

    # -- 2. Filters ----------------------------------------------------------
    # Drop rows with no usable outcome/inputs BEFORE conversion; the package
    # would drop them anyway, but doing it here keeps the intermediate file small.
    df = df.dropna(subset=["user", "end_time", "wallclock_req", "run_time", "state"])

    if args.start:
        start = datetime.fromisoformat(args.start)
        df = df[df["end_time"] >= start]
    if args.end:
        end = datetime.fromisoformat(args.end)
        df = df[df["end_time"] <= end]
    if df.empty:
        raise SystemExit("No rows remain after the date filter. Relax --start/--end.")

    # Optional: keep only the N busiest users. Handy for fast smoke tests.
    if args.max_users:
        busiest = df["user"].value_counts().head(args.max_users).index
        df = df[df["user"].isin(busiest)]

    # -- 3./4. Reshape into the canonical sacct columns ----------------------
    out = {
        "User": _clean_text(df["user"]),
        "End": df["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "Timelimit": [seconds_to_slurm(x) for x in df["wallclock_req"].to_numpy()],
        "Elapsed": [seconds_to_slurm(x) for x in df["run_time"].to_numpy()],
        "State": _clean_text(df["state"]).str.upper().str.split().str[0],
    }
    if "job_id" in df.columns:
        out["Job ID"] = df["job_id"]

    # Energy is optional and absent in the plain Eagle archive. If the caller
    # points at a column holding joules, pass it through as ConsumedEnergyRaw.
    if args.energy_col:
        out["ConsumedEnergyRaw"] = df[args.energy_col].fillna("")

    import pandas as pd  # local import keeps the top of the file readable
    converted = pd.DataFrame(out)

    # -- 5. Write ------------------------------------------------------------
    dest = Path(args.converted) if args.converted else Path(
        tempfile.mkstemp(prefix="lattice24_", suffix=".psv")[1]
    )
    converted.to_csv(dest, sep="|", index=False)

    # A few facts worth surfacing before the (slow) modelling stage.
    n_timeouts = int((converted["State"] == "TIMEOUT").sum())
    meta = {
        "parquet": str(src),
        "rows_in_file": rows_start,
        "rows_converted": int(len(converted)),
        "users": int(converted["User"].nunique()),
        "timeouts": n_timeouts,
        "timeout_share_of_jobs": (n_timeouts / len(converted)) if len(converted) else None,
        "first_end": str(converted["End"].min()),
        "last_end": str(converted["End"].max()),
        "converted_path": str(dest),
        "energy_column": args.energy_col,
    }
    print(
        f"  converted {meta['rows_converted']:,} rows, {meta['users']:,} users, "
        f"{meta['timeouts']:,} TIMEOUT rows "
        f"({100 * meta['timeout_share_of_jobs']:.1f}% of jobs)",
        flush=True,
    )
    print(f"  wrote {dest}", flush=True)

    # -- 5b. Optional human-viewable CSV ------------------------------------
    # The intermediate file above is pipe-delimited and tuned for the parser,
    # not for people. If --csv was given, also write a plain CSV of the same
    # rows so it can be opened in any spreadsheet. --csv-rows caps the size so
    # the file stays viewable (0 = write every row).
    if args.csv:
        view = converted if not args.csv_rows else converted.head(args.csv_rows)
        csv_path = Path(args.csv)
        if csv_path.parent and str(csv_path.parent) not in ("", "."):
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        view.to_csv(csv_path, index=False)
        print(f"  viewable CSV: {csv_path} ({len(view):,} rows)", flush=True)

    return str(dest), meta


# ---------------------------------------------------------------------------
# 2. The Lattice24 pipeline (calls the frozen reference implementation)
# ---------------------------------------------------------------------------

def run_pipeline(psv_path: str, args) -> tuple[dict, dict, dict]:
    """
    Run the published method, stage by stage, exactly as the CLI does:

        read_records      parse + hash users + derive the TIMEOUT label
        mark_restart_*    tag likely checkpoint-restart links (intentional timeouts)
        build_windows     assemble the (24-job history -> next-job label) samples
        check_sufficient  refuse if the data cannot support an honest answer
        forward_chain     train on months 1..k, score month k+1 (never reversed)
        shuffle_control   scramble labels; a result above chance invalidates the run
        build_summary     assemble the report/summary payload
        write_outputs     write the HTML report and JSON summary

    Returns (summary, forward_chain_result, control_result).
    """
    from lattice24_assess import __version__, core, report

    # -- Parse ---------------------------------------------------------------
    print("  parsing records ...", flush=True)
    jobs, stats = core.read_records(psv_path, delimiter="|")
    print(f"    {stats['rows_read']:,} rows -> {len(jobs):,} usable finished jobs", flush=True)

    # -- Restart-chain heuristic (secondary attribute, not the target) -------
    stats.update(core.mark_restart_chains(jobs, gap_s=args.restart_gap * 3600.0))
    print(
        f"    {stats['timeouts_restart_likely']:,} of {stats['timeouts_seen']:,} timeouts "
        f"look like checkpoint-restart links (same user, same limit, within {args.restart_gap:g}h)",
        flush=True,
    )

    # -- Windows: the (history -> label) samples -----------------------------
    print("  building windows ...", flush=True)
    w = core.build_windows(jobs)
    stats.update({
        "windows": int(len(w.y)),
        "timeouts": int(w.y.sum()),
        "n_months": len(w.months),
        "first_month": w.months[0],
        "last_month": w.months[-1],
    })
    print(
        f"    {stats['windows']:,} scoreable windows, {stats['timeouts']:,} timeout windows, "
        f"{stats['n_months']} months ({stats['first_month']}..{stats['last_month']})",
        flush=True,
    )

    # -- Refuse rather than degrade (raises core.Refusal) --------------------
    core.check_sufficient(w)

    # -- Forward-chained training and scoring --------------------------------
    print("  forward-chaining (train on past months, score the next) ...", flush=True)
    t0 = time.time()
    fc = core.forward_chain(w, seed=args.seed)
    print(f"    {fc['n_splits']} splits in {time.time() - t0:.0f}s", flush=True)

    # -- Label-shuffle control: is there real signal, or a leak? -------------
    print("  running label-shuffle control ...", flush=True)
    ctl = core.shuffle_control(w, seed=args.seed)
    if ctl["passed"]:
        print(
            f"    control mean AUC {ctl['mean']:.3f} (passed, tolerance +/-{ctl['tolerance']:.3f})",
            flush=True,
        )
    else:
        print("    CONTROL FAILED - the report will show no headline figure.", flush=True)

    # -- Assemble and write the same artifacts the CLI writes ----------------
    summary = report.build_summary(
        {"version": __version__, "site_label": args.site}, stats, fc, ctl
    )
    json_path, html_path = report.write_outputs(args.out, summary)
    print(f"    report : {html_path}\n    summary: {json_path}", flush=True)
    return summary, fc, ctl


# ---------------------------------------------------------------------------
# 3. Scorecard-shaped readout (connects the run to the benchmark spec)
# ---------------------------------------------------------------------------

def scorecard_rows(summary: dict) -> list[dict]:
    """
    Re-express the Lattice24 output in the shape the benchmark scorecard wants.

    The method flags the top (100 - p)% of jobs at percentile p. So the row at
    percentile 95 == "top 5% flagged", which is the benchmark's operating point.
    'Energy captured' is only computable when the input carried energy.
    """
    fc = summary["forward_chained"]
    pooled = fc["pooled"]
    energy = fc["energy"]
    total_pos_energy = energy.get("timeout_kwh") or 0.0

    rows = []
    for p in (50, 75, 80, 90, 95):
        # NOTE: in memory the threshold keys are ints; after a JSON round-trip
        # they become strings. Accept both so this works either way.
        t = pooled["thresholds"].get(str(p), pooled["thresholds"].get(p))
        if t is None:
            continue
        captured = (
            t["energy_at_stake_kwh"] / total_pos_energy if total_pos_energy > 0 else None
        )
        rows.append({
            "flag_rate_pct": 100 - p,          # top-k% of jobs flagged
            "recall": t["recall"],
            "false_flag": t["false_flag"],
            "energy_captured": captured,
        })
    return rows


def print_scorecard(summary: dict) -> None:
    """Human-readable summary of the numbers a benchmark reviewer cares about."""
    fc = summary["forward_chained"]
    ctl = summary["shuffle_control"]
    inp = summary["input"]

    print("\n" + "-" * 72)
    print("SCORECARD-RELEVANT RESULTS (Lattice24 / benchmark task T1)")
    print("-" * 72)
    if not ctl.get("passed"):
        print("Run INVALID: the label-shuffle control failed. Numbers below are void.")
    print(f"Pooled AUC (median-of-months spirit): {fc['pooled']['auc']:.4f}")
    print(f"Per-split AUC range: {min(fc['auc_per_split']):.3f}-{max(fc['auc_per_split']):.3f} "
          f"over {fc['n_splits']} splits")
    print(f"Timeout share of scored windows: "
          f"{inp['timeouts'] / inp['windows'] * 100:.1f}%")

    print(f"\n{'flag top':>9} {'recall':>8} {'false flag':>11} {'energy captured':>16}")
    for r in scorecard_rows(summary):
        ec = f"{100 * r['energy_captured']:.1f}%" if r["energy_captured"] is not None else "n/a (no energy)"
        print(f"{r['flag_rate_pct']:>8}% {100 * r['recall']:>7.1f}% "
              f"{100 * r['false_flag']:>10.1f}% {ec:>16}")

    if (fc["energy"].get("timeout_kwh") or 0.0) == 0.0:
        print("\nNOTE: no energy column was present, so 'energy captured' cannot be "
              "computed. Supply one with --energy-col to fill those cells.")
    print("\nReminder: these are energy AT STAKE, not energy saved.")
    print("-" * 72)


# ---------------------------------------------------------------------------
# 4. Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run the published Lattice24 timeout method on an Eagle-style "
                    "Parquet archive. Loads, converts, trains, evaluates, reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python scripts/run_lattice24.py --start 2022-03-01\n"
               "  python scripts/run_lattice24.py --max-users 200 --start 2022-01-01\n",
    )
    ap.add_argument("--parquet", default=str(REPO_ROOT / "data" / "eagle_data.parquet"),
                    help="input Parquet archive (default: data/eagle_data.parquet)")
    ap.add_argument("--out", default="./run_lattice24",
                    help="output directory for the report and summary")
    ap.add_argument("--site", default=None, help="label for the report heading")
    ap.add_argument("--start", default=None, help="only rows with end_time >= this ISO date")
    ap.add_argument("--end", default=None, help="only rows with end_time <= this ISO date")
    ap.add_argument("--max-users", type=int, default=None,
                    help="keep only the N busiest users (fast smoke tests)")
    ap.add_argument("--energy-col", default=None,
                    help="optional column holding per-job energy in joules")
    ap.add_argument("--restart-gap", type=float, default=2.0, metavar="HOURS",
                    help="checkpoint-restart linking window (default 2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--converted", default=None,
                    help="path to write the intermediate sacct-style file "
                         "(default: a temporary file)")
    ap.add_argument("--keep-converted", action="store_true",
                    help="do not delete the intermediate file on exit")
    ap.add_argument("--csv", default=None, metavar="PATH",
                    help="also write the converted rows as a plain, "
                         "spreadsheet-friendly CSV to PATH")
    ap.add_argument("--csv-rows", type=int, default=20_000,
                    help="rows to include in the --csv file (default 20000; "
                         "0 = all rows)")
    ap.add_argument("--limit-rows", type=int, default=None,
                    help="read only the first N rows (debugging)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print("lattice24-assess - end-to-end runner")
    print("Runs entirely on this machine. No network calls. Nothing is uploaded.\n")

    t_start = time.time()
    psv_path, meta = load_and_convert(args)

    # Clean up the intermediate file only if we created it ourselves and the
    # caller did not ask to keep it.
    generated_temp = args.converted is None
    try:
        summary, _fc, _ctl = run_pipeline(psv_path, args)
    except Exception as exc:
        # The package raises core.Refusal when the data cannot support an honest
        # answer; surface that message plainly instead of a traceback.
        from lattice24_assess.core import Refusal
        if isinstance(exc, Refusal):
            print("\n" + "=" * 72, file=sys.stderr)
            print("No report was produced - the data cannot support an honest answer.",
                  file=sys.stderr)
            print("=" * 72, file=sys.stderr)
            print(str(exc), file=sys.stderr)
            return 2
        raise
    finally:
        if generated_temp and not args.keep_converted:
            try:
                os.remove(psv_path)
            except OSError:
                pass

    print_scorecard(summary)
    print(f"\nTotal wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
