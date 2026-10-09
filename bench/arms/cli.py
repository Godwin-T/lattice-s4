"""
Arm runner CLI.

    python -m bench.arms.cli --arm A \
        --manifest data/manifests/eagle_parquet.folds.json \
        --canonical data/canonical/eagle_parquet.parquet \
        --out runs

Writes `runs/<run_id>/predictions.parquet` and `runs/<run_id>/metadata.json`
(one directory per run, so datasets and variants cannot overwrite each other).
`run_id` is `A-<dataset>-<safe|published>-<raw|iso>`, so the window rule and the
calibration variant are both named in the path. The predictions conform to
`metrics.md` section 7; nothing here computes a metric.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ARMS = ("A", "B1", "B2", "B3")

# The metadata keys the report block reads. Declared here rather than inline so a
# new arm that omits one fails a test instead of failing *after* its run — the
# report is the last thing to touch a run's output, which is the worst possible
# moment to find out a key is missing.
REPORT_KEYS = ("run_id", "window_rule", "submit_time_safe", "folds", "test_rows",
               "unscorable_rows", "unscorable_fraction", "fold_auc_median",
               "control_mean", "control_aucs", "fit_seconds_total")


def arm_kwargs(arm: str, args) -> dict:
    """
    The keyword arguments this arm's `run` is called with.

    A pure function of the parsed arguments, so the dispatch can be checked
    without fitting anything: `inspect.signature(run).bind(**arm_kwargs(...))`
    catches a renamed parameter at test time rather than at the launch of a
    multi-hour run. The per-arm differences are real and load-bearing — B1 is
    uncapped where B2/B3 take `train_row_cap`, and only Arm A has `calibrate` —
    so they are stated here once instead of implied by four near-identical call
    blocks.
    """
    common = {
        "manifest_path": args.manifest, "canonical_path": args.canonical,
        "outdir": args.out, "seed": args.seed,
        "control_folds": args.control_folds, "folds_limit": args.folds,
        "max_chunk_rows": args.chunk_rows, "window": args.window,
    }
    if arm == "A":
        return common | {
            "calibrate": args.calibrate, "windows_dir": args.windows_dir,
            "rebuild_windows": args.rebuild_windows,
        }
    if arm == "B1":
        return common | {
            "features_dir": args.windows_dir,
            "rebuild_features": args.rebuild_windows,
            "num_threads": args.threads,
        }
    if arm in ("B2", "B3"):
        return common | {
            "features_dir": args.windows_dir,
            "rebuild_features": args.rebuild_windows, "num_threads": args.threads,
            "train_row_cap": args.train_rows, "device": args.device,
        }
    raise SystemExit(f"arm {arm!r} is not implemented yet")


def _report(meta: dict) -> list[str]:
    """The lines printed after a run, from the metadata the arm wrote."""
    lines = [f"  run            {meta['run_id']}",
             f"  window rule    {meta['window_rule']} "
             f"(submit-time safe: {meta['submit_time_safe']})",
             f"  folds          {meta['folds']}",
             f"  test rows      {meta['test_rows']:,} "
             f"(unscorable {meta['unscorable_rows']:,} = "
             f"{100 * (meta['unscorable_fraction'] or 0):.2f}%)"]
    lines.append(f"  median fold AUC {meta['fold_auc_median']:.4f}"
                 if meta["fold_auc_median"] is not None
                 else "  median fold AUC n/a")
    if meta["control_mean"] is not None:
        lines.append(f"  control mean AUC {meta['control_mean']:.4f} "
                     f"over {len(meta['control_aucs'])} shuffled fits")
    lines.append(f"  fit time       {meta['fit_seconds_total']}s")
    return lines


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, so the dispatch can be exercised without a terminal."""
    ap = argparse.ArgumentParser(
        prog="bench.arms",
        description="Run one arm against a frozen split manifest.")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--manifest", required=True, help="folds.json to run against")
    ap.add_argument("--canonical", required=True, help="canonical job table")
    ap.add_argument("--out", default="runs", help="output directory")
    ap.add_argument("--window", choices=("safe", "published"), default="safe",
                    help="history-window rule (default: safe). 'safe' uses only "
                         "jobs that had ended before the target was submitted — "
                         "what every arm is scored under. 'published' uses the "
                         "previous 24 jobs in submit order, ended or not, exactly "
                         "as the published replication does; it is not "
                         "submit-time safe and exists only for the side-by-side "
                         "comparison. See bench/arms/common.py.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true",
                    help="also apply isotonic calibration (a separate run variant)")
    ap.add_argument("--control-folds", type=int, default=None,
                    help="run the label-shuffle control on only N folds, spread "
                         "evenly across the run (default: all folds). Each "
                         "controlled fold fits the model `control-repeats` extra "
                         "times, so this is the main lever on total runtime — but "
                         "do not narrow it to the cheapest folds: the earliest "
                         "ones have the fewest users and the arm has no signal on "
                         "them, which leaves the gate nothing to test. See "
                         "control_gate_finding.md section 6.")
    ap.add_argument("--folds", type=int, default=None, metavar="N",
                    help="run only the first N folds — the cheapest ones, for a "
                         "quick check (omit for a full run)")
    ap.add_argument("--windows-dir", default=None,
                    help="where to cache the window table (default: <out>/../windows)")
    ap.add_argument("--rebuild-windows", action="store_true",
                    help="rebuild the cached window table even if it exists")
    ap.add_argument("--chunk-rows", type=int, default=250_000,
                    help="maximum rows per chunk when building the window cache "
                         "(default 250000). Lower it if the machine still runs "
                         "out of memory; it does not change the result.")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads to use (default 2, so the rest of the "
                         "machine stays responsive)")
    ap.add_argument("--train-rows", type=int, default=250_000, metavar="N",
                    help="cap the per-fold training rows (B2/B3 only, default "
                         "250000; 0 disables). A capped run is not directly "
                         "comparable to B1's uncapped one, so the cap is recorded "
                         "in metadata.json either way. See arm-b-planning.md 7.")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="torch device for B2/B3; auto selects CUDA when available")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Set the thread budget BEFORE importing anything that pulls in Polars or
    # the maths libraries, otherwise the limits are ignored and a run can
    # saturate every core.
    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    if not Path(args.manifest).exists():
        raise SystemExit(f"manifest not found: {args.manifest}")
    if not Path(args.canonical).exists():
        raise SystemExit(f"canonical table not found: {args.canonical}")

    from importlib import import_module

    module = import_module(f".arm_{args.arm.lower()}", package=__package__)
    print(f"arm {args.arm} <- {args.manifest}")
    print(f"  threads        {args.threads}")
    meta = module.run(**arm_kwargs(args.arm, args))

    for line in _report(meta):
        print(line)
    print(f"  wrote          {args.out}/{meta['run_id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
