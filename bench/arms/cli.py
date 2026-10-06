"""
Arm runner CLI.

    python -m bench.arms.cli --arm A \
        --manifest data/manifests/eagle_parquet.folds.json \
        --canonical data/canonical/eagle_parquet.parquet \
        --out runs

Writes `runs/<run_id>/predictions.parquet` and `runs/<run_id>/metadata.json`
(one directory per run, so datasets and variants cannot overwrite each other).
The predictions conform to `metrics.md` section 7; nothing here computes a
metric.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ARMS = ("A",)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.arms",
        description="Run one arm against a frozen split manifest.")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--manifest", required=True, help="folds.json to run against")
    ap.add_argument("--canonical", required=True, help="canonical job table")
    ap.add_argument("--out", default="runs", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--calibrate", action="store_true",
                    help="also apply isotonic calibration (a separate run variant)")
    ap.add_argument("--control-folds", type=int, default=None,
                    help="run the label-shuffle control on only the last N folds "
                         "(default: all folds)")
    ap.add_argument("--folds", type=int, default=None, metavar="N",
                    help="run only the first N folds — the cheapest ones, for a "
                         "quick check (omit for a full run)")
    ap.add_argument("--windows-dir", default=None,
                    help="where to cache the window table (default: <out>/../windows)")
    ap.add_argument("--rebuild-windows", action="store_true",
                    help="rebuild the cached window table even if it exists")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads to use (default 2, so the rest of the "
                         "machine stays responsive)")
    args = ap.parse_args(argv)

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

    if args.arm == "A":
        from .arm_a import run
        print(f"arm A <- {args.manifest}")
        print(f"  threads        {args.threads}")
        meta = run(manifest_path=args.manifest, canonical_path=args.canonical,
                   outdir=args.out, seed=args.seed, calibrate=args.calibrate,
                   control_folds=args.control_folds, windows_dir=args.windows_dir,
                   rebuild_windows=args.rebuild_windows, folds_limit=args.folds)
    else:                                     # pragma: no cover - guarded above
        raise SystemExit(f"arm {args.arm!r} is not implemented yet")

    print(f"  run            {meta['run_id']}")
    print(f"  folds          {meta['folds']}")
    print(f"  test rows      {meta['test_rows']:,} "
          f"(unscorable {meta['unscorable_rows']:,} = "
          f"{100 * (meta['unscorable_fraction'] or 0):.2f}%)")
    print(f"  median fold AUC {meta['fold_auc_median']:.4f}"
          if meta["fold_auc_median"] is not None else "  median fold AUC n/a")
    if meta["control_mean"] is not None:
        print(f"  control mean AUC {meta['control_mean']:.4f} "
              f"over {len(meta['control_aucs'])} shuffled fits")
    print(f"  fit time       {meta['fit_seconds_total']}s")
    print(f"  wrote          {args.out}/{meta['run_id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
