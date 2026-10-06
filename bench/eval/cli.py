"""
Evaluator CLI.

    python -m bench.eval.cli \
        --predictions runs/A-eagle_parquet-raw/predictions.parquet \
        --manifest data/manifests/eagle_parquet.folds.json \
        --canonical data/canonical/eagle_parquet.parquet \
        --arm-metadata runs/A-eagle_parquet-raw/metadata.json \
        --out results --threads 2

Writes `results/<run_id>.json`. Nothing here trains anything, and no metric is
chosen at run time — every rule is in `metrics.md`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.eval",
        description="Score one arm's predictions against a frozen split.")
    ap.add_argument("--predictions", required=True, help="the arm's hand-in table")
    ap.add_argument("--manifest", required=True, help="the frozen folds.json")
    ap.add_argument("--canonical", required=True, help="canonical job table")
    ap.add_argument("--arm-metadata", default=None,
                    help="the arm's metadata.json (carries the shuffle-control AUCs "
                         "and the system metrics)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--task", default="T1", choices=("T1", "T2"))
    ap.add_argument("--unit", default="month", choices=("month", "user"),
                    help="bootstrap resampling unit: months for the full set "
                         "(default), users for the 20,000-job sample")
    ap.add_argument("--resamples", type=int, default=200,
                    help="bootstrap resamples (metrics.md targets 1000; 200 is "
                         "the practical default on a multi-million-row run, and "
                         "the number actually used is recorded in the result)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads to use (default 2, so the rest of the "
                         "machine stays responsive)")
    args = ap.parse_args(argv)

    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    for label, path in (("manifest", args.manifest), ("canonical", args.canonical),
                        ("predictions", args.predictions)):
        if not Path(path).exists():
            raise SystemExit(f"{label} not found: {path}")

    # Imports after the thread budget is set.
    from .predictions import check_contract, load_predictions
    from . import results as results_mod

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    preds = load_predictions(args.predictions)
    print(f"eval: {args.predictions}")
    print(f"  rows {preds.height:,} | arm {preds['arm'][0]} | task {args.task}")

    problems = check_contract(preds, manifest, args.canonical)
    if problems:
        print("\nCONTRACT VIOLATION — the run is refused:", file=os.sys.stderr)
        for p in problems:
            print(f"  - {p}", file=os.sys.stderr)
        return 1
    print("  contract: OK")

    metadata = None
    if args.arm_metadata and Path(args.arm_metadata).exists():
        metadata = json.loads(Path(args.arm_metadata).read_text(encoding="utf-8"))

    result = results_mod.evaluate(
        preds, manifest, args.canonical, task=args.task,
        resamples=args.resamples, seed=args.seed,
        control_aucs=(metadata or {}).get("control_aucs"), metadata=metadata,
        unit=args.unit)

    if not result["valid"]:
        print(f"  INVALID RUN: {result['invalid_reason']}")
        print("  no headline figure will be reported")
    else:
        headline = result["headline"]
        print(f"  control mean AUC {result['control']['mean']:.4f} "
              f"(band {result['control']['band'][0]}-{result['control']['band'][1]})")
        if headline["median"] is not None:
            print(f"  headline {headline['metric']}: median {headline['median']:.3f}, "
                  f"pooled {headline['pooled']:.3f}")
        else:
            print(f"  headline {headline['metric']}: not available "
                  f"({headline['n_months']} usable month(s))")

    path = results_mod.write(result, args.out)
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
