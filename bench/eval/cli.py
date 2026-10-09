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
import time
from pathlib import Path

# A pure-constants module: safe to import before the thread budget is set below.
from .config import BOOTSTRAP_RESAMPLES


def headline_line(headline: dict | None) -> str:
    """
    The one-line headline summary, for each shape `results.evaluate` can emit.

    There are three, and conflating them was a real bug: a valid run on a dataset
    with no measured energy emits `{"available": False, "reason": ...}` with **no
    `median` key at all** (metrics.md §9 wants "not available", never zeros), so a
    printer that read `headline["median"]` crashed *after* the bootstrap and
    *before* the result was written — thirty-five minutes of work discarded. This
    is a function so that shape has a test.
    """
    if headline is None:
        return "headline: none"
    metric = headline.get("metric", "headline")
    if headline.get("available") is False:
        return f"{metric}: not available — {headline.get('reason', 'no reason given')}"
    if headline.get("median") is not None:
        return (f"{metric}: median {headline['median']:.3f}, "
                f"pooled {headline['pooled']:.3f}")
    # Valid and available, but no month cleared the minimum: a sampling accident,
    # not a missing column, so it is reported as the month count it is.
    return (f"{metric}: not available "
            f"({headline.get('n_months', 0)} usable month(s))")


def gate_2_line(gate: dict | None) -> str:
    """
    The Gate 2 verdict — did the arm beat both naive baselines (metrics.md §6.1).

    Three shapes again, and the same lesson as `headline_line`: an undecided
    comparison is neither a pass nor a failure, and must not print as either.
    """
    if gate is None:
        return "gate 2 (baselines): not computed"
    if not gate["comparisons"]:
        return "gate 2 (baselines): nothing to compare against"
    undecided = [c for c in gate["comparisons"] if c["beats"] is None]
    if undecided:
        return (f"gate 2 (baselines): UNDECIDED — {len(undecided)} comparison(s) "
                f"could not be made")
    verdict = "beats both" if gate["passed"] else "DOES NOT BEAT"
    margins = ", ".join(f"{c['metric']} vs {c['baseline']} {c['margin']:+.4f}"
                        for c in gate["comparisons"])
    return f"gate 2 (baselines): {verdict}  ({margins})"


def calibration_line(calibration: dict | None) -> str:
    """The ECE one-liner, for each shape `_calibration_block` can emit."""
    if calibration is None:
        return "calibration: not computed"
    if not calibration["available"]:
        return f"calibration: not available — {calibration['reason']}"
    block = calibration["ece"]
    if block["median"] is None:
        return (f"calibration: ECE not available "
                f"({block['n_months']} usable month(s))")
    return (f"calibration: ECE median {block['median']:.4f}, "
            f"pooled {block['pooled']:.4f} ({calibration['bins']} bins)")


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
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES,
                    help="bootstrap resamples (metrics.md §5.1: 1000; the draws "
                         "are weighted, not re-scored, so the default is "
                         "practical on a multi-million-row run, and the number "
                         "actually used is recorded in the result)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip the two naive baselines (metrics.md §6.1). They "
                         "cost an extra pass over the test rows, so a quick "
                         "re-score can turn them off; the result then records "
                         "them as null rather than as computed")
    ap.add_argument("--no-calibration", action="store_true",
                    help="skip ECE (metrics.md E5); the result records it as null")
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads to use (default 2, so the rest of the "
                         "machine stays responsive)")
    args = ap.parse_args(argv)

    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)

    # Line-buffer stdout, so a run piped to a log reports as it goes instead of
    # dumping everything at exit (a 20-minute run otherwise looks like a hang).
    try:
        os.sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                  # pragma: no cover
        pass

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

    # Keep only what the evaluator reads from here on. The contract check needed
    # the full hand-in table; scoring does not, and `job_id` alone is nine
    # million Python-visible strings once it is turned into an array. The `run_id`
    # and `arm` columns stay because the result file names the run, and
    # `probability` is kept only when calibration will read it — it is a float per
    # row, and nothing else downstream wants it.
    keep = ["run_id", "arm", "row_id", "test_month", "score"]
    if not args.no_calibration and "probability" in preds.columns:
        keep.append("probability")
    preds = preds.select(keep)

    metadata = None
    if args.arm_metadata and Path(args.arm_metadata).exists():
        metadata = json.loads(Path(args.arm_metadata).read_text(encoding="utf-8"))

    # The bootstrap is a long silent stretch on a full-set run, and piped output
    # is block-buffered, so without an explicit heartbeat a 20-minute step looks
    # like a hang. One line every 100 draws, with elapsed time and live RSS.
    started = time.time()

    def _heartbeat(done: int, total: int) -> None:
        if done % 100 and done != total:
            return
        elapsed = time.time() - started
        rss = ""
        try:
            with open("/proc/self/status", encoding="ascii") as handle:
                for line in handle:
                    if line.startswith("VmRSS"):
                        rss = f" | rss {int(line.split()[1]) / 1024:,.0f} MB"
                        break
        except OSError:
            pass
        print(f"  bootstrap {done}/{total} draws | {elapsed:,.0f}s elapsed{rss}",
              file=os.sys.stderr, flush=True)

    result = results_mod.evaluate(
        preds, manifest, args.canonical, task=args.task,
        resamples=args.resamples, seed=args.seed,
        control_aucs=(metadata or {}).get("control_aucs"),
        control_folds=(metadata or {}).get("control_folds_used"),
        arm_auc=(metadata or {}).get("fold_auc_by_fold"),
        metadata=metadata, unit=args.unit, progress=_heartbeat,
        with_baselines=not args.no_baselines,
        with_calibration=not args.no_calibration)

    # Always printed, valid or not: the two questions the gate asks, and where
    # the spec's old fixed band would have landed (reported, not decisive).
    control = result["control"]
    if control["n"]:
        separation = control["separation"]
        if not separation["evaluated"]:
            verdict, gap = "NOT CHECKED", separation["reason"]
        else:
            verdict = "yes" if separation["passed"] else "NO"
            gap = (f"gap {separation['mean']:.4f} over {separation['n_folds']} "
                   f"folds, need {separation['required']:.4f}")
        print(f"  control: {control['n']} shuffled fits, mean {control['mean']:.4f}, "
              f"sd {control['sd']:.4f}")
        print(f"    at chance: {'yes' if control['chance']['passed'] else 'NO'}"
              f"  ({control['chance']['distance']:.4f} from 0.5, tolerance "
              f"{control['chance']['tolerance']:.4f})")
        print(f"    beats its own shuffle: {verdict}  ({gap})")
        print(f"    spec band {control['band'][0]}-{control['band'][1]}: "
              f"{'would pass' if control['spec_band']['passed'] else 'would fail'}"
              f"  (reported only)")

    print("  " + gate_2_line(result["gate_2"]))
    print("  " + calibration_line(result["calibration"]))

    if not result["valid"]:
        print(f"  INVALID RUN: {result['invalid_reason']}")
        print("  no headline figure will be reported")
    else:
        print("  " + headline_line(result["headline"]))

    path = results_mod.write(result, args.out)
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
