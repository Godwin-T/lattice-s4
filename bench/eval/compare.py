"""
Compare two arms on the same split: is B really better than A? (B-F5, W1.10)

    python3 -m bench.eval.compare \
        --a runs/A-eagle_parquet-safe-raw/predictions.parquet \
        --b runs/B1-eagle_parquet-safe-t1/predictions.parquet \
        --manifest data/manifests/eagle_parquet.folds.json \
        --canonical data/canonical/eagle_parquet.parquet \
        --task T1 --unit month --out results

Writes `results/compare-<a>_vs_<b>-<task>.json`.

**Why this is a separate program rather than a line in a result file.** A single
arm's result answers "how well does this arm do". "Is B better than A" is a
question about the *difference*, and the difference has its own confidence range
that cannot be recovered from the two arms' separate ranges. Two ranges that
overlap do not mean "no difference" — each carries the arm's own noise, and
combining them that way is the standard error of the eye. Worse, it errs
*conservatively*: it hides real differences instead of inventing them, so a
report built on it would understate every real win.

The construction here is **paired**: one shared sequence of draws, both arms
scored within each draw, the difference taken there. The months that happen to
be easy for both arms cancel, so what remains is the arms' disagreement.

**The rule, stated rather than implied.** B beats A when the paired range of
`B − A` on AUC excludes zero and lies above it. An undecided range is reported
as undecided — never as "no difference", and never as a win.

**The rows.** The two arms must have been scored on the same rows, or there is
nothing to pair. They are aligned on `row_id` and the intersection is taken, with
`a_only` / `b_only` counts recorded in the report: a coverage difference is
itself a finding, and it must be visible rather than silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

COMPARE_SCHEMA = "lattice24-benchmark/compare/1"

# The metric the H1 verdict is stated on. AUC for both datasets (arm-b-planning.md
# B-D3): the energy-bearing headline lives on Kestrel, but "B1 beats A" is a claim
# about ranking, and it is made the same way on both datasets.
VERDICT_METRIC = "auc"

VERDICT_RULE = ("B beats A when the paired confidence range of `B - A` on AUC "
                "excludes zero and lies above it")


def verdict(difference: dict, metric: str = VERDICT_METRIC) -> dict:
    """
    The H1 verdict from a paired difference, and the sentence that states it.

    Three outcomes, kept distinct: B wins, A wins, or the range straddles zero
    and nothing can be claimed. The last is not "no difference" — the range may
    be wide, or the effect small — and it is never printed as a win.
    """
    block = difference["metrics"][metric]
    favours = block["favours"]
    low, high = block["ci_low"], block["ci_high"]
    median = block["median"]
    if favours is None:
        if low is None:
            return {"metric": metric, "beats": None, "favours": None,
                    "statement": (f"{metric}: too few usable draws for a range, "
                                  f"so nothing can be claimed")}
        return {"metric": metric, "beats": False, "favours": None,
                "statement": (f"{metric}: B - A is {median:+.4f}, but the range "
                              f"[{low:+.4f}, {high:+.4f}] straddles zero, so B "
                              f"is not shown to beat A")}
    winner = "B" if favours == "b" else "A"
    return {"metric": metric, "beats": favours == "b", "favours": favours,
            "statement": (f"{metric}: {winner} is better by {median:+.4f} "
                          f"(range [{low:+.4f}, {high:+.4f}], excludes zero)")}


def align(a, b):
    """
    The rows both arms scored, and the counts the join dropped.

    Sorted by `row_id`, so the two frames are in one order and a paired draw
    lands on the same rows on both sides. The drops are returned, not discarded:
    an arm that scored fewer rows answered a smaller question, and the report
    should say so.
    """
    import polars as pl

    common = (a.select(["row_id"]).join(b.select(["row_id"]),
                                        on="row_id", how="inner"))
    a_aligned = a.join(common, on="row_id", how="inner").sort("row_id")
    b_aligned = b.join(common, on="row_id", how="inner").sort("row_id")
    counts = {
        "common": common.height,
        "a_only": a.height - common.height,
        "b_only": b.height - common.height,
        "a_total": a.height,
        "b_total": b.height,
    }
    if not a_aligned["row_id"].equals(b_aligned["row_id"]):
        raise ValueError("alignment failed: the two arms are not on the same rows")
    # Both arms must agree on the truth as well as the rows, or the comparison is
    # between two different questions. They read the same canonical table, so a
    # mismatch here is a bug, not a data property.
    if not a_aligned["label"].equals(b_aligned["label"]):
        raise ValueError("the two arms disagree on the labels of rows they share")
    del a, b
    return a_aligned, b_aligned, counts


def per_month_difference(arrays_a: dict, arrays_b: dict) -> dict:
    """
    Each arm's AUC per month and the paired difference, with no resampling.

    The month-by-month view is what makes a verdict checkable by hand: a median
    difference can hide a win that comes from two months, and this shows it.
    """
    from .aggregate import summary
    from .ranking import auc

    fn = lambda a: auc(a["label"], a["score"])          # noqa: E731
    per_a = summary(arrays_a, fn)["per_month"]
    per_b = summary(arrays_b, fn)["per_month"]
    out = {}
    for month in sorted(set(per_a) | set(per_b)):
        va, vb = per_a.get(month), per_b.get(month)
        usable = (va is not None and vb is not None
                  and not math.isnan(va) and not math.isnan(vb))
        out[str(month)] = {
            "a": None if va is None or math.isnan(va) else float(va),
            "b": None if vb is None or math.isnan(vb) else float(vb),
            "difference": (float(vb - va) if usable else None),
        }
    return out


def compare(*, path_a: str, path_b: str, manifest_path: str, canonical_path: str,
            task: str = "T1", unit: str = "month", resamples: int = 200,
            seed: int = 42, progress=None) -> dict:
    """Score two hand-in tables against one split and return the report."""
    from .aggregate import bootstrap_difference, to_arrays
    from .predictions import attach_truth, check_contract, load_predictions

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw_a, raw_b = load_predictions(path_a), load_predictions(path_b)
    for label, table in (("A", raw_a), ("B", raw_b)):
        problems = check_contract(table, manifest, canonical_path)
        if problems:
            raise SystemExit(f"arm {label}'s run fails the contract: {problems}")

    # Only the columns the metrics read; `job_id` is nine million strings.
    def _scored(table):
        table = table.select(["row_id", "test_month", "score"])
        return attach_truth(table, canonical_path, task)

    a, b, counts = align(_scored(raw_a), _scored(raw_b))
    arrays_a, arrays_b = to_arrays(a), to_arrays(b)
    names = {"a": {"run_id": raw_a["run_id"][0], "arm": raw_a["arm"][0]},
             "b": {"run_id": raw_b["run_id"][0], "arm": raw_b["arm"][0]}}
    del raw_a, raw_b, a, b

    difference = bootstrap_difference(arrays_a, arrays_b, unit=unit,
                                      resamples=resamples, seed=seed,
                                      progress=progress)
    months = per_month_difference(arrays_a, arrays_b)
    return {
        "schema": COMPARE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": manifest.get("dataset"),
        "task": task,
        "manifest_sha256": manifest.get("checksums", {}).get("manifest_sha256"),
        "arm_a": names["a"],
        "arm_b": names["b"],
        "rows": counts,
        "per_month_auc": months,
        "difference": difference,
        "verdict": {**verdict(difference), "rule": VERDICT_RULE},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.eval.compare",
        description="Paired comparison of two arms on one split (B-F5).")
    ap.add_argument("--a", required=True, help="arm A's hand-in table")
    ap.add_argument("--b", required=True, help="arm B's hand-in table")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--task", default="T1", choices=("T1", "T2"))
    ap.add_argument("--unit", default="month", choices=("month", "user"))
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)

    for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    try:
        os.sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):                  # pragma: no cover
        pass

    started = time.time()

    def _heartbeat(done: int, total: int) -> None:
        if done % 50 and done != total:
            return
        print(f"  draw {done}/{total} | {time.time() - started:,.0f}s elapsed",
              file=os.sys.stderr, flush=True)

    report = compare(path_a=args.a, path_b=args.b, manifest_path=args.manifest,
                     canonical_path=args.canonical, task=args.task,
                     unit=args.unit, resamples=args.resamples, seed=args.seed,
                     progress=_heartbeat)

    rows = report["rows"]
    print(f"compare {report['arm_a']['run_id']} vs {report['arm_b']['run_id']}")
    print(f"  rows common {rows['common']:,}"
          f" | A only {rows['a_only']:,} | B only {rows['b_only']:,}")
    for name, block in report["difference"]["metrics"].items():
        if block["ci_low"] is None:
            print(f"  {name}: no range ({block['n_draws']} usable draws)")
        else:
            print(f"  {name}: B - A {block['median']:+.4f} "
                  f"[{block['ci_low']:+.4f}, {block['ci_high']:+.4f}]")
    print(f"  verdict: {report['verdict']['statement']}")

    path = Path(args.out) / (f"compare-{report['arm_a']['run_id']}"
                             f"_vs_{report['arm_b']['run_id']}"
                             f"-{args.task.lower()}.json").replace("/", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
