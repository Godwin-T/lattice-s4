"""
Assemble one result file, and enforce the refusal rules (metrics.md §9, §10).

The refusal rules are the point of this module. A number that cannot be stood
behind is not published: an invalid control suppresses the headline, a metric
resting on too few months is dropped, and energy that does not exist is
reported as "not available" rather than as zero.
"""
from __future__ import annotations

import gc
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import aggregate, control as control_mod
from .config import (
    ECE_BINS, HEADLINE_K, K_VALUES, MIN_MONTHS_FOR_METRIC, POSITIVE_STATES,
)
from .energy import energy_captured
from .ranking import auc, average_precision, flag_top_k, recall_and_false_flag

RESULT_SCHEMA = "lattice24-benchmark/result/1"

# Why an energy metric has no number. Two causes must never be confused:
# "too few months" is a sampling accident, "no measured energy" is a property of
# the dataset. metrics.md §9 requires the second to read as "not available" —
# never as zero, and never as the first.
NO_ENERGY = ("not available: this dataset carries no measured energy "
             "(no positive row has an energy reading)")

# Why a diagnostic carries no confidence range. The shared bootstrap scores only
# the metrics baked into `bootstrap.Preparation`/`Draw` (auc, pr_auc, the five
# operating points); the baselines and ECE are diagnostics, and metrics.md asks
# for no range on them. The absence is written down rather than left as a pair of
# unexplained nulls for a reader to wonder about.
NO_BOOTSTRAP = ("no confidence range: this is a diagnostic, not a headline "
                "metric, and metrics.md asks for no range on it")

# Gate 2 (metrics.md §6.1): every trained arm must beat both naive baselines. The
# rule is stated in the result so a reader is not left to infer it — the two
# views are recorded, and the comparison is made on the median month.
GATE_2_RULE = ("the arm's median AUC and median PR-AUC must each exceed the "
               "baseline's (pooled values are recorded alongside, for context)")


def _block(arrays, fn, ci, *, resamples, unit):
    """One metric's three views plus the CI drawn by the shared bootstrap."""
    block = aggregate.summary(arrays, fn)
    block["ci_low"], block["ci_high"] = ci
    block["resamples"] = resamples
    block["unit"] = unit
    # Which view the range belongs to. It is the *pooled* figure: each draw
    # scores all its resampled rows in one pass, so the range is the spread of
    # the pooled metric under month resampling. The `median` and `iqr` beside it
    # describe the per-month distribution and are a different quantity — on a
    # skewed month set the median can even sit outside the range, which reads as
    # a bug unless this says otherwise.
    block["ci_on"] = "pooled"
    if block["n_months"] < MIN_MONTHS_FOR_METRIC:
        # Too few months to say anything; keep the count, drop the numbers.
        block.update({"median": None, "iqr": None, "pooled": None,
                      "ci_low": None, "ci_high": None, "ci_on": None,
                      "suppressed": "fewer than "
                                    f"{MIN_MONTHS_FOR_METRIC} usable months"})
    return block


def _diagnostic_block(arrays, fn) -> dict:
    """A metric's three views with no confidence range (see `NO_BOOTSTRAP`)."""
    block = aggregate.summary(arrays, fn)
    block.update({"ci_low": None, "ci_high": None, "ci_on": None,
                  "ci_note": NO_BOOTSTRAP})
    return block


def _score_baselines(arm_rows: pl.DataFrame, *, canonical_path: str,
                     manifest: dict, task: str) -> dict:
    """
    The two naive baselines (metrics.md §6.1), scored on the arm's own rows.

    Restricted to the rows the arm scored *deliberately*. A baseline asked about
    the whole test set answers an easier question than the arm did — it happily
    scores a user's first job, which the arm's history features cannot — and the
    arm's margin over it would be flattered by the difference. Same rows, same
    label, same months; only the score differs.

    Neither baseline gets a confidence range: they are the yardstick, not the
    headline (see `NO_BOOTSTRAP`).
    """
    from .baselines import longest_time_limits, repeat_last_outcome
    from .predictions import attach_truth

    blocks: dict[str, dict] = {}
    # Each baseline is built by a thunk, not by a function call in the tuple: a
    # tuple literal would materialise *both* frames before the loop began, and
    # each carries a `job_id` string per row — nine million of them. Built
    # lazily, one frame is released before the next is created.
    for name, build in (
            ("repeat_last_outcome",
             lambda: repeat_last_outcome(canonical_path, manifest,
                                         POSITIVE_STATES[task])),
            ("longest_time_limit",
             lambda: longest_time_limits(canonical_path, manifest))):
        scored = build()
        truth = attach_truth(scored.join(arm_rows, on="row_id", how="inner"),
                             canonical_path, task)
        del scored
        arrays = aggregate.to_arrays(truth)
        del truth
        blocks[name] = {
            "auc": _diagnostic_block(
                arrays, lambda a: auc(a["label"], a["score"])),
            "pr_auc": _diagnostic_block(
                arrays, lambda a: average_precision(a["label"], a["score"])),
            "rows_scored": int(len(arrays["label"])),
        }
        del arrays
    return blocks


def _calibration_block(arrays: dict) -> dict:
    """
    Expected calibration error and the curve behind it (metrics.md E5).

    Measured on the probabilities as handed in; isotonic calibration, if an arm
    chooses it, is fitted on validation months by the arm and never here. Reads
    `probability`, which `to_arrays` carries only when the frame has it — a frame
    without one says so rather than inventing an ECE of zero.
    """
    from .calibration import ece, reliability_curve

    if "probability" not in arrays:
        return {
            "available": False, "bins": ECE_BINS,
            "reason": ("the hand-in table carries no `probability` column, so "
                       "calibration cannot be measured"),
        }
    probabilities = arrays["probability"]
    block = _diagnostic_block(
        arrays, lambda a: ece(a["probability"], a["label"]))
    return {
        "available": True,
        "bins": ECE_BINS,
        "ece": block,
        "reliability_curve": reliability_curve(probabilities, arrays["label"]),
        "rows_with_probability": int((~np.isnan(probabilities)).sum()),
        "note": ("ECE of the probabilities as handed in. Isotonic calibration, "
                 "where an arm uses it, is fitted by the arm on validation "
                 "months only and is never fitted here."),
    }


def _gate_2(auc_view: dict, ap_view: dict, baselines: dict) -> dict:
    """
    Compare the arm's AUC and PR-AUC against each baseline (see `GATE_2_RULE`).

    The two views are the arm's own `auc` and `pr_auc` result blocks — not the
    per-fold AUCs the control gate reads, which are a different quantity.
    """
    comparisons = []
    for name, blocks in baselines.items():
        for metric, arm_block in (("auc", auc_view), ("pr_auc", ap_view)):
            base_block = blocks[metric]
            arm_median, base_median = arm_block["median"], base_block["median"]
            if arm_median is None or base_median is None:
                comparisons.append({
                    "baseline": name, "metric": metric, "beats": None,
                    "reason": "a median is suppressed, so the comparison cannot "
                              "be made"})
                continue
            comparisons.append({
                "baseline": name, "metric": metric,
                "arm_median": arm_median, "baseline_median": base_median,
                "arm_pooled": arm_block["pooled"],
                "baseline_pooled": base_block["pooled"],
                "margin": arm_median - base_median,
                "beats": bool(arm_median > base_median)})
    # A comparison that could not be made is not a pass; the gate certifies only
    # when every one of them was decided and every one was won.
    decided = [c for c in comparisons if c["beats"] is not None]
    return {
        "rule": GATE_2_RULE,
        "passed": bool(comparisons) and len(decided) == len(comparisons)
                  and all(c["beats"] for c in decided),
        "comparisons": comparisons,
    }


def evaluate(preds: pl.DataFrame, manifest: dict, canonical_path: str,
             *, task: str = "T1", resamples: int = 200, seed: int = 42,
             control_aucs=None, control_folds=None, arm_auc=None,
             metadata: dict | None = None,
             unit: str = "month", progress=None,
             with_baselines: bool = True, with_calibration: bool = True) -> dict:
    """
    Score one arm's predictions and return the result payload.

    `with_baselines` and `with_calibration` cost an extra pass over the test
    rows, so a quick re-score can turn them off; the result then carries `null`
    for that block rather than a block that looks computed. They default on
    because metrics.md §6.1 and E5 both ask for them on every arm.
    """
    from .predictions import attach_truth

    truth = attach_truth(preds, canonical_path, task,
                         with_probability=with_calibration)
    scored = truth.filter(pl.col("score").is_not_null())
    # The counts are taken as plain ints, then both frames are released before
    # the arrays are built. All three are derived from the same nine million
    # rows, and holding the joined frame, its filtered copy and the NumPy bundle
    # at once is what makes the evaluator heavier than the arm it scores.
    scored_rows, rows_with_a_score = truth.height, scored.height
    # The baselines are scored on exactly these rows, so the set has to be taken
    # before `scored` is released — and only when the baselines are wanted, since
    # it is another nine million integers to hold through the bootstrap.
    rows_for_baselines = scored.select(["row_id"]) if with_baselines else None
    del truth
    arrays = aggregate.to_arrays(scored)
    del scored
    gc.collect()

    # The baselines are computed first and only their small summary blocks are
    # kept: their frames are nine million rows each, and holding one while the
    # arm's arrays are also live is what the release discipline above exists to
    # avoid.
    baselines = None
    if rows_for_baselines is not None:
        baselines = _score_baselines(rows_for_baselines, canonical_path=canonical_path,
                                     manifest=manifest, task=task)
        del rows_for_baselines
        gc.collect()

    # Every metric's confidence range comes from one shared set of draws, so the
    # resampling is paid for once instead of once per metric (see aggregate.py).
    cis = aggregate.bootstrap_cis(arrays, unit=unit, resamples=resamples,
                                  seed=seed, ks=K_VALUES, progress=progress)

    auc_block = _block(arrays, lambda a: auc(a["label"], a["score"]),
                       cis["auc"], resamples=resamples, unit=unit)
    ap_block = _block(arrays, lambda a: average_precision(a["label"], a["score"]),
                      cis["pr_auc"], resamples=resamples, unit=unit)

    calibration = _calibration_block(arrays) if with_calibration else None

    energy_blocks: dict[str, dict] = {}
    recall_blocks: dict[str, dict] = {}
    false_flag_blocks: dict[str, dict] = {}
    for k in K_VALUES:
        energy_blocks[str(k)] = _block(
            arrays, lambda a, k=k: energy_captured(
                pl.DataFrame({"label": a["label"], "score": a["score"],
                              "energy_j": a["energy"]}), k)["energy_captured"],
            cis["energy_captured"][k], resamples=resamples, unit=unit)
        recall_blocks[str(k)] = _block(
            arrays, lambda a, k=k: recall_and_false_flag(
                a["label"], flag_top_k(a["score"], k))[0],
            cis["recall"][k], resamples=resamples, unit=unit)
        false_flag_blocks[str(k)] = _block(
            arrays, lambda a, k=k: recall_and_false_flag(
                a["label"], flag_top_k(a["score"], k))[1],
            cis["false_flag"][k], resamples=resamples, unit=unit)

    # `control_folds` and `arm_auc` let the gate ask its second question — did
    # the arm beat its own shuffle on the folds the control actually ran on.
    # Both come from the arm's metadata; without them the gate cannot certify
    # the run and says so rather than passing it.
    control = control_mod.gate(control_aucs, arm_auc, control_folds)

    # Energy coverage, so no energy figure can be read without it.
    positives = arrays["label"]
    energy_known = ~np.isnan(arrays["energy"])
    coverage = (float((positives & energy_known).sum() / positives.sum())
                if positives.sum() else None)
    contributing_months = 0
    if energy_blocks[str(HEADLINE_K)]["per_month"]:
        contributing_months = sum(
            1 for v in energy_blocks[str(HEADLINE_K)]["per_month"].values()
            if v is not None)

    # A dataset with no measured energy produces zero coverage on every month,
    # which `_block` would report as "fewer than 3 usable months" — true of the
    # symptom, false as the reason, and it reads as a sampling accident rather
    # than a property of the data. Replace it with the dataset-level cause so a
    # reader is never told the months were too few when they were never possible.
    energy_available = bool(coverage)
    if not energy_available:
        for block in energy_blocks.values():
            block.update({"median": None, "iqr": None, "pooled": None,
                          "ci_low": None, "ci_high": None, "ci_on": None,
                          "suppressed": NO_ENERGY})

    result = {
        "schema": RESULT_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": preds["run_id"][0] if preds.height else None,
        "arm": preds["arm"][0] if preds.height else None,
        "task": task,
        "dataset": manifest.get("dataset"),
        "manifest_sha256": manifest.get("checksums", {}).get("manifest_sha256"),
        "positive_states": sorted(POSITIVE_STATES[task]),
        "counts": {
            "scored_rows": scored_rows,
            "rows_with_a_score": rows_with_a_score,
            "unscorable_rows": scored_rows - rows_with_a_score,
            "positives": int(positives.sum()),
            "energy_coverage_over_positives": coverage,
            "months_contributing_to_energy": contributing_months,
        },
        "valid": control["passed"],
        "invalid_reason": None if control["passed"] else control["reason"],
        # The run may be valid and still have no headline: a valid run on a
        # dataset with no measured energy reports "not available", not nulls.
        "energy_available": energy_available,
        "metrics": {
            "auc": auc_block,
            "pr_auc": ap_block,
            "energy_captured": energy_blocks,
            "recall": recall_blocks,
            "false_flag": false_flag_blocks,
        },
        "headline": None,
        # Gate 2 (metrics.md §6.1) and calibration (E5). `gate_2` and
        # `calibration` are null only when the caller turned them off; the
        # baselines themselves are null on the same switch.
        "gate_2": _gate_2(auc_block, ap_block, baselines) if baselines else None,
        "baselines": baselines,
        "calibration": calibration,
        "control": control,
        "system": (metadata or {}).get("system") or {
            "runs_offline": metadata.get("runs_offline") if metadata else None,
            "data_leaves_site": metadata.get("data_leaves_site") if metadata else None,
            "fit_seconds_total": metadata.get("fit_seconds_total") if metadata else None,
            "inference_seconds_per_1m": (metadata.get("inference_seconds_per_1m")
                                         if metadata else None),
            "cost_usd_per_1m": metadata.get("cost_usd_per_1m") if metadata else None,
        },
    }

    if control["passed"]:
        if energy_available:
            headline = energy_blocks[str(HEADLINE_K)]
            result["headline"] = {
                "metric": f"energy_captured@{HEADLINE_K}%",
                "available": True,
                "median": headline["median"],
                "iqr": headline["iqr"],
                "pooled": headline["pooled"],
                "ci_low": headline["ci_low"],
                "ci_high": headline["ci_high"],
                "ci_on": headline["ci_on"],
                "n_months": headline["n_months"],
                "coverage_over_positives": coverage,
            }
        else:
            # Explicitly not available, rather than a block of nulls that a
            # reader has to interpret. The run stays valid — it is the energy
            # column, not the arm, that is missing.
            result["headline"] = {
                "metric": f"energy_captured@{HEADLINE_K}%",
                "available": False,
                "reason": NO_ENERGY,
            }
    return result


def write(result: dict, outdir: str, run_id: str | None = None) -> str:
    """Write `results/<run_id>.json` and return the path."""
    from pathlib import Path
    import json

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    name = (run_id or result.get("run_id") or "run").replace("/", "_")
    path = outdir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return str(path)
