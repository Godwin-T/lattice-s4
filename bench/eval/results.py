"""
Assemble one result file, and enforce the refusal rules (metrics.md §9, §10).

The refusal rules are the point of this module. A number that cannot be stood
behind is not published: an invalid control suppresses the headline, a metric
resting on too few months is dropped, and energy that does not exist is
reported as "not available" rather than as zero.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import aggregate, control as control_mod
from .config import (
    HEADLINE_K, K_VALUES, MIN_MONTHS_FOR_METRIC, POSITIVE_STATES,
)
from .energy import energy_captured
from .ranking import auc, average_precision, flag_top_k, recall_and_false_flag

RESULT_SCHEMA = "lattice24-benchmark/result/1"


def _metric(arrays, fn, *, resamples, seed, unit):
    block = aggregate.views(arrays, fn, unit=unit, resamples=resamples, seed=seed)
    if block["n_months"] < MIN_MONTHS_FOR_METRIC:
        # Too few months to say anything; keep the count, drop the numbers.
        block.update({"median": None, "iqr": None, "pooled": None,
                      "ci_low": None, "ci_high": None,
                      "suppressed": "fewer than "
                                    f"{MIN_MONTHS_FOR_METRIC} usable months"})
    return block


def evaluate(preds: pl.DataFrame, manifest: dict, canonical_path: str,
             *, task: str = "T1", resamples: int = 200, seed: int = 42,
             control_aucs=None, metadata: dict | None = None,
             unit: str = "month") -> dict:
    """Score one arm's predictions and return the result payload."""
    from .predictions import attach_truth

    truth = attach_truth(preds, canonical_path, task)
    scored = truth.filter(pl.col("score").is_not_null())
    arrays = aggregate.to_arrays(scored)

    auc_block = _metric(arrays, lambda a: auc(a["label"], a["score"]),
                        resamples=resamples, seed=seed, unit=unit)
    ap_block = _metric(arrays, lambda a: average_precision(a["label"], a["score"]),
                       resamples=resamples, seed=seed, unit=unit)

    energy_blocks: dict[str, dict] = {}
    recall_blocks: dict[str, dict] = {}
    false_flag_blocks: dict[str, dict] = {}
    for k in K_VALUES:
        energy_blocks[str(k)] = _metric(
            arrays, lambda a, k=k: energy_captured(
                pl.DataFrame({"label": a["label"], "score": a["score"],
                              "energy_j": a["energy"]}), k)["energy_captured"],
            resamples=resamples, seed=seed, unit=unit)
        recall_blocks[str(k)] = _metric(
            arrays, lambda a, k=k: recall_and_false_flag(
                a["label"], flag_top_k(a["score"], k))[0],
            resamples=resamples, seed=seed, unit=unit)
        false_flag_blocks[str(k)] = _metric(
            arrays, lambda a, k=k: recall_and_false_flag(
                a["label"], flag_top_k(a["score"], k))[1],
            resamples=resamples, seed=seed, unit=unit)

    control = control_mod.gate(control_aucs)

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
            "scored_rows": truth.height,
            "rows_with_a_score": scored.height,
            "unscorable_rows": truth.height - scored.height,
            "positives": int(positives.sum()),
            "energy_coverage_over_positives": coverage,
            "months_contributing_to_energy": contributing_months,
        },
        "valid": control["passed"],
        "invalid_reason": None if control["passed"] else control["reason"],
        "metrics": {
            "auc": auc_block,
            "pr_auc": ap_block,
            "energy_captured": energy_blocks,
            "recall": recall_blocks,
            "false_flag": false_flag_blocks,
        },
        "headline": None,
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
        headline = energy_blocks[str(HEADLINE_K)]
        result["headline"] = {
            "metric": f"energy_captured@{HEADLINE_K}%",
            "median": headline["median"],
            "iqr": headline["iqr"],
            "pooled": headline["pooled"],
            "ci_low": headline["ci_low"],
            "ci_high": headline["ci_high"],
            "n_months": headline["n_months"],
            "coverage_over_positives": coverage,
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
