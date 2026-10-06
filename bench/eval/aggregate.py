"""
Aggregation and uncertainty (metrics.md §3 and §5).

Every metric is reported three ways, because they answer different questions:

* **per month** — the raw unit, so instability is visible;
* **median ± IQR** — the typical month, robust to one freak month;
* **pooled** — all test months together, comparable to published numbers.

Confidence ranges come from resampling: **months** for the full set, **users**
for the 20,000-job sample (jobs within a user move together, so resampling jobs
alone would make the range look narrower than it is).

Everything works on plain NumPy arrays rather than frames. That is a cost
decision as much as a style one: a bootstrap does thousands of resamples, and
re-slicing a nine-million-row frame each time would make the evaluator far
heavier than the arm it is scoring.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import polars as pl

from .config import ALPHA, BOOTSTRAP_RESAMPLES


def to_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """The columns every metric needs, as NumPy, once."""
    return {
        "label": frame["label"].to_numpy().astype(bool),
        "score": frame["score"].to_numpy().astype(float),
        "energy": frame["energy_j"].to_numpy().astype(float),
        "month": frame["test_month"].to_numpy(),
        "user": (frame["user_hash"].to_numpy() if "user_hash" in frame.columns
                 else frame["job_id"].to_numpy()),
    }


def subset(arrays: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[index] for k, v in arrays.items()}


def _group_indices(arrays: dict[str, np.ndarray], unit: str) -> list[np.ndarray]:
    key = arrays["month"] if unit == "month" else arrays["user"]
    order = np.argsort(key, kind="mergesort")
    sorted_key = key[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
    return np.split(order, boundaries[1:])


def _clean(values) -> list[float]:
    return [float(v) for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))]


def views(arrays: dict[str, np.ndarray],
          metric_fn: Callable[[dict[str, np.ndarray]], float | None],
          *, unit: str = "month",
          resamples: int = BOOTSTRAP_RESAMPLES,
          alpha: float = ALPHA,
          seed: int = 42) -> dict:
    """
    Compute one metric in all three views, with a bootstrap confidence range.

    `metric_fn` takes the same array bundle and returns a float, or None when
    the metric is undefined (for example, energy on a dataset that has none).
    """
    month_key = arrays["month"]
    months = sorted(set(month_key.tolist()))
    per_month: dict[str, float | None] = {}
    for month in months:
        per_month[month] = metric_fn(subset(arrays, np.flatnonzero(month_key == month)))

    usable = _clean(per_month.values())
    if usable:
        median = float(np.median(usable))
        iqr = float(np.percentile(usable, 75) - np.percentile(usable, 25))
    else:
        median = iqr = None

    pooled = metric_fn(arrays)

    ci_low = ci_high = None
    if usable:
        groups = _group_indices(arrays, unit)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(resamples):
            picked = rng.integers(0, len(groups), len(groups))
            index = np.concatenate([groups[i] for i in picked])
            value = metric_fn(subset(arrays, index))
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                draws.append(float(value))
        if len(draws) >= 20:
            ci_low = float(np.percentile(draws, 100 * alpha / 2))
            ci_high = float(np.percentile(draws, 100 * (1 - alpha / 2)))

    return {
        "per_month": per_month,
        "median": median,
        "iqr": iqr,
        "pooled": pooled,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_months": len(usable),
        "resamples": resamples,
        "unit": unit,
    }
