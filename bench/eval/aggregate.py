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

**One draw, many metrics.** The confidence ranges for *all* the metrics come
from `bootstrap_cis`, which walks one shared sequence of draws and scores every
metric against each. That matters: resampling is the evaluator's dominant cost,
and a per-metric bootstrap both repeats the work and — because a metric that
re-sorts the drawn rows tie-breaks differently from one that does not — would
give the metrics slightly different samples to disagree about. The per-month,
median and pooled views in `summary` need no resampling at all.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import polars as pl

from . import bootstrap
from .config import ALPHA, BOOTSTRAP_RESAMPLES, K_VALUES

# Fewer draws than this and a percentile is noise, so no range is reported
# rather than a meaningless one (metrics.md §5.1).
MIN_DRAWS_FOR_CI = 20


def _encode(series: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Integer codes for a string column, and the values those codes stand for.

    `Series.to_numpy()` on a String column builds one Python `str` per row — on
    nine million rows that is roughly half a gigabyte *per column*, and it also
    makes `np.unique` compare strings nine million times. A Categorical moves the
    work to the handful of distinct values: the per-row array becomes four bytes
    of code, and `np.unique` then sorts integers. `codes[i]` is the position of
    row `i`'s value in the returned `values` table.
    """
    categorical = series.cast(pl.Categorical)
    return (categorical.to_physical().to_numpy(),
            categorical.cat.get_categories().to_numpy())


def to_arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """
    The columns every metric needs, as NumPy, once.

    `month` and `user` come back as integer codes with their value table beside
    them (`month_values`, `user_values`) rather than as strings; see `_encode`.
    `user` is the canonical `user_hash` where it is available, because `job_id`
    is not unique and grouping by it would merge unrelated jobs into one unit.

    `probability` is included only when the frame carries it: it is read by
    calibration alone, not by any ranking metric or by the bootstrap, so a frame
    without it stays exactly as it was.
    """
    month, month_values = _encode(frame["test_month"])
    if "user_hash" in frame.columns:
        user_source = frame["user_hash"]
    elif "job_id" in frame.columns:
        user_source = frame["job_id"]
    else:
        raise ValueError("frame has neither `user_hash` nor `job_id`, so no "
                         "user-level resampling unit can be formed")
    user, user_values = _encode(user_source)
    arrays = {
        "label": frame["label"].to_numpy().astype(bool),
        "score": frame["score"].to_numpy().astype(float),
        "energy": frame["energy_j"].to_numpy().astype(float),
        "month": month,
        "month_values": month_values,
        "user": user,
        "user_values": user_values,
    }
    if "probability" in frame.columns:
        arrays["probability"] = frame["probability"].to_numpy().astype(float)
    return arrays


# The `*_values` entries are the small lookup tables behind the integer-coded
# `month` and `user` columns (see `_encode`). They are indexed by *code*, not by
# row, so a row-wise `subset` must leave them alone.
VALUE_TABLES = ("month_values", "user_values")


def subset(arrays: dict[str, np.ndarray], index: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[index] for k, v in arrays.items() if k not in VALUE_TABLES}


def unit_ids(arrays: dict[str, np.ndarray], unit: str) -> tuple[np.ndarray, int]:
    """One integer unit-id per row, and how many units there are."""
    key = arrays["month"] if unit == "month" else arrays["user"]
    _values, codes = np.unique(key, return_inverse=True)
    return codes.reshape(-1).astype(np.int64), int(_values.size)


def _clean(values) -> list[float]:
    return [float(v) for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))]


def summary(arrays: dict[str, np.ndarray],
            metric_fn: Callable[[dict[str, np.ndarray]], float | None]) -> dict:
    """
    One metric in the per-month, median ± IQR and pooled views — no resampling.

    `metric_fn` takes the same array bundle and returns a float, or None when
    the metric is undefined (for example, energy on a dataset that has none).
    """
    month_key = arrays["month"]
    # `month` holds integer codes (see `_encode`); the result file is keyed by
    # the month string, so map each code back through the value table.
    values = arrays.get("month_values")
    per_month = {}
    for code in np.unique(month_key):
        label = values[code] if values is not None else code
        per_month[label] = metric_fn(
            subset(arrays, np.flatnonzero(month_key == code)))
    per_month = dict(sorted(per_month.items(), key=lambda item: item[0]))

    usable = _clean(per_month.values())
    if usable:
        median = float(np.median(usable))
        iqr = float(np.percentile(usable, 75) - np.percentile(usable, 25))
    else:
        median = iqr = None

    return {
        "per_month": per_month,
        "median": median,
        "iqr": iqr,
        "pooled": metric_fn(arrays),
        "n_months": len(usable),
    }


def _record(values: list[float], value) -> None:
    if value is None:
        return
    value = float(value)
    if not math.isnan(value):
        values.append(value)


def _range(values: list[float], alpha: float) -> tuple[float | None, float | None]:
    if len(values) < MIN_DRAWS_FOR_CI:
        return None, None
    return (float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))))


def bootstrap_cis(arrays: dict[str, np.ndarray], *,
                  unit: str = "month",
                  resamples: int = BOOTSTRAP_RESAMPLES,
                  alpha: float = ALPHA,
                  seed: int = 42,
                  ks=K_VALUES,
                  progress=None) -> dict:
    """
    Confidence ranges for every metric, from one shared set of draws.

    Returns the ranges nested to match the result file: `{"auc", "pr_auc",
    "energy_captured", "recall", "false_flag"}`, each a `(low, high)` pair, the
    last three keyed by operating point `k`.

    The draws are shared deliberately. Every metric sees the same resampled
    months (or users), so a difference between two metrics is a difference in
    the metrics, not in the samples they happened to draw.

    `progress`, if given, is called as `progress(done, total)` on every draw. On
    the full Eagle set this loop runs for tens of minutes with nothing to show
    for it, so a caller that wants a heartbeat can supply one; the default is
    silent and this module prints nothing itself.
    """
    gid, n_groups = unit_ids(arrays, unit)
    prep = bootstrap.Preparation(arrays["label"], arrays["score"], arrays["energy"], gid)
    rng = np.random.default_rng(seed)

    acc: dict[str, list[float]] = {"auc": [], "pr_auc": []}
    for prefix in ("energy_captured", "recall", "false_flag"):
        for k in ks:
            acc[f"{prefix}:{k}"] = []

    for done in range(1, resamples + 1):
        draw = prep.draw(bootstrap.draw_counts(n_groups, rng))
        _record(acc["auc"], draw.auc())
        _record(acc["pr_auc"], draw.average_precision())
        for k in ks:
            recall, false_flag, energy = draw.operating_point(k)
            _record(acc[f"energy_captured:{k}"], energy)
            _record(acc[f"recall:{k}"], recall)
            _record(acc[f"false_flag:{k}"], false_flag)
        if progress is not None:
            progress(done, resamples)

    return {
        "auc": _range(acc["auc"], alpha),
        "pr_auc": _range(acc["pr_auc"], alpha),
        "energy_captured": {k: _range(acc[f"energy_captured:{k}"], alpha) for k in ks},
        "recall": {k: _range(acc[f"recall:{k}"], alpha) for k in ks},
        "false_flag": {k: _range(acc[f"false_flag:{k}"], alpha) for k in ks},
    }


# The metrics a paired difference is reported on. AUC and PR-AUC are the
# headline comparison (arm-b-planning.md B-D3); the operating-point metrics are
# deliberately not here, because "B beats A" is a claim about ranking quality,
# and a difference of recall@5% would need its own rule stated in the document.
DIFFERENCE_METRICS = ("auc", "pr_auc")


def bootstrap_difference(arrays_a: dict[str, np.ndarray],
                         arrays_b: dict[str, np.ndarray], *,
                         unit: str = "month",
                         resamples: int = BOOTSTRAP_RESAMPLES,
                         alpha: float = ALPHA, seed: int = 42,
                         metrics=DIFFERENCE_METRICS,
                         progress=None) -> dict:
    """
    The **paired** confidence range of `B − A` (metrics.md §5, PRD's H1 rule).

    Paired, not two separate runs. Both arms are scored on the *same* draw of the
    same months, and the difference is taken within the draw. That is the whole
    point: arm-to-arm variation from a lucky resample cancels, so the range is
    about the two model's disagreement rather than about their shared dependence
    on a handful of months. Comparing two independent confidence ranges instead —
    "A's range and B's range overlap, so there is no difference" — is the
    standard error of the eye, and it fails *conservatively*: it hides real
    differences rather than inventing them.

    The two arms must have been scored on the same rows in the same order. That
    is checked, not assumed: a paired difference between two different
    populations is not defined, and silently intersecting them would hide that
    the arms answered different questions. `compare.py` aligns them first.

    Returns `{"unit", "resamples", "alpha", "n_units", "metrics": {name: ...}}`,
    where each metric carries the difference's median and mean over the draws,
    its percentile range, and `favours` — `"b"` or `"a"` when the range excludes
    zero, otherwise `None`.
    """
    if len(arrays_a["label"]) != len(arrays_b["label"]):
        raise ValueError(
            f"the two arms scored different numbers of rows "
            f"({len(arrays_a['label'])} vs {len(arrays_b['label'])}), so no "
            f"paired difference exists")
    gid_a, n_groups = unit_ids(arrays_a, unit)
    gid_b, n_groups_b = unit_ids(arrays_b, unit)
    if n_groups != n_groups_b or not np.array_equal(gid_a, gid_b):
        raise ValueError(
            "the two arms are not grouped into the same units in the same order, "
            "so their draws would not be paired")

    prep_a = bootstrap.Preparation(arrays_a["label"], arrays_a["score"],
                                   arrays_a["energy"], gid_a)
    prep_b = bootstrap.Preparation(arrays_b["label"], arrays_b["score"],
                                   arrays_b["energy"], gid_b)
    rng = np.random.default_rng(seed)
    acc: dict[str, list[float]] = {m: [] for m in metrics}

    for done in range(1, resamples + 1):
        counts = bootstrap.draw_counts(n_groups, rng)   # one draw, both arms
        draw_a, draw_b = prep_a.draw(counts), prep_b.draw(counts)
        for name in metrics:
            # `B - A`, in that order. The sign is the whole verdict — a caller
            # reading `favours == "b"` as "B is better" must not be silently
            # handed A's win — so it is written the way the rule reads.
            if name == "auc":
                value = draw_b.auc() - draw_a.auc()
            elif name == "pr_auc":
                value = draw_b.average_precision() - draw_a.average_precision()
            else:                                       # pragma: no cover
                raise ValueError(f"no paired difference defined for {name!r}")
            _record(acc[name], value)
        if progress is not None:
            progress(done, resamples)

    out: dict[str, dict] = {}
    for name in metrics:
        low, high = _range(acc[name], alpha)
        values = acc[name]
        out[name] = {
            "median": float(np.median(values)) if values else None,
            "mean": float(np.mean(values)) if values else None,
            "ci_low": low,
            "ci_high": high,
            "n_draws": len(values),
            # `None` is neither a win nor a loss: an undecided range must never
            # be read as "no difference" by a caller that only checks a boolean.
            "favours": ("b" if low is not None and low > 0.0
                        else "a" if high is not None and high < 0.0
                        else None),
        }
    return {"unit": unit, "resamples": resamples, "alpha": alpha,
            "n_units": n_groups, "metrics": out}
