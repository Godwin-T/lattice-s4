"""
Ranking metrics (metrics.md section 3).

Written by hand rather than pulled from a library so every definition here is
the one the document states, and so the toy examples in the tests can be
checked by arithmetic.

`score` is higher = riskier. `labels` is a boolean array: True = the job really
did have the outcome we care about.
"""
from __future__ import annotations

import numpy as np


def auc(labels, scores) -> float:
    """
    Rank-based AUC with ties averaged.

    Plain meaning: pick one positive and one negative at random; this is the
    chance the model ranked the positive as riskier. 0.5 = a coin flip.

    Vectorised. The ranks come from one `argsort` plus a `bincount` over the
    runs of equal scores, rather than a Python walk over every element. That
    walk was the evaluator's dominant cost: the bootstrap calls this hundreds of
    times on multi-million-row arrays, so a per-element loop made a full run take
    tens of minutes where `argsort` makes it seconds.
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0 or len(s) == 0:
        return float("nan")

    # Quicksort, not a stable sort: AUC depends only on the *runs* of equal
    # scores (their average rank), not on the order within a run, so stability
    # buys nothing here and costs ~30% of the sort. Verified identical to a
    # stable sort on 7M rows.
    order = np.argsort(s)
    ordered = s[order]

    # Average rank within each run of equal scores. A run covering sorted
    # positions i..j (0-based) shares rank 0.5 * (i + j) + 1.0 — the same value
    # the previous loop wrote, computed from a cumulative run id instead.
    n = len(ordered)
    positions = np.arange(1, n + 1, dtype=float)
    starts = np.empty(n, dtype=bool)
    starts[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=starts[1:])
    group = np.cumsum(starts) - 1
    counts = np.bincount(group)
    rank_sums = np.bincount(group, weights=positions)
    ranks = np.empty(n, dtype=float)
    ranks[order] = (rank_sums / counts)[group]

    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    """
    PR-AUC (average precision).

    Same "does it rank well" idea as AUC, but focused on the rare outcome. A
    model that says "nothing is risky" scores near the base rate here, where
    plain accuracy would look excellent.

    Vectorised. Recall rises only at a positive (by 1/n_pos), so the sum over
    every position collapses to a sum of `precision` at the positive positions —
    no per-element Python loop.
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    if len(s) == 0 or y.sum() == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    ordered = y[order]
    # precision at position p (1-based) is tp / p, since tp + fp = p.
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(precision[ordered].sum() / int(y.sum()))


def flag_top_k(scores, k: int) -> np.ndarray:
    """
    Boolean mask: is this job in the riskiest k% of the month?

    The cut is a percentile of *this month's* scores, so a month with an unusual
    score scale cannot distort the comparison between arms (metrics.md §3). The
    mask may cover slightly more than k% where scores tie.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return np.zeros(0, dtype=bool)
    return s >= np.percentile(s, 100 - k)


def recall_and_false_flag(labels, flagged) -> tuple[float, float]:
    """
    Of the jobs that really had the outcome, how many did we flag;
    of the jobs that did not, how many did we flag anyway.
    """
    y = np.asarray(labels).astype(bool)
    f = np.asarray(flagged).astype(bool)
    tp = int((f & y).sum())
    fp = int((f & ~y).sum())
    return (tp / max(int(y.sum()), 1), fp / max(int((~y).sum()), 1))
