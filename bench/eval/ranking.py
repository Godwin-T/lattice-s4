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
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0 or len(s) == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ordered = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average rank for ties
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    """
    PR-AUC (average precision).

    Same "does it rank well" idea as AUC, but focused on the rare outcome. A
    model that says "nothing is risky" scores near the base rate here, where
    plain accuracy would look excellent.
    """
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=float)
    if len(s) == 0 or y.sum() == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / int(y.sum())

    ap, previous_recall = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - previous_recall)
        previous_recall = r
    return float(ap)


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
