"""
Weighted resampling for the bootstrap (metrics.md §3 and §5).

**Why this exists.** The CI resamples whole units — months for the full set,
users for the 20,000-job sample — and recomputes a metric on each draw. The
obvious implementation concatenates the drawn rows and recomputes from scratch.
That is what this replaced, and it was untenable: every draw re-sorted seven
million rows *once per metric*, so a full-set run at the documented 1,000
resamples was a five-hour job.

**The trick.** A draw only changes *how many times each row appears*, never the
rows' scores. So:

* sort by score **once**, outside the draw loop;
* represent a draw by a **weight per row** — how many times its unit was drawn;
* compute each metric in one pass over the pre-sorted rows, using the weights.

`weight w` on a row means exactly `the row appears w times`, so the result is
identical to the materialised draw, not an approximation. `tests/test_eval.py`
checks weighted against materialised, and the tie rule below is what makes them
agree rather than merely be close.

**Tie rule.** Rows with equal scores are ordered by their position in the input,
and a row's replicated copies stay together. That is deterministic and — unlike
the materialised version, whose tie order depended on which units a draw
happened to pick — does not vary with the draw. Two metrics need it: average
precision (below), and the percentile behind `flag_top_k`, where equal values
make the interpolated cut unambiguous either way.

**How the pieces fit.** The functions at the bottom of this module (`auc`,
`average_precision`, ...) are the metric definitions, each taking a raw array
bundle plus a `weights` vector and an `order`. `Preparation`/`Draw` wrap the same
definitions for the hot path: `Preparation` orders the fixed arrays **once**,
and each `Draw` gathers weights into that order and shares the cumulative sums
between metrics, so a run of many draws never re-sorts or re-gathers per metric.
`aggregate.bootstrap_cis` drives them; the plain functions exist so a test can
check one metric against a materialised draw in isolation.

Everything here works on plain NumPy arrays; the caller supplies them once.
"""
from __future__ import annotations

import numpy as np

# `H[k] = 1 + 1/2 + ... + 1/k`, with `H[0] = 0`. Average precision needs
# `H[b + w] - H[b]` (see `average_precision`); building it once and growing it on
# demand costs one cumsum, versus a per-row Python loop.
_H = np.zeros(1)


def _harmonic(n: int) -> np.ndarray:
    global _H
    if n >= _H.size:
        size = max(n + 1, 2 * _H.size, 1024)
        table = np.empty(size, dtype=np.float64)
        table[0] = 0.0
        table[1:] = np.cumsum(1.0 / np.arange(1, size, dtype=np.float64))
        _H = table
    return _H


def draw_counts(n_groups: int, rng) -> np.ndarray:
    """
    How many times each unit is drawn, for one bootstrap resample.

    Draws `n_groups` units with replacement — exactly the resampling the
    materialised version did — and leaves the fan-out to rows to `counts[gid]`.
    """
    return np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)


def order_by_score(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The two orders every metric needs, computed once per run of draws.

    * ascending — ranks for AUC, and the cut for `flag_top_k`;
    * descending, stable — average precision, where ties stay in input order.
    """
    return np.argsort(scores), np.argsort(-scores, kind="stable")


# ---------------------------------------------------------------------------
# The metric definitions. Each `_*_ordered` takes arrays already in the order it
# needs (weights included) so the hot path pays for ordering once; the public
# wrappers at the end apply the order for the one-shot callers.
# ---------------------------------------------------------------------------

def _auc_ordered(y, ss, w, cum) -> float:
    before = cum - w
    n_pos = float(np.dot(w, y))
    n_neg = (float(cum[-1]) if cum.size else 0.0) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")

    # Average the midrank across each run of equal scores. The midrank is
    # constant within a run, so the sum of positive rank is a sum over *runs*:
    # `reduceat` the positive copies per run and pair them with the run's
    # midrank, rather than expanding a midrank back to every row.
    starts = np.empty(ss.size, dtype=bool)
    starts[0] = True
    np.not_equal(ss[1:], ss[:-1], out=starts[1:])
    first = np.flatnonzero(starts)
    run_start = np.minimum.reduceat(before, first)
    run_end = np.maximum.reduceat(cum, first)
    midrank = 0.5 * (run_start + 1.0 + run_end)
    positive_run = np.add.reduceat(w * y, first)

    rank_sum = float(np.dot(positive_run, midrank))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision_ordered(y, w) -> float:
    n_pos = float(np.dot(w, y))
    if n_pos == 0.0:
        return float("nan")

    cum = np.cumsum(w)
    before = cum - w
    pos_before = np.cumsum(w * y) - w * y

    h = _harmonic(int(cum[-1]) if cum.size else 0)
    # `contribution` is already the sum over a row's `w` copies, so the positives
    # are selected by `y` alone -- multiplying by `w` here would count each of a
    # row's copies once per copy. Only positive rows contribute; the rest are
    # multiplied by zero.
    contribution = w + (pos_before - before) * (h[before + w] - h[before])
    return float(np.dot(y, contribution) / n_pos)


def _percentile_ordered(ss, cum, q) -> float:
    total = float(cum[-1]) if cum.size else 0.0
    if total == 0.0:
        return float("nan")

    # NumPy's default linear interpolation: the value at fractional unit
    # position `q/100 * (N - 1)`, where `N` is the total weight.
    position = (q / 100.0) * (total - 1.0)
    lo = int(np.floor(position))
    hi = int(np.ceil(position))
    value_lo = ss[np.searchsorted(cum, lo + 1, side="left")]
    value_hi = ss[np.searchsorted(cum, hi + 1, side="left")]
    return float(value_lo + (value_hi - value_lo) * (position - lo))


def _energy_ordered(energy, pos_known, w, mask) -> float | None:
    """
    Share of the positive rows' energy sitting in the flagged rows.

    `energy` must have its unknowns already zeroed (see `clean_energy`): the
    mask below cannot zero a NaN by itself, because `NaN * 0` is still NaN and
    `np.dot` would carry it into the total.
    """
    total = float(np.dot(w * energy, pos_known))
    if total <= 0.0:
        return None
    caught = float(np.dot(w * energy, pos_known & mask))
    return caught / total


def clean_energy(energy) -> np.ndarray:
    """Energy as float, with unknown values replaced by 0 so they add nothing."""
    return np.nan_to_num(np.asarray(energy, dtype=float), nan=0.0)


def _recall_false_ordered(y, w, mask, total: float) -> tuple[float, float]:
    n_pos = float(np.dot(w, y))
    n_neg = total - n_pos
    tp = float(np.dot(w, y & mask))
    fp = float(np.dot(w, (~y) & mask))
    return tp / max(n_pos, 1.0), fp / max(n_neg, 1.0)


class Preparation:
    """
    A draw-independent canvas: the fixed arrays in the two orders metrics use,
    plus each row's unit id, all computed once.

    Holding this across a run of draws is what makes the bootstrap cheap — a
    draw then costs one gather per order, not a sort per metric.
    """

    def __init__(self, labels, scores, energy, gid):
        labels = np.asarray(labels).astype(bool)
        scores = np.asarray(scores, dtype=float)
        energy = np.asarray(energy, dtype=float)
        self.asc, self.desc = order_by_score(scores)

        self.score_asc = scores[self.asc]
        self.label_asc = labels[self.asc]
        # Zeroed unknowns, so a missing-energy row adds nothing to either side
        # of the ratio without a NaN leaking through `np.dot`.
        self.energy_asc = clean_energy(energy)[self.asc]
        # "Positive and its energy is known" — the population energy metrics
        # range over. A positive with unknown energy is excluded, not scored 0.
        self.pos_known_asc = self.label_asc & ~np.isnan(energy[self.asc])
        self.label_desc = labels[self.desc]

        self.gid_asc = gid[self.asc]
        self.gid_desc = gid[self.desc]

    def draw(self, counts: np.ndarray) -> "Draw":
        return Draw(self, counts)


class Draw:
    """
    One bootstrap draw: per-row weights, and the prefix sums every metric
    shares.

    The scores are held in ascending order, so the "flag the riskiest `k%`" cut
    is a **suffix** of that order — everything from some index `j` to the end.
    That is what makes the operating points cheap: once the prefix sums of the
    weights, the positive copies and the positive energy are known (three
    cumsums, built once and reused by every `k`), each cut is a binary search
    and a subtraction, not another pass over seven million rows.
    """

    def __init__(self, prep: Preparation, counts: np.ndarray):
        self._p = prep
        self.w_asc = counts[prep.gid_asc]
        self.w_desc = counts[prep.gid_desc]
        self._cum = self._cw_pos = self._cw_energy = None

    @property
    def cum(self) -> np.ndarray:
        if self._cum is None:
            self._cum = np.cumsum(self.w_asc)
        return self._cum

    @property
    def total(self) -> float:
        return float(self.cum[-1]) if self.cum.size else 0.0

    @property
    def n_pos(self) -> float:
        self._prefixes()
        return float(self._cw_pos[-1]) if self._cw_pos.size else 0.0

    def _prefixes(self) -> None:
        if self._cw_pos is None:
            p = self._p
            self._cw_pos = np.cumsum(self.w_asc * p.label_asc)
            self._cw_energy = np.cumsum(self.w_asc * p.energy_asc * p.pos_known_asc)

    def auc(self) -> float:
        return _auc_ordered(self._p.label_asc, self._p.score_asc, self.w_asc, self.cum)

    def average_precision(self) -> float:
        return _average_precision_ordered(self._p.label_desc, self.w_desc)

    def operating_point(self, k: int) -> tuple[float, float, float | None]:
        """Recall, false-flag rate and energy captured at the riskiest `k%`."""
        self._prefixes()
        p = self._p
        cum = self.cum
        total = self.total
        if total == 0.0:
            return float("nan"), float("nan"), None

        cut = _percentile_ordered(p.score_asc, cum, 100.0 - k)
        # First row at or above the cut — the suffix the flag covers.
        j = int(np.searchsorted(p.score_asc, cut, side="left"))
        below = (float(cum[j - 1]) if j else 0.0)

        n_pos = self.n_pos
        n_neg = total - n_pos
        flagged = total - below
        tp = n_pos - (float(self._cw_pos[j - 1]) if j else 0.0)
        recall = tp / max(n_pos, 1.0)
        false_flag = (flagged - tp) / max(n_neg, 1.0)

        energy_total = float(self._cw_energy[-1]) if self._cw_energy.size else 0.0
        if energy_total <= 0.0:
            energy = None
        else:
            caught = energy_total - (float(self._cw_energy[j - 1]) if j else 0.0)
            energy = caught / energy_total
        return recall, false_flag, energy


# ---------------------------------------------------------------------------
# One-shot wrappers: same definitions, applied to a whole array bundle. Used by
# the tests to check the weighted result against a materialised draw.
# ---------------------------------------------------------------------------

def auc(labels, scores, weights, order) -> float:
    """Rank-based AUC with ties averaged (metrics.md §3), under row weights.

    The weighted midrank of a row is `(units_before + units_through + 1) / 2`,
    which is the average rank its `w` replicated copies would hold. Summing
    `w * midrank` over the positives is then exactly the sum of ranks of the
    positive copies, so the Mann-Whitney form of AUC applies unchanged.
    """
    w = weights[order]
    return _auc_ordered(labels[order], scores[order], w, np.cumsum(w))


def average_precision(labels, scores, weights, order) -> float:
    """PR-AUC / average precision (metrics.md §3), under row weights.

    Average precision sums, over the positive rows, the precision at the moment
    each is reached. A row with weight `w` is `w` copies at consecutive
    positions, so its copies contribute `sum_{j=1..w} (a + j) / (b + j)` where
    `b` is the copies before it and `a` the positives before it. That sum
    telescopes to `w + (a - b) * (H[b + w] - H[b])` with `H` the harmonic
    numbers, so the whole thing is one pass and no per-row loop.
    """
    return _average_precision_ordered(labels[order], weights[order])


def percentile(scores, weights, q, order) -> float:
    """`np.percentile(scores, q)` of the materialised draw, without materialising."""
    w = weights[order]
    return _percentile_ordered(scores[order], np.cumsum(w), q)


def energy_captured(labels, energy, weights, flagged) -> float | None:
    """Share of the positive rows' energy sitting in the flagged rows."""
    known = ~np.isnan(np.asarray(energy, dtype=float))
    return _energy_ordered(clean_energy(energy),
                           np.asarray(labels).astype(bool) & known,
                           weights, flagged)


def recall_and_false_flag(labels, weights, flagged) -> tuple[float, float]:
    """Weighted recall of the positives, and the false-flag rate on the negatives."""
    return _recall_false_ordered(np.asarray(labels).astype(bool), weights,
                                 flagged, float(weights.sum()))
