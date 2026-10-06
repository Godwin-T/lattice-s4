"""
Calibration (metrics.md section 6).

Plain question: when the arm says "80% likely", does it happen about 80% of the
time? ECE is the average gap between the promise and reality — 0 is perfectly
honest.

Isotonic calibration is only ever fitted on validation months; the caller is
responsible for passing those, and must never pass the month being scored.
"""
from __future__ import annotations

import numpy as np

from .config import ECE_BINS


def ece(probabilities, labels, bins: int = ECE_BINS) -> float:
    """Expected calibration error over equal-width probability bins."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels).astype(float)
    keep = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[keep], y[keep]
    if len(p) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        total += (n / len(p)) * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(total)


def reliability_curve(probabilities, labels,
                      bins: int = ECE_BINS) -> list[dict]:
    """The per-bin numbers behind the ECE, for a reliability diagram."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels).astype(float)
    keep = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[keep], y[keep]
    edges = np.linspace(0.0, 1.0, bins + 1)

    curve = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        n = int(in_bin.sum())
        curve.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n": n,
            "mean_predicted": float(p[in_bin].mean()) if n else None,
            "observed_rate": float(y[in_bin].mean()) if n else None,
        })
    return curve


def fit_isotonic(probabilities, labels):
    """
    Fit a monotone probability map. Monotone means it can fix calibration but
    can never change the ranking, so AUC is identical before and after.
    """
    from sklearn.isotonic import IsotonicRegression

    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels).astype(float)
    keep = ~np.isnan(p)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(p[keep], y[keep])
    return calibrator
