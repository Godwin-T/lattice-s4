"""
The label-shuffle control gate (metrics.md section 6.2, decision M1).

The arm shuffles its *training* labels, refits, and reports the AUC of each
shuffled fit. With nothing real to learn, those AUCs should sit near 0.5. If
they do not, information is leaking and the whole run is invalid — no headline
figure is published.

The gate is the spec's fixed band. The published tool's wider tolerance (which
accounts for small samples) is computed too, purely so a borderline result is
visible rather than hidden.
"""
from __future__ import annotations

import math

import numpy as np

from .config import CONTROL_BAND, CONTROL_TOLERANCE_FLOOR


def gate(control_aucs, band: tuple[float, float] = CONTROL_BAND,
         tolerance_floor: float = CONTROL_TOLERANCE_FLOOR) -> dict:
    """Judge the shuffled fits and say whether the run may be reported."""
    values = np.asarray([a for a in (control_aucs or [])
                         if a is not None and not math.isnan(a)], dtype=float)
    if len(values) == 0:
        return {"n": 0, "mean": None, "passed": False, "band": list(band),
                "tolerance": None, "min": None, "max": None,
                "reason": "no usable shuffled fits"}

    mean = float(values.mean())
    sem = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
    tolerance = max(2.0 * sem, tolerance_floor) if not math.isnan(sem) else tolerance_floor
    passed = band[0] <= mean <= band[1]

    return {
        "n": int(len(values)),
        "mean": mean,
        "min": float(values.min()),
        "max": float(values.max()),
        "band": list(band),
        "tolerance": float(tolerance),
        "passed": bool(passed),
        "reason": None if passed else (
            f"mean AUC {mean:.3f} is outside the {band[0]}-{band[1]} band"),
    }
