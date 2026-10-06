"""
Energy metrics (metrics.md section 5).

The headline: of all the energy sitting in jobs that really had the outcome,
how much did our flagged jobs hold?

Two rules that keep it honest:

* only jobs that actually carry a measured reading count — we never impute, and
  never treat missing energy as zero;
* coverage is reported alongside, so "70% captured" cannot be read as a strong
  claim when it was computed over only half the energy.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .ranking import flag_top_k


def energy_captured(frame: pl.DataFrame, k: int,
                    label_col: str = "label",
                    energy_col: str = "energy_j") -> dict:
    """
    The headline metric at a k% flag rate, plus everything needed to read it.

    Returns `energy_captured = None` when the frame carries no usable energy,
    which must be reported as "not available", never as zero.
    """
    labels = frame[label_col].to_numpy().astype(bool)
    scores = frame["score"].to_numpy().astype(float)
    energy = frame[energy_col].to_numpy().astype(float)
    known = ~np.isnan(energy)

    positives = labels & known
    n_positives = int(labels.sum())
    n_positives_with_energy = int(positives.sum())
    total_positive_energy = float(energy[positives].sum())

    result = {
        "k": k,
        "n_positives": n_positives,
        "n_positives_with_energy": n_positives_with_energy,
        "coverage_pos": (n_positives_with_energy / n_positives) if n_positives else None,
        "total_positive_energy_j": total_positive_energy,
        "energy_captured": None,
        "energy_captured_j": None,
    }
    if total_positive_energy <= 0 or len(scores) == 0:
        return result

    flagged = flag_top_k(scores, k)
    caught = float(energy[flagged & positives].sum())
    result["energy_captured_j"] = caught
    result["energy_captured"] = caught / total_positive_energy
    return result
