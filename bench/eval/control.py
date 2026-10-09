"""
The label-shuffle control gate (metrics.md section 6.2, decision M1).

An arm scrambles its *training* labels, refits, and reports the AUC of each
shuffled fit. With nothing real to learn those fits must score at chance. If
they do not, information is reaching the features from the labels, and the
whole run is invalid: no headline figure is published.

The gate asks two questions, and both are answered from the arm's *own*
measurements rather than from a fixed number.

**1. Is the null at chance?** The shuffled fits must sit within
`max(2 * SEM, 0.02)` of 0.5, where SEM is the standard error of their own mean.

The width has to come from the draws, because the shuffled AUC of a
four-feature model is not tightly concentrated near 0.5 at all. With nothing to
learn the fit settles on a random direction in the four-dimensional feature
space, and a random direction is not orthogonal to the real signal — so a
*single* shuffled draw scatters with sd ~0.2, from 0.25 to 0.75 (measured on
both datasets; `control_gate_finding.md` sections 2-3, checked against a null of
literal random directions that makes no claim about models at all).

That makes the spec's fixed 0.45-0.55 band about **one standard error wide** for
a 15-draw mean, so it condemns a completely clean arm 30-40% of the time
(section 4) — which is how Kestrel's two runs came to be certified at 0.5012 and
0.5117 while Eagle's clean run was stopped at 0.5739 (section 5). The band is
still computed and reported as `spec_band` so the change is visible in every
result file, but it no longer decides.

The width is `sd / sqrt(n)` over the draws, which treats them as independent
when they are not: the shuffles within a fold share a training set and a test
month, so their AUCs are positively correlated and the true standard error is
**larger** than this one. That is the safe direction to be wrong in — the
tolerance used is the tightest the data can justify, so a run that clears it
would clear any honest widening of it. The alternative (estimating the width
from the spread of the per-fold means) is not usable here: with five folds that
spread has four degrees of freedom, and its 95% interval spans a factor of five,
which is exactly the uncertainty the old fixed band pretended not to have.

**2. Did the arm beat its own shuffle?** On the folds the control ran, the arm's
own AUC must exceed the shuffled mean by at least `separation_sem` standard
errors of the *paired* difference. This is the question the data can answer,
and it is the one that catches a leak: a pipeline whose features carry the
label scores high when the labels are scrambled too, so the gap between the arm
and its null collapses. Pairing per fold cancels fold difficulty (early months
are harder and have fewer users), which is what makes the test sharp.

This check needs the arm's per-fold AUCs on the folds the control actually used
(`arm_auc` keyed by fold id, plus `folds` giving each draw's fold). Arm A
records both. When they are absent the check is reported as **not evaluated**
rather than silently passed — a reader can then see that only question 1 was
answered.
"""
from __future__ import annotations

import math

import numpy as np

from .config import (
    CONTROL_BAND, CONTROL_SEPARATION_SEM, CONTROL_TOLERANCE_FLOOR,
)


def _usable(control_aucs, folds):
    """
    Pair every finite draw with its fold id, dropping the rest.

    `folds` may be shorter than `control_aucs` (older metadata recorded no fold
    ids); a draw past its end simply carries `None`.
    """
    draws = []
    for i, value in enumerate(control_aucs or []):
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        fold = folds[i] if folds is not None and i < len(folds) else None
        draws.append((value, fold))
    return draws


def _separation(draws, arm_auc, minimum, floor):
    """
    The arm's AUC against its own null, paired fold by fold.

    For each fold the control ran on, the gap is `arm AUC - mean shuffled AUC`.
    The pooled gap must clear `minimum` standard errors of its own mean.

    Pairing is what removes the fold-to-fold difficulty that otherwise swamps
    the comparison: the shuffled null is centred at 0.5 on *every* fold, but
    early months score lower for both the arm and the null, and pairing cancels
    that.

    The gap also has to clear an absolute `floor`. Without it the test is
    vacuous in the degenerate case: if every fold shows the same gap, the gaps
    have no spread, the standard error is zero, and an arm that is exactly as
    good as its own shuffle would be certified on `0 >= 0`.
    """
    if not arm_auc:
        reason = ("the arm's per-fold AUCs on the control folds were not "
                  "recorded, so the arm could not be compared with its own "
                  "shuffle")
    elif any(fold is None for _, fold in draws):
        # A control labelled only in part would quietly compare the arm on a
        # subset and call the rest certified.
        reason = ("some shuffled fits carry no fold id, so the arm could not be "
                  "paired against all of its own draws")
    else:
        reason = None
    if reason:
        return {"evaluated": False, "passed": None, "n_folds": 0,
                "mean": None, "sem": None, "folds": [], "reason": reason}

    per_fold: dict[object, list[float]] = {}
    for value, fold in draws:
        per_fold.setdefault(fold, []).append(value)

    gaps, used = [], []
    for fold in sorted(per_fold, key=str):
        arm = arm_auc.get(fold)
        if arm is None:
            arm = arm_auc.get(str(fold))
        if arm is None or math.isnan(float(arm)):
            return {"evaluated": False, "passed": None, "n_folds": 0,
                    "mean": None, "sem": None, "folds": [],
                    "reason": f"no arm AUC was recorded for control fold {fold}"}
        gaps.append(float(arm) - float(np.mean(per_fold[fold])))
        used.append(fold)

    if len(gaps) < 2:
        return {"evaluated": False, "passed": None, "n_folds": len(gaps),
                "mean": float(gaps[0]) if gaps else None, "sem": None,
                "folds": used,
                "reason": "the control ran on a single fold, so the gap over "
                          "folds has no spread to test against"}

    gaps = np.asarray(gaps, dtype=float)
    mean = float(gaps.mean())
    sem = float(gaps.std(ddof=1) / math.sqrt(len(gaps)))
    required = max(minimum * sem, floor)
    return {
        "evaluated": True,
        "passed": bool(mean >= required),
        "n_folds": len(gaps),
        "mean": mean,
        "sem": sem,
        "required": float(required),
        "folds": used,
        "reason": None,
    }


def gate(control_aucs, arm_auc=None, folds=None, *,
         band: tuple[float, float] = CONTROL_BAND,
         tolerance_floor: float = CONTROL_TOLERANCE_FLOOR,
         separation_sem: float = CONTROL_SEPARATION_SEM) -> dict:
    """Judge the shuffled fits and say whether the run may be reported."""
    draws = _usable(control_aucs, folds)
    if not draws:
        return {"n": 0, "mean": None, "passed": False, "band": list(band),
                "tolerance": None, "min": None, "max": None,
                "sd": None, "sem": None,
                "spec_band": {"passed": False, "low": band[0], "high": band[1]},
                "chance": {"passed": False, "distance": None, "tolerance": None},
                "separation": {"evaluated": False, "passed": None, "n_folds": 0,
                               "mean": None, "sem": None, "folds": [],
                               "reason": "no usable shuffled fits"},
                "reason": "no usable shuffled fits"}

    values = np.asarray([value for value, _ in draws], dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    sem = (sd / math.sqrt(len(values))) if not math.isnan(sd) else float("nan")
    tolerance = (max(2.0 * sem, tolerance_floor) if not math.isnan(sem)
                 else tolerance_floor)

    distance = abs(mean - 0.5)
    chance_ok = distance <= tolerance
    # The spec's fixed band, kept and reported in every result file so a reader
    # can see what the old rule would have said. It does not decide anything.
    spec_ok = band[0] <= mean <= band[1]

    separation = _separation(draws, arm_auc, separation_sem, tolerance_floor)
    reasons = []
    if not chance_ok:
        reasons.append(
            f"the shuffled fits are not at chance: their mean {mean:.3f} sits "
            f"{distance:.3f} from 0.5, beyond the {tolerance:.3f} the draws "
            f"themselves allow ({len(values)} draws, sd {sd:.3f})")
    if separation["evaluated"] and not separation["passed"]:
        reasons.append(
            f"the arm does not beat its own shuffle: the paired gap over "
            f"{separation['n_folds']} folds is {separation['mean']:.3f}, short "
            f"of {separation['required']:.3f} ({separation_sem:g} SEM)")
    if not separation["evaluated"]:
        reasons.append(
            f"the arm could not be compared with its own shuffle "
            f"({separation['reason']})")

    passed = not reasons
    return {
        "n": int(len(values)),
        "mean": mean,
        "min": float(values.min()),
        "max": float(values.max()),
        "sd": sd,
        "sem": sem,
        "band": list(band),
        "tolerance": float(tolerance),
        "spec_band": {"passed": bool(spec_ok), "low": float(band[0]),
                      "high": float(band[1])},
        "chance": {"passed": bool(chance_ok), "distance": float(distance),
                   "tolerance": float(tolerance)},
        "separation": separation,
        "passed": bool(passed),
        "reason": None if passed else "; ".join(reasons),
    }
