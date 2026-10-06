"""Evaluator: the toy examples from metrics.md, the contract, and refusals."""
import datetime as dt
import json

import numpy as np
import polars as pl

from bench.eval import calibration, control as control_mod, results as results_mod
from bench.eval.aggregate import to_arrays, views
from bench.eval.energy import energy_captured
from bench.eval.predictions import check_contract
from bench.eval.ranking import auc, average_precision, flag_top_k, recall_and_false_flag
from bench.splits.eligibility import eligible_summary, eligible_users
from bench.splits.folds import build_folds, month_stats


# --- ranking ---------------------------------------------------------------

def test_auc_known_values():
    assert auc([1, 0], [1.0, 0.0]) == 1.0          # perfectly ordered
    assert auc([0, 1], [1.0, 0.0]) == 0.0          # perfectly reversed
    assert auc([1, 0], [1.0, 1.0]) == 0.5          # tied -> coin flip


def test_average_precision_is_one_for_perfect_ranking():
    assert average_precision([1, 1, 0, 0], [4.0, 3.0, 2.0, 1.0]) == 1.0
    # Both positives ranked last: the first is found at precision 1/3 (recall
    # 1/2), the second at precision 1/2 (recall 1).
    # AP = (0.5 - 0) * (1/3) + (1 - 0.5) * 0.5 = 0.4166...
    assert round(average_precision([0, 0, 1, 1], [4.0, 3.0, 2.0, 1.0]), 4) == 0.4167


# --- the worked example from metrics.md section 3.2 ------------------------

def _worked_example():
    """100 jobs; 10 time out holding 1,000 kWh; the top 5% catch 4 holding 700."""
    labels, scores, energy = [], [], []
    for i in range(100):
        positives = i < 10                          # jobs 0-9 time out
        labels.append(positives)
        if not positives:
            # one ordinary job is flagged by mistake, the rest are not
            scores.append(6.5 if i == 10 else 1.0)
            energy.append(10.0)
        elif i < 4:
            scores.append(10.0 - i)                 # the four we catch
            energy.append(175.0)                    # 4 x 175 = 700 kWh
        else:
            scores.append(5.0 - (i - 4) * 0.1)      # the six we miss
            energy.append(50.0)                     # 6 x 50 = 300 kWh
    return pl.DataFrame({"label": labels, "score": scores, "energy_j": energy})


def test_energy_captured_matches_the_documented_example():
    frame = _worked_example()
    out = energy_captured(frame, 5)
    assert out["total_positive_energy_j"] == 1000.0
    assert round(out["energy_captured"], 3) == 0.700    # 700 of 1,000 kWh
    assert out["coverage_pos"] == 1.0

    flagged = flag_top_k(frame["score"].to_numpy(), 5)
    recall, false_flag = recall_and_false_flag(frame["label"].to_numpy(), flagged)
    assert round(recall, 3) == 0.400                    # 4 of 10 timeouts
    assert round(false_flag, 4) == 0.0111               # 1 of 90 good jobs


def test_energy_is_not_available_when_there_is_none():
    frame = _worked_example().with_columns(pl.lit(None, dtype=pl.Float64).alias("energy_j"))
    out = energy_captured(frame, 5)
    assert out["energy_captured"] is None               # not zero
    assert out["total_positive_energy_j"] == 0.0


# --- calibration -----------------------------------------------------------

def test_ece_is_zero_when_perfectly_calibrated():
    probabilities = np.full(100, 0.5)
    labels = np.array([1] * 50 + [0] * 50)
    assert calibration.ece(probabilities, labels) < 1e-9


def test_ece_is_large_when_over_confident():
    probabilities = np.full(100, 0.99)
    labels = np.zeros(100)
    assert calibration.ece(probabilities, labels) > 0.5


# --- control gate ----------------------------------------------------------

def test_control_band_admits_chance_and_rejects_a_leak():
    assert control_mod.gate([0.49, 0.51, 0.50])["passed"] is True
    rejected = control_mod.gate([0.67, 0.65, 0.69])
    assert rejected["passed"] is False
    assert "outside" in rejected["reason"]


# --- aggregation -----------------------------------------------------------

def test_views_are_computed_over_months():
    frame = pl.DataFrame({
        "label": [True, False] * 6,
        "score": [0.9, 0.1] * 6,
        "energy_j": [1.0] * 12,
        "test_month": ["2020-01", "2020-01", "2020-02", "2020-02",
                       "2020-03", "2020-03", "2020-04", "2020-04",
                       "2020-05", "2020-05", "2020-06", "2020-06"],
        "user_hash": [f"u{i}" for i in range(12)],
    })
    arrays = to_arrays(frame)
    out = views(arrays, lambda a: auc(a["label"], a["score"]),
                resamples=40, seed=1)
    assert out["n_months"] == 6
    assert out["median"] == 1.0                          # every month separates
    assert out["pooled"] == 1.0


# --- the contract, end to end ---------------------------------------------

def _synthetic(tmp_path):
    rows = []
    for month in range(1, 13):
        for user in range(20):
            for k in range(30):
                submit = dt.datetime(2020, month, 1, tzinfo=dt.timezone.utc) \
                    + dt.timedelta(hours=k * 8)
                ratio = 0.9 if (k % 7 == 0) else 0.3
                state = "TIMEOUT" if ratio == 0.9 else "COMPLETED"
                rows.append({
                    "job_id": f"{month:02d}-{user:03d}-{k:03d}",
                    "user_hash": f"u{user:03d}",
                    "submit_time": submit,
                    "end_time": submit + dt.timedelta(seconds=ratio * 3600),
                    "elapsed_s": ratio * 3600, "timelimit_s": 3600.0,
                    "state": state, "energy_j": 100.0 if state == "TIMEOUT" else 10.0,
                    "energy_tier": "measured",
                })
    df = pl.DataFrame(rows).with_columns(
        pl.col("submit_time").cast(pl.Datetime("us", "UTC")),
        pl.col("end_time").cast(pl.Datetime("us", "UTC")))
    canonical = tmp_path / "canonical.parquet"
    df.write_parquet(canonical)

    months = sorted(df["submit_time"].dt.strftime("%Y-%m").unique().to_list())
    folds, locked, _ = build_folds(months, month_stats(df.lazy()))
    manifest = {
        "schema": "lattice24-benchmark/folds/1", "dataset": "synthetic",
        "eligible_users": eligible_summary(eligible_users(df.lazy())),
        "folds": folds, "locked": {"months": locked},
        "checksums": {"manifest_sha256": "deadbeef"},
    }
    manifest_path = tmp_path / "synthetic.folds.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return df, str(canonical), manifest


def _predictions(df, manifest, scores=None):
    test_months = {f["test_month"] for f in manifest["folds"]}
    keep = df.filter(
        pl.col("submit_time").dt.strftime("%Y-%m").is_in(test_months)
    ).with_columns(
        pl.col("submit_time").dt.strftime("%Y-%m").alias("test_month"))
    if scores is None:
        scores = keep["elapsed_s"].to_numpy() / keep["timelimit_s"].to_numpy()
    rows = []
    for fold in manifest["folds"]:
        part = keep.filter(pl.col("test_month") == fold["test_month"])
        mask = (keep["test_month"] == fold["test_month"]).to_numpy()
        for job_id, score in zip(part["job_id"].to_list(), np.asarray(scores)[mask]):
            rows.append({
                "run_id": "test-run", "arm": "A", "task": "T1",
                "fold_id": fold["fold_id"], "test_month": fold["test_month"],
                "job_id": job_id, "score": float(score),
                "probability": float(score), "latency_ms": None,
            })
    return pl.DataFrame(rows)


def test_contract_passes_for_a_complete_hand_in(tmp_path):
    df, canonical, manifest = _synthetic(tmp_path)
    preds = _predictions(df, manifest)
    assert check_contract(preds, manifest, canonical) == []


def test_contract_catches_a_missing_row_and_an_extra_row(tmp_path):
    df, canonical, manifest = _synthetic(tmp_path)
    preds = _predictions(df, manifest)

    missing = check_contract(preds.slice(1), manifest, canonical)
    assert any("were not scored" in p for p in missing)

    extra_row = pl.DataFrame([{**preds.row(0, named=True), "job_id": "not-a-job"}])
    extra = check_contract(pl.concat([preds, extra_row]), manifest, canonical)
    assert any("not test rows" in p for p in extra)


def test_evaluate_reports_metrics_and_suppresses_when_invalid(tmp_path):
    df, canonical, manifest = _synthetic(tmp_path)
    preds = _predictions(df, manifest)

    good = results_mod.evaluate(preds, manifest, canonical, resamples=30,
                                control_aucs=[0.49, 0.51, 0.50])
    assert good["valid"] is True
    assert good["headline"] is not None
    assert good["metrics"]["energy_captured"]["5"]["median"] is not None
    assert good["counts"]["energy_coverage_over_positives"] == 1.0
    assert good["manifest_sha256"] == "deadbeef"

    bad = results_mod.evaluate(preds, manifest, canonical, resamples=30,
                               control_aucs=[0.67, 0.65, 0.69])
    assert bad["valid"] is False
    assert bad["headline"] is None                       # no number published
    assert "outside" in bad["invalid_reason"]
