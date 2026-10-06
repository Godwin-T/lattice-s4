"""Arm A: window features, the history rule, and the hand-in contract."""
import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from bench.arms import common
from bench.arms.arm_a import run as run_arm_a
from bench.splits.eligibility import eligible_summary, eligible_users
from bench.splits.folds import build_folds, month_stats


def _canonical(rows: list[dict]) -> pl.DataFrame:
    """The subset of the canonical schema the arms need."""
    return pl.DataFrame(rows).with_columns(
        pl.col("submit_time").cast(pl.Datetime("us", "UTC")),
        pl.col("end_time").cast(pl.Datetime("us", "UTC")),
    )


def _job(job_id, user, submit, used_s, limit_s, state="COMPLETED", energy=1.0):
    return {
        "job_id": str(job_id), "user_hash": user,
        "submit_time": submit, "end_time": submit + dt.timedelta(seconds=used_s),
        "elapsed_s": float(used_s), "timelimit_s": float(limit_s),
        "state": state, "energy_j": energy,
    }


def test_window_features_match_a_hand_computation():
    base = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    ratios = [round(0.1 + 0.03 * i, 4) for i in range(24)]      # 24 history jobs
    rows = [_job(i, "u1", base + dt.timedelta(hours=i), r * 3600, 3600)
            for i, r in enumerate(ratios)]
    rows.append(_job(99, "u1", base + dt.timedelta(hours=100), 1800, 3600))

    windows = common.build_windows(_canonical(rows).lazy()).collect()
    target = windows.filter(pl.col("job_id") == "99").row(0, named=True)

    hist = np.array(ratios)
    assert target["scoreable"] is True
    assert target["f_mean"] == pytest.approx(hist.mean())
    assert target["f_std"] == pytest.approx(hist.std())            # ddof=0
    assert target["f_range"] == pytest.approx(hist.max() - hist.min())
    assert target["f_madiff"] == pytest.approx(np.abs(np.diff(hist)).mean())


def test_history_still_running_makes_the_target_unscorable():
    base = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    rows = [_job(i, "u1", base + dt.timedelta(hours=i), 60, 3600) for i in range(24)]
    # Job 23 is still running when job 99 is submitted -> window not knowable.
    rows[23] = _job(23, "u1", base + dt.timedelta(hours=23), 60, 3600)
    rows[23]["end_time"] = base + dt.timedelta(hours=200)
    rows.append(_job(99, "u1", base + dt.timedelta(hours=100), 1800, 3600))

    windows = common.build_windows(_canonical(rows).lazy()).collect()
    target = windows.filter(pl.col("job_id") == "99").row(0, named=True)
    assert target["scoreable"] is False


def _synthetic_dataset() -> pl.DataFrame:
    """12 months, 60 users, comfortably above every guardrail."""
    rows = []
    for month in range(1, 13):
        for user in range(60):
            for k in range(12):
                jid = f"{month:02d}-{user:02d}-{k:02d}"
                submit = dt.datetime(2020, month, 1, tzinfo=dt.timezone.utc) \
                    + dt.timedelta(days=k, hours=user % 12)
                ratio = 0.85 if (k % 5 == 0 and user % 3 == 0) else 0.3
                state = "TIMEOUT" if ratio == 0.85 else "COMPLETED"
                rows.append(_job(jid, f"u{user:03d}", submit,
                                 used_s=ratio * 3600, limit_s=3600, state=state))
    return _canonical(rows)


def _manifest_for(df: pl.DataFrame, tmp_path) -> str:
    users = eligible_users(df.lazy())
    months = sorted(df["submit_time"].dt.strftime("%Y-%m").unique().to_list())
    folds, locked, _ = build_folds(months, month_stats(df.lazy()))
    manifest = {
        "schema": "lattice24-benchmark/folds/1",
        "dataset": "synthetic",
        "eligible_users": eligible_summary(users),
        "folds": folds,
        "locked": {"months": locked},
    }
    path = tmp_path / "synthetic.folds.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def test_arm_a_emits_one_row_per_test_job(tmp_path):
    df = _synthetic_dataset()
    canonical = tmp_path / "canonical.parquet"
    df.write_parquet(canonical)
    manifest = _manifest_for(df, tmp_path)

    outdir = tmp_path / "runs"
    meta = run_arm_a(manifest_path=manifest, canonical_path=str(canonical),
                     outdir=str(outdir), seed=7, control_repeats=1, control_folds=2)
    preds = pl.read_parquet(outdir / meta["run_id"] / "predictions.parquet")

    # Contract: one row per test job per fold, in the documented shape.
    assert list(preds.columns) == list(common.PREDICTION_SCHEMA)
    assert preds.height == meta["test_rows"]
    assert preds["job_id"].n_unique() == preds.height
    assert set(preds["task"].unique().to_list()) == {"T1"}
    assert set(preds["arm"].unique().to_list()) == {"A"}

    # The first months have no 24-job history at all, so those folds must be
    # skipped rather than crashing — and every test row still appears.
    assert meta["skipped_folds"], "expected the warm-up folds to be skipped"
    skipped_months = {f["test_month"] for f in meta["skipped_folds"]}
    skipped_rows = preds.filter(pl.col("test_month").is_in(skipped_months))
    assert skipped_rows.height > 0
    assert skipped_rows["score"].null_count() == skipped_rows.height

    # Scorable rows carry a probability; unscorable ones are explicit nulls.
    scored = preds.filter(pl.col("score").is_not_null())
    assert scored.height > 0
    assert scored["probability"].is_between(0, 1).all()
    # The control ran and produced numbers.
    assert meta["control_mean"] is not None


def test_arm_a_refuses_a_mismatched_eligible_population(tmp_path):
    df = _synthetic_dataset()
    canonical = tmp_path / "canonical.parquet"
    df.write_parquet(canonical)
    manifest_path = _manifest_for(df, tmp_path)
    manifest = json.loads(open(manifest_path).read())
    manifest["eligible_users"] = {"count": 999, "sha256": "wrong"}
    open(manifest_path, "w").write(json.dumps(manifest))

    with pytest.raises(SystemExit):
        run_arm_a(manifest_path=manifest_path, canonical_path=str(canonical),
                  outdir=str(tmp_path / "runs"))
