"""Split generator: month floor, eligibility, folds, sampling, validation."""
import datetime as dt

import polars as pl
import pytest

from bench.splits import eligibility, folds as folds_mod, sample as sample_mod
from bench.splits.config import MIN_FOLDS
from bench.splits.months import timeline
from bench.splits.validate import check_manifest


def _table(month_rows: dict[str, int], n_users: int = 40,
           timeout_every: int = 5) -> pl.DataFrame:
    """A minimal canonical-shaped frame: enough columns for the splitter."""
    rows = []
    for month, n in month_rows.items():
        year, mon = (int(x) for x in month.split("-"))
        for i in range(n):
            rows.append({
                "job_id": f"{month}-{i:05d}",
                "user_hash": f"u{i % n_users:03d}",
                "state": "TIMEOUT" if i % timeout_every == 0 else "COMPLETED",
                "submit_time": dt.datetime(year, mon, 1 + (i % 27), tzinfo=dt.timezone.utc),
            })
    return pl.DataFrame(rows)


# --- T2: months -------------------------------------------------------------

def test_timeline_drops_months_below_the_floor():
    df = _table({"2020-01": 1_500, "2020-02": 500, "2020-03": 2_000})
    months, dropped = timeline(df.lazy())
    assert months == ["2020-01", "2020-03"]
    assert dropped == [("2020-02", 500)]


# --- T3: eligibility --------------------------------------------------------

def test_eligibility_keeps_only_users_with_enough_history():
    df = pl.concat([
        _table({"2020-01": 240}, n_users=30),     # 8 jobs each
        _table({"2020-02": 250}, n_users=10),     # 25 jobs each
    ])
    users = eligibility.eligible_users(df.lazy())
    assert users.height == 10
    summary = eligibility.eligible_summary(users)
    assert summary["count"] == 10 and len(summary["sha256"]) == 64


# --- T4: folds --------------------------------------------------------------

def test_folds_expand_and_respect_the_holdout():
    months_in = {f"2020-{m:02d}": 800 for m in range(1, 11)}   # 10 months
    df = _table(months_in, n_users=200, timeout_every=5)
    months, _ = timeline(df.lazy(), min_rows=1)   # floor tested separately
    stats = folds_mod.month_stats(df.lazy())
    got, locked, open_months = folds_mod.build_folds(months, stats)

    assert locked == ["2020-05", "2020-06", "2020-07", "2020-08", "2020-09", "2020-10"]
    assert open_months == ["2020-01", "2020-02", "2020-03", "2020-04"]
    assert [f["test_month"] for f in got] == ["2020-02", "2020-03", "2020-04"]
    # Expanding: each fold's train set is every earlier open month.
    assert got[0]["train_months"] == ["2020-01"]
    assert got[2]["train_months"] == ["2020-01", "2020-02", "2020-03"]
    # Deterministic and internally consistent.
    assert [f["n_train"] for f in got] == sorted(f["n_train"] for f in got)


def test_folds_refuse_when_too_short():
    df = _table({f"2020-{m:02d}": 500 for m in range(1, 4)})   # only 3 months
    months, _ = timeline(df.lazy())
    with pytest.raises(ValueError):
        folds_mod.build_folds(months, folds_mod.month_stats(df.lazy()))


# --- T5: sample -------------------------------------------------------------

def test_quota_allocation_is_proportional_with_a_floor():
    sizes = {("A", 1): 10_000, ("B", 1): 9_000, ("C", 1): 3, ("D", 1): 2}
    quotas = sample_mod.allocate_quotas(sizes, n=1_000, floor=25)
    assert sum(quotas.values()) == 1_000
    assert quotas[("C", 1)] == 3 and quotas[("D", 1)] == 2     # take all that exist
    assert quotas[("A", 1)] > quotas[("B", 1)]                 # proportional


def test_sample_is_deterministic_and_inside_the_test_months():
    months_in = {f"2020-{m:02d}": 600 for m in range(1, 13)}   # 12 months -> 5 folds
    df = _table(months_in, n_users=120, timeout_every=4)
    months, _ = timeline(df.lazy(), min_rows=1)
    _, _, open_months = folds_mod.build_folds(months, folds_mod.month_stats(df.lazy()))
    test_months = open_months[1:]

    users = sample_mod.with_activity_quartile(
        eligibility.eligible_users(df.lazy(), min_jobs=1))
    a = sample_mod.build_sample(df.lazy(), test_months, users, seed=42, n=300)
    b = sample_mod.build_sample(df.lazy(), test_months, users, seed=42, n=300)
    assert a["job_ids"] == b["job_ids"]                 # deterministic
    assert a["n"] == len(a["job_ids"])
    # Every sampled job really comes from a test month.
    assert all(any(job.startswith(m) for m in test_months) for job in a["job_ids"])


def test_sample_records_quota_and_taken_per_stratum():
    months_in = {f"2020-{m:02d}": 600 for m in range(1, 13)}
    df = _table(months_in, n_users=120, timeout_every=4)
    months, _ = timeline(df.lazy(), min_rows=1)
    _, _, open_months = folds_mod.build_folds(months, folds_mod.month_stats(df.lazy()))
    users = sample_mod.with_activity_quartile(
        eligibility.eligible_users(df.lazy(), min_jobs=1))
    result = sample_mod.build_sample(df.lazy(), open_months[1:], users, seed=1, n=300)
    assert result["strata"]
    assert all({"state", "activity_quartile", "quota", "taken"} <= set(s)
               for s in result["strata"])
    assert sum(s["taken"] for s in result["strata"]) == result["n"]


# --- T6: validation ---------------------------------------------------------

def _good_manifest():
    return {
        "schema": "lattice24-benchmark/folds/1",
        "locked": {"months": ["2020-09", "2020-10"]},
        "folds": [{"fold_id": 1, "test_month": "2020-02",
                   "train_months": ["2020-01"], "n_train": 500, "n_test": 500,
                   "pos_train": 50, "pos_test": 50}],
        "sample": {"job_ids": ["a", "b"], "job_ids_sha256": None},
    }


def test_validate_rejects_a_test_month_inside_its_train_set():
    bad = _good_manifest()
    bad["folds"][0]["train_months"] = ["2020-02"]          # leaks itself
    problems = check_manifest(bad)
    assert any("test month inside train months" in p for p in problems)


def test_validate_rejects_a_holdout_month_used_in_a_fold():
    bad = _good_manifest()
    bad["folds"][0]["test_month"] = "2020-10"              # locked
    problems = check_manifest(bad)
    assert any("holdout month used in a fold" in p for p in problems)


def test_validate_rejects_a_broken_sample_hash():
    bad = _good_manifest()
    bad["sample"]["job_ids_sha256"] = "not-the-right-hash"
    problems = check_manifest(bad)
    assert any("job_ids_sha256" in p for p in problems)
