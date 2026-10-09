import datetime as dt
import json

import polars as pl
import pytest

from bench.arms.common import with_row_id
from bench.arms.domain_features import (
    FEATURE_COLUMNS,
    build_domain_features,
    ensure_domain_features,
    validate_domain_features,
)

UTC = dt.timezone.utc


def _rows(n=8):
    rows = []
    submit = dt.datetime(2020, 1, 1, tzinfo=UTC)
    for i in range(n):
        end = submit + dt.timedelta(seconds=10)
        rows.append({
            "job_id": str(i), "user_hash": "u1", "submit_time": submit,
            "end_time": end, "elapsed_s": 10.0, "timelimit_s": 100.0,
            "state": "TIMEOUT" if i == 1 else "COMPLETED",
        })
        submit = end + dt.timedelta(seconds=1)
    return rows


def _canonical(rows):
    return with_row_id(pl.DataFrame(rows).with_columns(
        pl.col("submit_time").cast(pl.Datetime("us", "UTC")),
        pl.col("end_time").cast(pl.Datetime("us", "UTC")),
    ).lazy()).collect()


def test_domain_features_are_one_to_one_and_strictly_historical():
    canonical = _canonical(_rows())
    features = build_domain_features(canonical)

    validate_domain_features(features, canonical)
    assert features.height == canonical.height
    assert features.filter(pl.col("domain_history_available") == 0).height == 1
    # The timeout is only visible after its end, never on its own row.
    assert features["timeouts_last_5"].to_list()[:3] == [0, 0, 1]
    assert features["time_since_last_timeout_s"][2] == pytest.approx(1.0)


def test_future_outcome_changes_do_not_change_target_features():
    canonical = _canonical(_rows(8))
    before = build_domain_features(canonical)
    target_id = 5

    changed = canonical.with_columns(
        pl.when(pl.col("row_id") > target_id)
        .then(pl.lit("FAILED"))
        .otherwise(pl.col("state")).alias("state"),
        pl.when(pl.col("row_id") > target_id)
        .then(pl.lit(9999.0))
        .otherwise(pl.col("elapsed_s")).alias("elapsed_s"),
    )
    after = build_domain_features(changed)
    left = before.filter(pl.col("row_id") == target_id).select(FEATURE_COLUMNS)
    right = after.filter(pl.col("row_id") == target_id).select(FEATURE_COLUMNS)
    assert left.equals(right)


def test_missing_history_is_explicit_not_imputed_as_history():
    features = build_domain_features(_canonical(_rows(1)))
    assert features["domain_history_available"].to_list() == [0]
    assert features["hist_elapsed_p95_s"].null_count() == 1
    assert features["failures_last_10"].to_list() == [0]


def test_kestrel_restart_and_duplicate_history_is_traceable():
    rows = _rows(4)
    rows[0]["state"] = "TIMEOUT"
    rows[1]["state"] = "COMPLETED"
    for row in rows:
        row.update({
            "script_hash": "script-a", "partition": "p1", "qos": "normal",
            "cpus_req": 1, "mem_req": 1.0, "gpus_req": 0, "nodes_req": 1,
            "energy_j": 10.0,
        })
    features = build_domain_features(_canonical(rows), dataset="kestrel")

    # Row 1 follows the timeout at row 0 within two hours and has the same
    # shape/runtime, so both detector histories point to row 0.
    row = features.filter(pl.col("row_id") == 1).row(0, named=True)
    assert row["prior_timeout_chain"] == 1
    assert row["prior_timeout_same_script"] is True
    assert row["prior_timeout_row_id"] == 0
    assert row["prior_duplicate_workload"] == 1
    assert row["prior_duplicate_same_script"] is True
    assert row["prior_duplicate_same_shape"] is True
    assert row["prior_duplicate_runtime_delta_s"] == pytest.approx(0.0)


def test_kestrel_only_history_is_unavailable_without_script_hash():
    features = build_domain_features(_canonical(_rows(3)), dataset="eagle_parquet")
    assert features["prior_timeout_chain"].null_count() == features.height
    assert features["prior_duplicate_workload"].null_count() == features.height


def test_domain_cache_is_versioned_and_reusable(tmp_path):
    canonical = _canonical(_rows(4))
    canonical_path = tmp_path / "kestrel.parquet"
    canonical.write_parquet(canonical_path)
    cache_dir = tmp_path / "cache"

    first = ensure_domain_features(
        canonical.lazy(), canonical_path, cache_dir,
        dataset="kestrel", max_chunk_rows=2,
    )
    second = ensure_domain_features(
        canonical.lazy(), canonical_path, cache_dir,
        dataset="kestrel", max_chunk_rows=2,
    )
    assert first == second
    meta = json.loads(first.with_suffix(".meta.json").read_text())
    assert meta["domain_feature_version"] == 1
    assert meta["rows"] == canonical.height
    assert meta["capabilities"]["kestrel_restart_history"] is False
    cached = pl.read_parquet(first)
    validate_domain_features(cached, canonical)
