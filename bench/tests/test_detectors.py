import polars as pl
import pytest

from bench.detectors import (
    detect_d1_timeout, detect_d2_restart_chain, detect_d3_failure,
    detect_d4_duplicate, detect_d5_wallclock,
    validate_findings,
)


def _jobs() -> pl.DataFrame:
    return pl.DataFrame({
        "row_id": [10, 11, 12, 13, 14],
        "job_id": ["a", "b", "c", "d", "e"],
        "user_hash": ["u", "u", "u", "v", "v"],
        "end_time": pl.datetime_range(
            pl.datetime(2025, 1, 1, 0), pl.datetime(2025, 1, 5, 0), interval="1d", eager=True
        ),
        "submit_time": pl.datetime_range(
            pl.datetime(2025, 1, 1), pl.datetime(2025, 1, 5), interval="1d", eager=True
        ),
        "state": ["TIMEOUT", "FAILED", "COMPLETED", "COMPLETED", "CANCELLED"],
        "energy_j": [None, 20.0, 30.0, None, None],
        "energy_tier": ["none", "measured", "measured", "none", "none"],
        "elapsed_s": [100.0, 50.0, 10.0, 80.0, 10.0],
        "timelimit_s": [100.0, 100.0, 100.0, 100.0, 0.0],
        "queue_wait_s": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cpus_req": [1, 2, 3, 4, 5],
        "cpus_used": [1, 1, 2, 4, 1],
        "script_hash": ["ha", "hb", "hc", "hd", "he"],
        "partition": ["p", "p", "p", "p", "p"],
        "qos": ["q", "q", "q", "q", "q"],
        "mem_req": [1.0, 1.0, 1.0, 1.0, 1.0],
        "gpus_req": [0, 0, 0, 0, 0],
        "nodes_req": [1, 1, 1, 1, 1],
    })


def _kestrel_pairs() -> pl.DataFrame:
    return pl.DataFrame({
        "row_id": [20, 21, 22, 23],
        "job_id": ["t", "r", "d1", "d2"],
        "user_hash": ["u", "u", "v", "v"],
        "submit_time": pl.datetime_range(
            pl.datetime(2025, 2, 1, 0), pl.datetime(2025, 2, 1, 3), interval="1h", eager=True
        ),
        "end_time": pl.datetime_range(
            pl.datetime(2025, 2, 1, 0, 30), pl.datetime(2025, 2, 1, 3, 30), interval="1h", eager=True
        ),
        "state": ["TIMEOUT", "FAILED", "COMPLETED", "COMPLETED"],
        "energy_j": [10.0, 20.0, 30.0, 40.0],
        "energy_tier": ["measured"] * 4,
        "elapsed_s": [3600.0, 100.0, 100.0, 105.0],
        "timelimit_s": [3600.0] * 4,
        "queue_wait_s": [1.0] * 4,
        "cpus_req": [1] * 4,
        "cpus_used": [1] * 4,
        "script_hash": ["s", "s", "d", "d"],
        "partition": ["p"] * 4,
        "qos": ["q"] * 4,
        "mem_req": [1.0] * 4,
        "gpus_req": [0] * 4,
        "nodes_req": [1] * 4,
    })


def test_d1_emits_timeout_with_source_and_energy_tier():
    out = detect_d1_timeout(_jobs())
    assert out["row_id"].to_list() == [10]
    assert out["finding_type"].to_list() == ["timeout_waste"]
    assert out["energy_tier"].to_list() == ["none"]
    validate_findings(out)


def test_d3_emits_only_hard_failure_states():
    out = detect_d3_failure(_jobs())
    assert out["row_id"].to_list() == [11]
    assert out["evidence_failure_state"].to_list() == ["FAILED"]


def test_d5_wallclock_excludes_timeout_and_scores_underuse():
    out = detect_d5_wallclock(_jobs())
    assert out["row_id"].to_list() == [12]
    assert out["score"].to_list() == pytest.approx([0.9])


def test_d2_links_timeout_to_next_same_user_and_limit():
    out = detect_d2_restart_chain(_kestrel_pairs())
    assert out["row_id"].to_list() == [21]
    assert out["evidence_prior_row_id"].to_list() == [20]
    assert out["finding_type"].to_list() == ["intentional_checkpoint"]


def test_d4_links_near_identical_workloads():
    out = detect_d4_duplicate(_kestrel_pairs())
    assert out["row_id"].to_list() == [23]
    assert out["evidence_prior_row_id"].to_list() == [22]
    assert out["finding_type"].to_list() == ["duplicate_workload"]


def test_kestrel_detectors_reject_missing_script_hash():
    frame = _kestrel_pairs().with_columns(pl.lit(None, dtype=pl.String).alias("script_hash"))
    with pytest.raises(ValueError, match="script_hash"):
        detect_d2_restart_chain(frame)
    with pytest.raises(ValueError, match="script_hash"):
        detect_d4_duplicate(frame)


def test_detectors_require_authoritative_row_id():
    with pytest.raises(ValueError, match="row_id"):
        detect_d1_timeout(_jobs().drop("row_id"))
