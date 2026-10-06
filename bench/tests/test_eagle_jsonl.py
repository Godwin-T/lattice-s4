"""End-to-end check of the Eagle 3-month JSONL adapter on a small synthetic file."""
import json

import polars as pl

from bench.ingest import validate
from bench.ingest.adapters import eagle_jsonl
from bench.ingest.filters import apply_filters
from bench.ingest.schema import CANONICAL_COLUMNS


def _row(**kw):
    base = {
        "job_id": 1, "user": "AAAA1111", "account": "ACCT2222", "script": "SCR3333",
        "partition": "standard", "qos": "normal",
        "submit_time": "2019-12-04T16:10:02.000Z",
        "start_time": "2019-12-04T16:10:07.000Z",
        "end_time": "2019-12-04T16:17:00.000Z",
        "wallclock_req": "P0DT0H30M0S", "wallclock_used": "P0DT0H6M53S",
        "queue_wait": "P0DT0H0M5S", "processors_req": 2, "processors_used": 72,
        "nodes_req": 2, "nodes_used": 2, "avg_power": 216.58333,
        "std_power": 0.5, "nodelist": ["n1", "n2"], "array_pos": None,
        "state": "TIMEOUT",
    }
    base.update(kw)
    return base


def _write(tmp_path, rows):
    p = tmp_path / "anon_jobs_2019-12.json"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_adapter_shapes_and_energy(tmp_path):
    rows = [
        _row(job_id=1, state="TIMEOUT"),
        _row(job_id=2, state="COMPLETED", wallclock_used="P0DT0H1M0S",
             avg_power=100.0, nodes_used=1, end_time="2019-12-04T16:11:02.000Z"),
        _row(job_id=3, state="CANCELLED by 1234", wallclock_used=None,
             avg_power=0.0, nodes_used=0),          # never ran
        _row(job_id=4, state="COMPLETED", wallclock_used="P0DT2H0M0S",
             avg_power=50.0, nodes_used=1),
    ]
    df = eagle_jsonl.load([_write(tmp_path, rows)])

    # Canonical shape, in order.
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.height == 4

    # Durations parsed to seconds; state normalised; ids passed through.
    r1 = df.filter(pl.col("job_id") == "1").row(0, named=True)
    assert r1["timelimit_s"] == 1800.0
    assert r1["elapsed_s"] == 413.0
    assert r1["state"] == "TIMEOUT"
    assert r1["user_hash"] == "AAAA1111"

    # Modelled energy = avg_power x nodes_used x elapsed_s.
    assert r1["energy_tier"] == "modelled"
    assert abs(r1["energy_j"] - 216.58333 * 2 * 413) < 1e-6

    # A cancelled job with no wallclock_used cannot carry energy.
    r3 = df.filter(pl.col("job_id") == "3").row(0, named=True)
    assert r3["elapsed_s"] is None
    assert r3["energy_j"] is None
    assert r3["energy_tier"] == "none"

    # Structural validators pass on the mapped frame.
    assert validate.run_checks(df) == []


def test_filters_drop_the_never_ran_row(tmp_path):
    rows = [_row(job_id=1), _row(job_id=2, wallclock_used=None)]
    df = eagle_jsonl.load([_write(tmp_path, rows)])
    kept, stats = apply_filters(df)
    assert stats["dropped_no_elapsed"] == 1
    assert kept.height == 1
    assert stats["rows_out"] == 1


def test_filters_drop_non_terminal_states(tmp_path):
    rows = [_row(job_id=1), _row(job_id=2, state="RUNNING")]
    df = eagle_jsonl.load([_write(tmp_path, rows)])
    kept, stats = apply_filters(df)
    assert stats["dropped_non_terminal_state"] == 1
    assert kept.height == 1
