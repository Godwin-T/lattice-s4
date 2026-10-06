"""Kestrel adapter: duration[ns], mixed timezones, measured energy, mem_req units."""
import datetime as dt

import polars as pl

from bench.ingest import validate
from bench.ingest.adapters import kestrel
from bench.ingest.filters import apply_filters
from bench.ingest.schema import CANONICAL_COLUMNS


def _ns(hours=0, minutes=0, seconds=0):
    return (hours * 3600 + minutes * 60 + seconds) * 1_000_000_000


def _fixture(tmp_path, *, tz="-07:00"):
    """One synthetic Kestrel month: duration[ns] columns, tz-aware timestamps."""
    n = 3
    submit = pl.Series([dt.datetime(2023, 8, 1, 12, 0, 0)] * n).dt.replace_time_zone(tz)
    frame = pl.DataFrame({
        "job_id": [1, 2, 3],
        "user_hash": ["u1", "u1", "u2"],
        "account_hash": ["a1", "a1", "a2"],
        "submit_script_hash": ["s1", "s1", "s2"],
        "partition": ["p1", "p1", "p2"],
        "qos": ["normal", "normal", "high"],
        "state_simple": ["TIMEOUT", "COMPLETED", "CANCELLED"],
        "submit_time": submit,
        "start_time": submit,
        "end_time": submit,
        "wallclock_req": pl.Series([_ns(hours=12), _ns(hours=1), _ns(minutes=30)]).cast(pl.Duration("ns")),
        "wallclock_used": pl.Series([_ns(hours=12), _ns(minutes=5), None]).cast(pl.Duration("ns")),
        "queue_wait": pl.Series([_ns(seconds=13), _ns(seconds=7), _ns(seconds=2)]).cast(pl.Duration("ns")),
        "processors_req": [128, 4, 1],
        "processors_used": [128, 4, 0],
        "nodes_req": [2, 1, 1],
        "nodes_used": [2, 1, 0],
        "memory_req": ["256G", "250000M", "500000G"],
        "consumed_energy_raw_joules": [1000.0, None, 0.0],
    })
    p = tmp_path / "kestrel_jobs_202308_0.parquet"
    frame.write_parquet(p)
    return str(p)


def test_shape_and_durations(tmp_path):
    df = kestrel.map_file(_fixture(tmp_path))
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df.height == 3
    # duration[ns] -> seconds
    r = df.filter(pl.col("job_id") == "1").row(0, named=True)
    assert r["timelimit_s"] == 43200.0
    assert r["elapsed_s"] == 43200.0
    assert r["queue_wait_s"] == 13.0


def test_timezone_converted_to_utc(tmp_path):
    # Submitted 12:00 at -07:00 -> 19:00 UTC.
    df = kestrel.map_file(_fixture(tmp_path, tz="-07:00"))
    ts = df.filter(pl.col("job_id") == "1")["submit_time"][0]
    assert ts.utcoffset() == dt.timedelta(0)          # stored as UTC
    assert (ts.hour, ts.minute) == (19, 0)
    assert df.schema["submit_time"] == pl.Datetime("us", "UTC")


def test_measured_energy_and_tier_invariant(tmp_path):
    df = kestrel.map_file(_fixture(tmp_path))
    tiers = dict(df.group_by("energy_tier").len().iter_rows())
    assert tiers.get("measured") == 2      # 1000 J and 0.0 J
    assert tiers.get("none") == 1          # the null reading
    r2 = df.filter(pl.col("job_id") == "2").row(0, named=True)
    assert r2["energy_j"] is None and r2["energy_tier"] == "none"


def test_hashes_pass_through_and_mem_units(tmp_path):
    df = kestrel.map_file(_fixture(tmp_path))
    r1 = df.filter(pl.col("job_id") == "1").row(0, named=True)
    assert r1["user_hash"] == "u1"          # already hashed -> untouched
    assert r1["script_hash"] == "s1"
    assert r1["mem_req"] == 256 * 1024      # 256G -> MB
    r2 = df.filter(pl.col("job_id") == "2").row(0, named=True)
    assert r2["mem_req"] == 250000.0        # 250000M -> MB


def test_never_ran_row_is_dropped(tmp_path):
    df = kestrel.map_file(_fixture(tmp_path))
    kept, stats = apply_filters(df)
    assert stats["dropped_no_elapsed"] == 1     # job 3 has no wallclock_used
    assert kept.height == 2


def test_validators_pass(tmp_path):
    df = kestrel.map_file(_fixture(tmp_path))
    assert validate.run_checks(df) == []


def test_month_of():
    assert kestrel.month_of("21913139/kestrel/kestrel_jobs_202308_0.parquet") == "2023-08"
