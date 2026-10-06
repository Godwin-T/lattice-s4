"""Eagle 11M adapter: salted hashing, seconds, no energy, sensitive columns unread."""
import datetime as dt

import polars as pl
import pytest

from bench.ingest import validate
from bench.ingest.adapters import eagle_parquet
from bench.ingest.filters import apply_filters
from bench.ingest.hashing import salted_hash
from bench.ingest.schema import CANONICAL_COLUMNS, FORBIDDEN_COLUMNS

SALT = "test-salt"


def _fixture(tmp_path):
    n = 4
    ts = pl.Series([dt.datetime(2018, 11, 14, 13, 26, 16)] * n)   # naive
    frame = pl.DataFrame({
        "job_id": [1, 2, 3, 4],
        "user": ["user0001", "user0001", "user0002", "user0002"],
        "account": ["account0001", "account0001", "account0002", "account0002"],
        "partition": ["gpu", "gpu", "standard", "standard"],
        "qos": ["normal", "normal", "high", "high"],
        "state": ["TIMEOUT", "COMPLETED", "CANCELLED", "RUNNING"],
        "submit_time": ts, "start_time": ts, "end_time": ts,
        "wallclock_req": [36000.0, 300.0, 600.0, 600.0],   # seconds
        "run_time": [36000.0, 116.0, None, 10.0],          # None => never ran
        "processors_req": [360, 1, 4, 4],
        "nodes_req": [180, 1, 1, 1],
        "gpus_req": [4, 0, 0, 0],
        "mem_req": [0.0, 0.0, 0.0, 0.0],
        # Sensitive columns: present in the file, must never reach the table.
        "name": ["name0001", "name0002", "name0003", "name0004"],
        "work_dir": ["/home/a", "/home/b", "/home/c", "/home/d"],
        "submit_line": ["sbatch x", "sbatch y", "sbatch z", "sbatch w"],
    })
    p = tmp_path / "eagle_data.parquet"
    frame.write_parquet(p)
    return str(p)


def _mapped(tmp_path, salt=SALT):
    return eagle_parquet.map_frame(
        pl.read_parquet(_fixture(tmp_path)), "eagle_data.parquet", salt)


def test_shape_and_sensitive_columns_dropped(tmp_path):
    df = _mapped(tmp_path)
    assert list(df.columns) == CANONICAL_COLUMNS
    assert not (FORBIDDEN_COLUMNS & set(df.columns))
    # The raw identifier columns must not survive conform().
    assert "user" not in df.columns and "account" not in df.columns


def test_identifiers_are_salted_hashes(tmp_path):
    df = _mapped(tmp_path)
    r = df.filter(pl.col("job_id") == "1").row(0, named=True)
    assert r["user_hash"] != "user0001"                    # not the raw value
    assert r["user_hash"] == salted_hash("user0001", SALT)  # exact hash
    assert r["account_hash"] == salted_hash("account0001", SALT)
    assert len(r["user_hash"]) == 16                       # 16 hex chars


def test_hashing_is_deterministic_across_calls(tmp_path):
    a = _mapped(tmp_path)["user_hash"].to_list()
    b = _mapped(tmp_path)["user_hash"].to_list()
    assert a == b


def test_different_salt_gives_different_hash(tmp_path):
    a = _mapped(tmp_path, salt="salt-one")["user_hash"].to_list()
    b = _mapped(tmp_path, salt="salt-two")["user_hash"].to_list()
    assert a != b


def test_missing_salt_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _mapped(tmp_path, salt=None)


def test_durations_are_seconds_and_timestamps_utc(tmp_path):
    df = _mapped(tmp_path)
    r = df.filter(pl.col("job_id") == "1").row(0, named=True)
    assert r["timelimit_s"] == 36000.0
    assert r["elapsed_s"] == 36000.0
    assert r["state"] == "TIMEOUT"
    assert df.schema["submit_time"] == pl.Datetime("us", "UTC")
    ts = df["submit_time"][0]
    assert ts.utcoffset() == dt.timedelta(0)
    assert (ts.hour, ts.minute) == (13, 26)      # no shift for naive input


def test_energy_is_absent_everywhere(tmp_path):
    df = _mapped(tmp_path)
    assert set(df["energy_tier"].unique().to_list()) == {"none"}
    assert df["energy_j"].null_count() == df.height


def test_filters_drop_running_and_never_ran(tmp_path):
    kept, stats = apply_filters(_mapped(tmp_path))
    assert stats["dropped_non_terminal_state"] == 1   # RUNNING
    assert stats["dropped_no_elapsed"] == 1           # the None run_time
    assert kept.height == 2


def test_validators_pass(tmp_path):
    kept, _ = apply_filters(_mapped(tmp_path))        # validators run post-filter
    assert validate.run_checks(kept) == []
    assert validate.lazy_checks(kept.lazy()) == []


def test_iter_canonical_chunks(tmp_path):
    chunks = list(eagle_parquet.iter_canonical(
        [_fixture(tmp_path)], salt=SALT, chunk_rows=2))
    assert [c.height for c in chunks] == [2, 2]
    assert all(list(c.columns) == CANONICAL_COLUMNS for c in chunks)
    # Hashing is per-value, so chunk boundaries cannot change the result.
    assert chunks[0]["user_hash"].to_list() == \
        _mapped(tmp_path)["user_hash"].to_list()[:2]
