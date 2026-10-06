"""Duration parsing: the four encodings and null handling."""
import polars as pl
import pytest

from bench.ingest.durations import duration_expr, parse_iso8601, parse_slurm


def test_parse_iso8601():
    assert parse_iso8601("P0DT0H30M0S") == 1800.0
    assert parse_iso8601("P2DT0H0M0S") == 172800.0
    assert parse_iso8601("P1DT12H10M17S") == 86400 + 12 * 3600 + 10 * 60 + 17
    assert parse_iso8601("P0DT0H0M7S") == 7.0


def test_parse_iso8601_rejects_junk():
    assert parse_iso8601("nonsense") is None
    assert parse_iso8601("") is None
    assert parse_iso8601(None) is None


def test_parse_slurm():
    assert parse_slurm("04:00:00") == 14400.0
    assert parse_slurm("1-00:00:00") == 86400.0
    assert parse_slurm("30:00") == 1800.0
    assert parse_slurm("2-12:30:15") == 2 * 86400 + 12 * 3600 + 30 * 60 + 15
    assert parse_slurm("bogus") is None


def test_expr_iso8601_keeps_null_null():
    df = pl.DataFrame({"d": ["P0DT0H30M0S", None, "P1DT0H0M0S"]})
    out = df.with_columns(duration_expr("d", "iso8601").alias("s"))["s"].to_list()
    assert out == [1800.0, None, 86400.0]


def test_expr_slurm_and_seconds():
    df = pl.DataFrame({"d": ["04:00:00", "1-00:00:00"]})
    assert df.with_columns(duration_expr("d", "slurm").alias("s"))["s"].to_list() == [14400.0, 86400.0]
    df2 = pl.DataFrame({"d": [3600.0, None]})
    assert df2.with_columns(duration_expr("d", "seconds").alias("s"))["s"].to_list() == [3600.0, None]


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        duration_expr("d", "fortnights")
