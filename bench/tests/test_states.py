"""State normalisation."""
import polars as pl

from bench.ingest.states import normalise_state, state_expr


def test_normalise_state():
    assert normalise_state("CANCELLED by 1234") == "CANCELLED"
    assert normalise_state(" timeout ") == "TIMEOUT"
    assert normalise_state("OUT_OF_MEMORY") == "OUT_OF_MEMORY"
    assert normalise_state(None) == ""
    assert normalise_state("") == ""


def test_state_expr():
    df = pl.DataFrame({"s": ["CANCELLED by 99", "timeout", None, "  NODE_FAIL "]})
    out = df.with_columns(state_expr("s").alias("n"))["n"].to_list()
    assert out == ["CANCELLED", "TIMEOUT", None, "NODE_FAIL"]
