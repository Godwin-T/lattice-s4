"""The row-level finding contract shared by Peepalytics detectors.

Findings deliberately use flat columns rather than opaque JSON blobs.  This
makes them easy to inspect in Parquet and gives T5's evidence validator a
deterministic set of fields to check.
"""
from __future__ import annotations

import polars as pl

FINDING_COLUMNS: tuple[str, ...] = (
    "row_id", "job_id", "detector", "detector_version", "finding_type",
    "score", "submit_time", "state", "energy_j", "energy_tier",
    "evidence_elapsed_s", "evidence_timelimit_s", "evidence_ratio",
    "evidence_queue_wait_s", "evidence_cpus_req", "evidence_cpus_used",
    "evidence_failure_state", "evidence_script_hash", "evidence_prior_row_id",
    "evidence_gap_s", "evidence_runtime_delta_s", "evidence_same_script",
)


def validate_findings(findings: pl.DataFrame) -> None:
    """Raise if a detector output violates the minimum auditable contract."""
    missing = sorted(set(FINDING_COLUMNS) - set(findings.columns))
    if missing:
        raise ValueError(f"finding output is missing columns: {missing}")
    if findings.is_empty():
        return
    if findings["row_id"].null_count() or findings["job_id"].null_count():
        raise ValueError("every finding must identify its source row and job")
    if findings["row_id"].n_unique() != findings.height:
        raise ValueError("a detector must emit at most one finding per row")
    if findings["detector"].null_count() or findings["detector_version"].null_count():
        raise ValueError("every finding must identify its detector and version")
    if findings["energy_tier"].is_in(["measured", "modelled", "estimated", "none"]).not_().any():
        raise ValueError("energy_tier contains an unknown value")
    bad_score = findings.filter(
        pl.col("score").is_null() | (pl.col("score") < 0) | (pl.col("score") > 1)
    )
    if bad_score.height:
        raise ValueError("finding scores must be finite probabilities in [0, 1]")
