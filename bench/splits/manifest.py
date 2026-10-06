"""
Manifest assembly and writing (splits.md T7).

The manifest is the frozen contract every arm reads. It records what was split,
by which rules, with which guardrails, and a hash of everything except its own
checksum block.

Note on `snapshot`: the splitter never reads the raw sources, so it cannot hash
them itself. It records the canonical table in full and points at the ingest
audit for the raw provenance, rather than inventing a hash it did not compute.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import (
    HISTORY_RULE, LOCKED_MONTHS, MANIFEST_SCHEMA, MIN_FOLDS,
    MIN_HISTORY_JOBS, MIN_MONTH_ROWS,
    MIN_MONTHS, MIN_TEST_POS, MIN_TEST_ROWS, MIN_TRAIN_POS, MIN_TRAIN_ROWS,
    MONTH_DERIVATION, POSITIVE_STATE, REFERENCE_WINDOW, TIME_KEY,
)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _raw_sources(canonical_path: Path) -> list[str]:
    """Read the ingest audit beside the canonical table, if it is there."""
    audit = canonical_path.with_suffix("").with_suffix(".audit.json")
    if not audit.exists():
        # `foo.parquet` -> `foo.audit.json`
        audit = canonical_path.parent / (canonical_path.stem + ".audit.json")
    try:
        return json.loads(audit.read_text(encoding="utf-8")).get("sources", [])
    except (OSError, ValueError):
        return []


def build_manifest(*, dataset: str, canonical_path: str, canonical_rows: int,
                   months: list[str], dropped_months: list[tuple[str, int]],
                   eligible: dict, folds: list[dict], locked: list[str],
                   open_months: list[str], sample: dict, seed: int) -> dict:
    """Assemble the manifest body (without its checksum block)."""
    cpath = Path(canonical_path)
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset": dataset,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"name": "bench.splits", "version": "1", "seed": seed},
        "snapshot": {
            "sources": _raw_sources(cpath),
            "note": "raw provenance; hashes are in the canonical table's ingest audit",
        },
        "canonical": {
            "path": str(cpath),
            "sha256": file_sha256(cpath),
            "rows": canonical_rows,
        },
        "rules": {
            "time_key": TIME_KEY,
            "month_derivation": MONTH_DERIVATION,
            "min_month_rows": MIN_MONTH_ROWS,
            "months_dropped_below_floor": [{"month": m, "rows": n}
                                           for m, n in dropped_months],
            "reference_window": REFERENCE_WINDOW,
            "min_history_jobs": MIN_HISTORY_JOBS,
            "history_rule": HISTORY_RULE,
            "locked_months": LOCKED_MONTHS,
            "label": {
                "task": "T1",
                "positive": f"state == '{POSITIVE_STATE}'",
                "negative": "any other kept state (DEADLINE included)",
            },
        },
        "guardrails": {
            "min_months": MIN_MONTHS,
            "min_train_rows": MIN_TRAIN_ROWS,
            "min_test_rows": MIN_TEST_ROWS,
            "min_train_pos": MIN_TRAIN_POS,
            "min_test_pos": MIN_TEST_POS,
            "min_folds": MIN_FOLDS,
        },
        "eligible_users": eligible,
        "timeline": {"months": len(months), "open_months": len(open_months),
                     "first_month": months[0], "last_month": months[-1]},
        "locked": {
            "months": locked,
            "n_rows": sum(f["n_test"] for f in folds if f["test_month"] in locked),
        },
        "folds": folds,
        "sample": sample,
    }


def write_manifest(manifest: dict, outdir: str | Path, dataset: str) -> tuple[Path, str]:
    """
    Write `<outdir>/<dataset>.folds.json` and return (path, sha256).

    The checksum covers the whole manifest except the checksum block itself,
    so it can be recomputed by any consumer.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset}.folds.json"

    # The checksum covers everything except the checksum block itself and the
    # generation timestamp, so two runs over the same input produce the same
    # digest (folds_manifest.md section 7: no wall-clock content inside
    # anything that is hashed).
    body = dict(manifest)
    body.pop("checksums", None)
    body.pop("generated_utc", None)
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    body["checksums"] = {"manifest_sha256": digest}

    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    return path, digest
