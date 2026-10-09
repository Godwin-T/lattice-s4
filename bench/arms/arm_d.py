"""Provisional Arm D: B1 LightGBM plus submit-time-safe domain features.

This thin adapter builds the ordinary B cache and the separate domain cache,
joins them on the authoritative ``row_id``, and reuses B1's tested fold,
control, and prediction machinery.  The metadata records the B1 provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..splits.eligibility import eligible_users
from . import arm_b1, common
from .domain_features import (
    DOMAIN_FEATURE_VERSION, FEATURE_COLUMNS as DOMAIN_FEATURE_COLUMNS,
    ensure_domain_features,
)
from .features import FEATURE_NAMES as B_FEATURE_NAMES, ensure_features

ARM = "D"
BASE_ARM = "B1"
TASKS = ("T1", "T2")


def _combined_cache(b_cache: Path, d_cache: Path, canonical_path: str | Path,
                    cache_dir: str | Path, *, rebuild: bool = False) -> Path:
    """Join B and domain caches once, with a provenance sidecar."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{Path(canonical_path).stem}.safe.d.features.parquet"
    meta_path = path.with_suffix(".meta.json")
    meta = {
        "canonical_sha256": common.file_sha256(canonical_path),
        "b_cache": str(b_cache), "domain_cache": str(d_cache),
        "b_feature_version": arm_b1.FEATURE_VERSION,
        "domain_feature_version": DOMAIN_FEATURE_VERSION,
    }
    if not rebuild and path.exists() and meta_path.exists():
        try:
            if json.loads(meta_path.read_text()) == meta:
                return path
        except (OSError, ValueError):
            pass

    b = pl.scan_parquet(b_cache)
    d = pl.scan_parquet(d_cache).select(["row_id", *DOMAIN_FEATURE_COLUMNS])
    b.join(d, on="row_id", how="inner", validate="1:1").sink_parquet(path)
    check = pl.scan_parquet(path).select("row_id").collect()
    if check.height != check["row_id"].n_unique():
        raise ValueError("combined D cache contains duplicate row_id values")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def run(*, manifest_path: str, canonical_path: str, outdir: str,
        tasks: tuple[str, ...] = TASKS, seed: int = 42, control_repeats: int = 3,
        control_folds: int | None = None, features_dir: str | None = None,
        domain_features_dir: str | None = None, rebuild_features: bool = False,
        folds_limit: int | None = None,
        max_chunk_rows: int = common.WINDOW_MAX_CHUNK_ROWS,
        params_dir: str = "params", num_threads: int = 2,
        window: str = common.RULE_SAFE) -> dict:
    """Run D using B1's frozen learner and domain-augmented features."""
    if window != common.RULE_SAFE:
        raise SystemExit("Arm D runs only under the 'safe' rule")
    manifest = common.load_manifest(manifest_path)
    dataset = manifest["dataset"]
    lf = common.with_row_id(pl.scan_parquet(canonical_path))
    users = eligible_users(lf)
    eligible = lf.filter(pl.col("user_hash").is_in(users["user_hash"].to_list()))
    cache_dir = Path(features_dir) if features_dir else Path(outdir).parent / "windows"
    d_cache_dir = (Path(domain_features_dir) if domain_features_dir
                   else cache_dir / "domain")
    b_cache = ensure_features(
        eligible, canonical_path, cache_dir, rebuild=rebuild_features,
        max_chunk_rows=max_chunk_rows, rule=window)
    d_cache = ensure_domain_features(
        eligible, canonical_path, d_cache_dir, dataset=dataset,
        rebuild=rebuild_features, max_chunk_rows=max_chunk_rows, rule=window)
    combined = _combined_cache(
        b_cache, d_cache, canonical_path, cache_dir, rebuild=rebuild_features)

    saved = {name: getattr(arm_b1, name) for name in (
        "ARM", "FEATURE_NAMES", "CATEGORICAL_FEATURES", "CATEGORY_INDICES",
        "FEATURE_VERSION", "ensure_features", "load_params")}
    original_load_params = arm_b1.load_params

    def load_b1_params(task, params_dir):
        old_arm = arm_b1.ARM
        arm_b1.ARM = BASE_ARM
        try:
            return original_load_params(task, params_dir)
        finally:
            arm_b1.ARM = old_arm

    try:
        arm_b1.ARM = ARM
        arm_b1.FEATURE_NAMES = (*B_FEATURE_NAMES, *DOMAIN_FEATURE_COLUMNS)
        arm_b1.CATEGORICAL_FEATURES = tuple(
            c for c in saved["CATEGORICAL_FEATURES"] if c in arm_b1.FEATURE_NAMES)
        arm_b1.CATEGORY_INDICES = [arm_b1.FEATURE_NAMES.index(c)
                                   for c in arm_b1.CATEGORICAL_FEATURES]
        arm_b1.FEATURE_VERSION = DOMAIN_FEATURE_VERSION
        arm_b1.ensure_features = lambda *args, **kwargs: combined
        arm_b1.load_params = load_b1_params
        meta = arm_b1.run(
            manifest_path=manifest_path, canonical_path=canonical_path,
            outdir=outdir, tasks=tasks, seed=seed,
            control_repeats=control_repeats, control_folds=control_folds,
            features_dir=str(cache_dir), rebuild_features=False,
            folds_limit=folds_limit, max_chunk_rows=max_chunk_rows,
            params_dir=params_dir, num_threads=num_threads, window=window)
    finally:
        for name, value in saved.items():
            setattr(arm_b1, name, value)

    for task_meta in [meta, *meta.get("tasks", {}).values()]:
        task_meta.update({
            "base_arm": BASE_ARM,
            "domain_features": str(d_cache),
            "domain_feature_version": DOMAIN_FEATURE_VERSION,
            "domain_feature_columns": list(DOMAIN_FEATURE_COLUMNS),
            "combined_features": str(combined),
        })
    for task in tasks:
        task_meta = meta["tasks"][task]
        (Path(outdir) / task_meta["run_id"] / "metadata.json").write_text(
            json.dumps(task_meta, indent=2), encoding="utf-8")
    return meta
