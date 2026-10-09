# Peepalytics Domain Features Plan

## Objective

Build the controllable portion of Arm D while B2/B3 tuning continues:

```text
D = best available B learner + submit-time-safe domain features + detector evidence
```

This plan deliberately excludes the T4 triage layer, Jev integration, and human
finding labels. Those are separate workstreams and are not required to build or
evaluate the domain feature layer.

The immediate goal is to determine whether HPC-specific domain features add
predictive value beyond the shared B feature layer.

The primary comparison will be:

```text
D versus B
```

The secondary comparison will be:

```text
D versus Arm A
```

The comparison must use the same frozen folds, same rows, same task, and same
evaluation harness.

## Current baseline

Already available:

- B1 LightGBM shared feature layer.
- B2 MLP and B3 GRU implementations.
- Frozen Eagle and Kestrel temporal manifests.
- Row-authoritative `row_id` values.
- Detector finding contract.
- D1 timeout detector.
- D2 Kestrel restart-chain detector.
- D3 repeated-failure detector.
- D4 Kestrel duplicate-workload detector.
- D5 wallclock underuse detector.
- Energy values and energy tiers in the canonical tables.
- Paired comparison and bootstrap evaluation machinery.

Not yet available:

- A D-specific feature cache.
- Submit-time-safe features derived from detector history.
- Account change-point features.
- Historical p95 request-habit features.
- Domain-feature leakage tests.
- A trained and evaluated D learner.
- A D-versus-B ablation report.

## Scope by dataset

Feature availability must be explicit. The system must not silently treat a
missing field as zero or pretend that every dataset supports every detector.

| Feature or detector | Eagle Parquet | Kestrel | Eagle JSONL |
|---|---:|---:|---:|
| Timeout history | Yes | Yes | Yes |
| Failure history | Yes | Yes | Yes |
| Wallclock request habit | Yes | Yes | Yes |
| Queue-wait history | No/mostly null | Yes | Yes |
| Script-hash restart evidence | No | Yes | Yes |
| Duplicate workloads | No | Yes | Yes/partial |
| CPU requested versus used | No reliable used CPU | Available but needs validation | Available |
| Memory requested versus used | No used memory | No reliable used memory | No reliable used memory |
| GPU requested versus used | Request only | Unavailable | Unavailable |
| Array position | No | No | Partial |
| Dependencies | No | No | No |
| Measured energy | No in the current primary table | Yes, partial row coverage | Partial |

The initial D feature experiment should run on Kestrel first because Kestrel
supports script-hash detectors and measured energy. Eagle should receive only
the capabilities its fields support.

## Feature design rules

Every feature must satisfy all of these rules:

1. It is keyed by the canonical `row_id`.
2. It uses only information available at or before the target's submission.
3. It has a documented source field or detector.
4. It has an explicit missing-value meaning.
5. It records its feature version.
6. It has a unit and a bounded interpretation where applicable.
7. It is reproducible from the canonical snapshot and manifest.
8. It can be traced back to source rows for evidence.

The feature builder must preserve one output row per canonical input row. Missing
history should produce nulls or explicit availability flags, not dropped rows.

## Proposed feature groups

### 1. Request-habit features

These compare the current request with the user's historical behaviour.

Initial features:

- `hist_elapsed_p95_s`
- `hist_elapsed_p50_s`
- `hist_elapsed_mean_s`
- `request_vs_elapsed_p95`
- `request_excess_over_elapsed_p95_s`
- `request_exceeds_elapsed_p95`
- `hist_timelimit_p95_s`
- `timelimit_vs_hist_timelimit_p95`

Definitions:

```text
request_vs_elapsed_p95 = timelimit_s / hist_elapsed_p95_s
request_excess_over_elapsed_p95_s = timelimit_s - hist_elapsed_p95_s
```

Only historical jobs ending strictly before the target submission may enter the
quantiles. A user with no legal history receives null values and an availability
indicator.

### 2. Restart-chain features

Derived from the prior-job side of D2, not from future outcomes.

Initial features:

- `prior_timeout_chain`
- `prior_timeout_same_script`
- `prior_timeout_chain_gap_s`
- `prior_timeout_row_id`
- `prior_timeout_energy_j`
- `prior_timeout_chain_count`
- `prior_timeout_chain_energy_sum_j`

For a target job, the legal feature is whether a prior same-user timeout ended
within two hours and used the same requested time limit. The target may use a
matching prior script hash because that hash existed before submission.

The feature builder must not use the target's future successor job to create a
feature for the target.

### 3. Duplicate-workload history

Derived from prior D4 matches.

Initial features:

- `prior_duplicate_workload`
- `prior_duplicate_count`
- `prior_duplicate_gap_s`
- `prior_duplicate_runtime_delta_s`
- `prior_duplicate_energy_sum_j`
- `prior_duplicate_same_script`
- `prior_duplicate_same_shape`

The resource shape should remain explicit:

```text
partition + qos + cpus_req + mem_req + gpus_req + nodes_req + timelimit_s
```

The D4 detector uses a 24-hour default window and a 10% runtime-difference
tolerance. These are configuration values, not hidden constants, and must be
recorded in feature metadata.

### 4. Failure-loop history

Extend the existing historical failure-rate features with recency and severity.

Initial features:

- `failures_last_5`
- `failures_last_10`
- `timeouts_last_5`
- `timeouts_last_10`
- `time_since_last_failure_s`
- `time_since_last_timeout_s`
- `same_script_failure_count`
- `same_account_failure_rate`
- `same_script_timeout_count`

All counts must be computed from rows before submission. The current target row
must not contribute to its own history.

### 5. Wallclock-underuse history

Extend D5 beyond the current rule-level finding.

Initial features:

- `current_wallclock_utilization`
- `current_wallclock_unused_fraction`
- `hist_wallclock_utilization_mean`
- `hist_wallclock_utilization_p25`
- `hist_wallclock_underuse_rate`
- `hist_wallclock_underuse_count`

The current row's elapsed time is available only after execution. Therefore:

- current-row wallclock underuse is suitable for retrospective findings;
- only historical wallclock underuse may be used as a pre-submit model feature.

The learner must not receive `elapsed_s`, current-row utilization, or current-row
energy when predicting the current job.

### 6. Queue and partition context

Use queue-wait and partition information only where available.

Initial features:

- `hist_queue_wait_mean_s`
- `hist_queue_wait_p95_s`
- `hist_queue_wait_max_s`
- `partition_queue_wait_mean_s`
- `partition_queue_wait_p95_s`
- `partition_job_count_prior`
- `user_partition_share_prior`
- `partition_load_bucket`

For Eagle Parquet, queue-wait fields are unavailable and must remain null. The
feature metadata must state that the feature is unavailable rather than treating
absence as zero queue pressure.

### 7. Account change-point features

Begin with simple, explainable rolling changes before adding a more complex
change-point algorithm.

Initial features:

- `account_failure_rate_recent`
- `account_failure_rate_previous`
- `account_failure_rate_delta`
- `account_timeout_rate_recent`
- `account_timeout_rate_previous`
- `account_timeout_rate_delta`
- `account_wallclock_underuse_recent`
- `account_wallclock_underuse_previous`
- `account_change_point_flag`

Recommended initial windows:

- recent window: last 10 legal jobs;
- comparison window: preceding 50 legal jobs;
- minimum history: 20 total legal jobs before emitting a change-point flag.

The change-point flag should be deterministic and explainable. A first version
can use a documented threshold on the rate difference. PELT or Isolation Forest
can be evaluated later as a separate experiment.

## Implementation phases

### Phase 1 — Stabilise the detector contract

Files:

- `bench/detectors/contract.py`
- `bench/detectors/basic.py`
- `bench/tests/test_detectors.py`

Tasks:

- Keep D1, D2, D3, D4, and D5 outputs flat and Parquet-friendly.
- Keep detector versions explicit.
- Keep prior evidence row IDs in pair findings.
- Add configuration metadata for D2 and D4 thresholds.
- Ensure empty detector outputs preserve the full schema.
- Add capability checks for required fields.

Acceptance:

- Every finding has a `row_id`.
- Every finding has a detector and version.
- Every finding has an energy tier.
- Pair findings identify their prior evidence row.
- Missing Kestrel-only fields fail loudly.
- Detector tests pass on synthetic data.

### Phase 2 — Create the D feature module

Create:

```text
bench/arms/domain_features.py
```

Responsibilities:

- Build submit-time-safe domain features from canonical rows.
- Reuse the same chronological ordering as `features.py`.
- Preserve `row_id` and one row per canonical input row.
- Add a feature availability map.
- Write a versioned Parquet cache.
- Record dataset capabilities and configuration in metadata.

Proposed public functions:

```python
build_domain_features(frame, *, dataset, rule="safe", config=None)
ensure_domain_features(canonical_path, outdir, *, dataset, manifest, rebuild=False)
validate_domain_features(features, canonical_frame)
```

Do not mutate the existing B feature cache. D needs its own cache so B remains
reproducible and the D-versus-B ablation remains interpretable.

### Phase 3 — Implement request habit and historical failure features

Implement the dataset-independent features first:

- historical elapsed quantiles;
- request-habit ratios;
- recent timeout/failure counts;
- time since last failure/timeout;
- same-script historical failure features.

Acceptance:

- Features are null when history is insufficient.
- Features never use target outcomes.
- Quantile calculations are deterministic.
- A toy chronological example reproduces exact expected values.

### Phase 4 — Integrate Kestrel D2 and D4 history

Add legal prior-history features from the implemented detectors:

- prior timeout chain;
- same-script restart;
- prior duplicate workload;
- prior duplicate count;
- prior duplicate runtime delta;
- prior duplicate energy.

Acceptance:

- Kestrel produces non-null D2/D4 features.
- Eagle Parquet marks these features unavailable.
- A target cannot receive evidence from a later row.
- The prior evidence `row_id` is traceable to the canonical table.

### Phase 5 — Add queue/partition context

Implement only the fields supported by each dataset:

- historical queue wait;
- partition queue-wait aggregates;
- prior partition activity;
- user partition share.

Acceptance:

- Kestrel and Eagle JSONL contain the supported queue features.
- Eagle Parquet contains nulls plus availability metadata.
- No queue feature is filled with zero merely because the source is missing.

### Phase 6 — Add account change-point features

Implement the simple two-window rate comparison first.

Acceptance:

- Minimum-history rule is enforced.
- Recent and previous windows are explicitly recorded.
- The change-point flag can be reproduced from the feature values.
- Future jobs cannot affect earlier change-point values.

### Phase 7 — D learner integration

Create a D arm or D-specific runner that:

1. loads the shared B feature cache;
2. loads the domain feature cache;
3. joins on `row_id`;
4. checks for duplicate or missing row IDs;
5. trains the selected B architecture;
6. writes predictions and metadata;
7. records domain feature version and capability metadata.

The first D model should use B1 as the provisional base learner if B2/B3 tuning
has not completed. Once B2/B3 results exist, rerun D with the selected best B
learner rather than silently mixing architectures.

## Leakage test plan

Every feature group needs at least one direct leakage test.

### Future-row mutation test

1. Build features for a chronological toy table.
2. Change or remove rows after the target submission.
3. Rebuild features.
4. Assert that the target's features are unchanged.

### Target-outcome test

1. Change the target row's final state, elapsed time, or energy.
2. Rebuild features.
3. Assert that pre-submit features are unchanged.

### Same-timestamp test

Rows ending exactly at the target submission timestamp must not be treated as
legal prior history. Use the existing strict `submit_time - 1 microsecond` rule.

### Row alignment test

Assert that:

- domain features have unique `row_id`s;
- every scoreable B row has exactly one D feature row;
- no feature row maps to a different job ID;
- manifests and feature metadata carry the same checksum.

## D-versus-B evaluation

For each dataset and task:

1. Run B using the frozen B parameters.
2. Run D using the same folds and same rows.
3. Evaluate both with the same evaluator.
4. Compare D minus B using the paired bootstrap comparison.
5. Report AUC, PR-AUC, and energy metrics where available.
6. Report the number of rows and months common to both arms.
7. Report feature availability by dataset.

The primary domain-feature claim is:

> D's domain features improve the best B learner.

If the confidence interval includes zero, report the domain-feature effect as
inconclusive. Do not claim that the features helped merely because D has a
higher point estimate.

## Evidence outputs

Every D prediction or report finding should be traceable to:

```text
prediction/finding
  -> domain feature name
  -> detector/version
  -> source row_id(s)
  -> source fields
  -> feature configuration
  -> canonical checksum
  -> manifest checksum
```

T5's full report validator is out of scope for this phase, but the feature and
detector outputs should already contain enough evidence for it to be built later.

## Definition of done for this phase

This feature phase is complete when:

- D1, D2, D3, D4, and D5 findings have stable contracts.
- A versioned `domain_features` cache exists.
- Request-habit features exist on Eagle and Kestrel.
- D2 and D4 prior-history features exist on Kestrel.
- Failure, timeout, queue, and partition history features are implemented where
  supported.
- Change-point features have deterministic minimum-history behaviour.
- Leakage tests pass.
- Capability metadata is emitted.
- A D runner can join B and domain features by `row_id`.
- A D-versus-B evaluation can be run without T4 triage.

## Immediate next coding tasks

1. Add `bench/arms/domain_features.py`.
2. Implement historical elapsed p95 and request-habit features.
3. Implement prior timeout-chain and duplicate-history joins for Kestrel.
4. Add domain-feature leakage tests.
5. Add the versioned domain-feature cache writer.
6. Add domain-feature columns to a provisional D1/B1 runner.
7. Run a small Kestrel smoke experiment.
8. Run the full D-versus-B evaluation after B2/B3 results are available.

T4 triage, Jev, and hand-labelled findings remain explicitly deferred.
