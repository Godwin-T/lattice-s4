# folds_manifest.md — The `folds.json` Contract

**Status:** design (not yet implemented).
**Implements:** `plan.md` §6 (Stage 2).
**Consumed by:** every arm, identically.

**Goal.** Freeze, in one file, exactly which rows are training and which are
testing for each fold, where the locked holdout is, and which 20 000 test jobs
the sample contains — so every arm answers the identical question.

---

## 1. Location and naming

```
data/manifests/<dataset>.folds.json     e.g. eagle.folds.json
```

Written **once** per dataset, then immutable. Re-running the generator on the
same canonical table must produce a byte-identical file (§7).

---

## 2. Field reference

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | format id, currently `"lattice24-benchmark/folds/1"` |
| `dataset` | string | `"eagle"` \| `"kestrel"` |
| `generated_utc` | string | ISO-8601 timestamp |
| `generator` | object | `{name, version, seed}` — the code that wrote this |
| `snapshot` | object | `{path, sha256, rows}` of the **raw** source |
| `canonical` | object | `{path, sha256, rows}` of the canonical table used |
| `rules` | object | how rows are keyed and labelled (see §3) |
| `guardrails` | object | the thresholds that gate every fold (see §4) |
| `locked` | object | the holdout months and their counts (see §5) |
| `folds` | array | the train/test split list (see §6) |
| `sample` | object | the frozen 20 000-job sample (see §5.2) |
| `checksums` | object | `{manifest_sha256}` |

### 2.1 `rules`

| Field | Meaning |
|---|---|
| `time_key` | always `"submit_time"` — ordering and month bucketing key |
| `month_derivation` | `"submit_time[0:7]"` |
| `reference_window` | `24` — the reference look-back (Arm A's published method). Sets the ≥25-job eligibility floor. **Not a cap**: an arm may look back further. |
| `min_history_jobs` | `25` — minimum jobs per user to be scoreable |
| `history_rule` | `"history_jobs_must_end_before_target_submit"` |
| `label` | `{task, positive, negative}` — for T1: `"state == 'TIMEOUT'"` vs any other kept state |

> **On `reference_window`.** 24 is *Arm A's* look-back, and it is what sets the
> ≥25-job eligibility floor — not a rule imposed on every arm. The spec has Arm
> B3 reading the last **32** jobs and Arm B using generic history aggregates
> rather than a fixed 24, so window length is an arm design choice. The one thing
> the harness does enforce: an arm with a longer look-back must still emit a
> prediction for **every** scored test row — it may not drop rows because its own
> history requirement is larger.

### 2.2 `guardrails`

Any fold failing these is dropped at generation time; if too few survive the
manifest is not written and generation refuses.

| Field | Value | Purpose |
|---|---|---|
| `min_months` | 6 | method's minimum-data floor |
| `min_train_rows` | 100 | fold must have enough training rows |
| `min_test_rows` | 50 | and enough test rows |
| `min_train_pos` | 5 | both sides must contain positives |
| `min_test_pos` | 5 | |
| `min_folds` | 3 | a single split cannot support a CI |

---

## 3. What a fold means

- **Expanding window:** `train_months` is always a prefix of the open months;
  `test_month` is the single month immediately after it.
- **Forward only:** `test_month` is never before or inside `train_months`.
- **Per dataset:** folds never mix Eagle and Kestrel.
- **Per fold, the arm gets:** the training rows, the test rows, the label rule,
  and the history rule. Nothing else about splitting is the arm's business.

---

## 4. The locked holdout

`locked.months` holds the **last six consecutive months** of the dataset's
timeline. These months:

- appear in **no** fold, on either side;
- are excluded from the sample;
- are scored **once**, after every arm is frozen.

---

## 5. The sample

`sample` describes the frozen 20 000 test jobs every arm is additionally scored
on (Arm C is compared **only** here). Per `plan.md` §14.1–14.2:

- `drawn_from`: `"open_test_months"` — never the locked holdout;
- `stratify_by`: `["state", "activity_quartile"]`;
- `activity_metric`: `"jobs_per_user_log_quartile"`;
- `strata`: per-stratum requested/actual counts, so a shortfall is visible;
- `job_ids`: the **authoritative** frozen list;
- `job_ids_sha256`: a hash of that list, so a consumer can verify it cheaply
  without diffing 20 000 strings.

If a stratum has fewer members than its quota, take all of them and record the
shortfall in `strata` — never substitute from another stratum.

---

## 6. Example manifest (Eagle, truncated)

```json
{
  "schema": "lattice24-benchmark/folds/1",
  "dataset": "eagle",
  "generated_utc": "2026-10-01T09:00:00Z",
  "generator": { "name": "make_folds.py", "version": "1", "seed": 42 },
  "snapshot": {
    "path": "data/eagle_data.parquet",
    "sha256": "…",
    "rows": 11030377
  },
  "canonical": {
    "path": "data/canonical/eagle.parquet",
    "sha256": "…",
    "rows": 10980000
  },
  "rules": {
    "time_key": "submit_time",
    "month_derivation": "submit_time[0:7]",
    "reference_window": 24,
    "min_history_jobs": 25,
    "history_rule": "history_jobs_must_end_before_target_submit",
    "label": {
      "task": "T1",
      "positive": "state == 'TIMEOUT'",
      "negative": "any other kept state"
    }
  },
  "guardrails": {
    "min_months": 6, "min_train_rows": 100, "min_test_rows": 50,
    "min_train_pos": 5, "min_test_pos": 5, "min_folds": 3
  },
  "locked": {
    "months": ["2022-09","2022-10","2022-11","2022-12","2023-01","2023-02"],
    "n_rows": 0,
    "n_pos": 0
  },
  "folds": [
    { "fold_id": 1,  "test_month": "2018-12", "train_months": ["2018-11"],
      "n_train": 0, "n_test": 0, "pos_train": 0, "pos_test": 0 },
    { "fold_id": 2,  "test_month": "2019-01", "train_months": ["2018-11","2018-12"],
      "n_train": 0, "n_test": 0, "pos_train": 0, "pos_test": 0 },
    { "fold_id": 45, "test_month": "2022-08",
      "train_months": ["2018-11","…","2022-07"],
      "n_train": 0, "n_test": 0, "pos_train": 0, "pos_test": 0 }
  ],
  "sample": {
    "n": 20000,
    "drawn_from": "open_test_months",
    "stratify_by": ["state", "activity_quartile"],
    "activity_metric": "jobs_per_user_log_quartile",
    "strata": [
      { "state": "TIMEOUT", "activity_quartile": "Q4", "quota": 0, "taken": 0 },
      { "state": "COMPLETED", "activity_quartile": "Q1", "quota": 0, "taken": 0 }
    ],
    "job_ids_sha256": "…",
    "job_ids": ["…"]
  },
  "checksums": { "manifest_sha256": "…" }
}
```

*(Counts are placeholders; they are filled when the manifest is generated.)*

---

## 7. Determinism requirements

Given the same canonical table and seed, generation must produce the **same
manifest digest** — the written file differs only in its `generated_utc` line.

- one fixed seed, recorded in `generator.seed`, drives the sample;
- months sorted ascending; folds numbered from 1 in test-month order;
- deterministic selection of sample members, ranked by a seeded hash of the
  `job_id`, so the choice cannot depend on row order or chunking;
- **no wall-clock content inside anything hashed**: `checksums.manifest_sha256`
  covers the whole manifest **except** `checksums` itself and `generated_utc`.

---

## 8. Validation a consumer must perform

Before an arm may run, the manifest must pass:

- [ ] `schema` is a known version.
- [ ] `snapshot.sha256` / `canonical.sha256` match the files actually present.
- [ ] Fold `test_month`s are unique and strictly increasing.
- [ ] For each fold, `test_month ∉ train_months` and every `train_month < test_month`.
- [ ] No fold month (either side) appears in `locked.months`.
- [ ] `sample.job_ids` all lie in open test months, none in `locked.months`.
- [ ] `sha256(sorted(sample.job_ids)) == sample.job_ids_sha256`.
- [ ] `checksums.manifest_sha256` recomputes over the manifest with
      `generated_utc` and `checksums` removed.
- [ ] `Σ fold.n_test` equals the number of rows in the open test months.
- [ ] Every remaining fold satisfies the guardrails in §2.2.

Any violation is a hard failure: the run stops rather than silently proceeding.

---

## 9. How an arm consumes it

For each fold, an arm must:

1. read the training rows and test rows **named by the manifest** (no re-splitting);
2. build its own features from those rows, honouring `history_rule`
   (`reference_window` is a floor, not a cap — use more history if your arm wants
   it);
3. fit on training rows only;
4. emit predictions in one shared shape:

```
{ fold_id, job_id, score, probability }
```

It must **not** add, drop or move rows between folds — including skipping test
rows its own history requirement cannot fill — nor change the label, fit on test
rows, use post-submit information, or choose its own operating point. The
evaluator assigns metrics; the arm only produces scores.

---

## 10. Out of scope

The canonical schema and per-dataset mapping (`canonical_table.md`), feature
design per arm, metric definitions, and energy-type accounting.
