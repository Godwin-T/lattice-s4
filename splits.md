# splits.md — Split Generator: Plan, Decisions and Tasks

**Status:** planned, not yet built.
**Implements:** `plan.md` §6 (Stage 2) and the contract in `folds_manifest.md`.
**Depends on:** the canonical tables produced by ingest (`canonical_table.md`).

**What it is.** One small program that reads a canonical job table and writes one
file: `data/manifests/<dataset>.folds.json`. That file freezes *which rows are
training and which are testing* for every arm. It contains no model, no features
and no metrics — only the split.

**Why it exists.** The benchmark's claim is "approach X beats approach Y". That
is only meaningful if every approach was asked the identical question on the
identical rows. The split is what makes the question identical.

---

## 1. The contract

| | |
|---|---|
| **Reads** | exactly one canonical table (`data/canonical/<dataset>.parquet`) |
| **Writes** | `data/manifests/<dataset>.folds.json`, plus an `.audit.json` |
| **Never reads** | the raw sources, any arm, any metric |
| **Guarantee** | same input ⇒ byte-identical output (seeded; no wall-clock inside anything hashed) |

The splitter never learns which machine it is. It sees a canonical table and
nothing else — the same rule that governs ingest.

---

## 2. Decisions (all settled)

### 2.1 New decisions made for this component

| # | Decision | Why |
|---|---|---|
| **S1** | **Drop months with fewer than 1,000 rows** from the timeline, *before* choosing the locked six. | Kestrel's trailing `2026-01` bucket holds only 180 rows — an artefact of a few jobs submitted just after New Year. Left in, it would consume one of the six holdout slots and shrink the final exam to "five real months plus one near-empty one". The floor removes only artefacts: the smallest *real* month in any dataset is 28,096 rows. Rows are **not deleted** — they stay in the canonical table, they are simply not part of the fold timeline. |
| **S2** | **Sample quotas are proportional, with a floor of 25 per stratum**, recording `quota` vs `taken`. | Pure proportional keeps the real mix but can starve a rare stratum (`TIMEOUT` from quiet users) down to a handful of rows; equal allocation would over-represent rare classes and stop resembling the data. Proportional-plus-floor keeps the mix while guaranteeing every stratum is large enough to say something. `taken < quota` is recorded, never silently topped up from elsewhere. |
| **S3** | **Activity quartiles are computed once over the whole dataset**, on `log1p(jobs_per_user)`. | Computed per test month, the same user could sit in Q4 one month and Q2 the next, so the sample's composition would drift as the timeline advances. Whole-dataset quartiles are fixed before any results exist. The log tames a heavy tail (Eagle 11M: median 443 jobs/user, busiest user 654,013) so one stratum does not absorb nearly all the jobs. |
| **S4** | **Ship the eligible-user set as a count + hash**, not as a list of ids. | The set is ~1,000 ids per dataset and fully recomputable from the canonical table. A hash lets any consumer verify it reproduced the same set, without the manifest carrying a large list. |

### 2.2 Decisions inherited from `plan.md` that bind this component

| # | Decision |
|---|---|
| §14.1 | The 20,000-job sample is drawn from **open test months only** — never the locked holdout. |
| §14.3 | The four energy-less Kestrel months stay in the timeline; they can be scored for AUC/recall, never for energy. |
| §14.4 | Everything is keyed and ordered by **`submit_time`** (not `end_time`); a job's history may contain only jobs that **ended** before it was submitted. |
| §14.5 | The 3-month Eagle set is **exempt from folds** — reproduction only. |
| §14.6 | T1's positive class is **`TIMEOUT` only**; `DEADLINE` is a negative, and its flagged count is reported separately. |
| — | `reference_window` = 24 is a **floor, not a cap** (Arm B3 looks back 32). |
| — | An arm may never drop test rows because its own history requirement is larger. |

### 2.3 Guardrails (from `folds_manifest.md` §2.2)

| Field | Value | Purpose |
|---|---|---|
| `min_months` | 6 | the method's minimum-data floor |
| `min_train_rows` | 100 | a fold must have enough training rows |
| `min_test_rows` | 50 | and enough test rows |
| `min_train_pos` / `min_test_pos` | 5 / 5 | both sides must contain positives |
| `min_folds` | 3 | a single split cannot support a confidence interval |

Refusal — rather than a degraded result — if fewer than 3 folds survive.

---

## 3. Inputs, as they actually are

Measured from the three canonical tables:

| Dataset | Rows | Submit-months | Users | Users ≥25 jobs | After the 1,000-row floor |
|---|---|---|---|---|---|
| `eagle_parquet` | 11,014,689 | 52 | 936 | 777 | 52 months (smallest 28,096) |
| `kestrel` | 9,320,707 | **30** | 1,087 | 871 | **29 months** — the 180-row `2026-01` stub is dropped |
| `eagle_jsonl` | 181,503 | 2 | 194 | 120 | exempt from folds (§14.5) |

Two facts worth carrying forward:

1. **Kestrel has 30 submit-months, not 29.** A monthly file contains jobs whose
   `submit_time` falls outside that file's nominal month, so file layout and the
   submit-month timeline disagree. The splitter uses `submit_time`, which is what
   the no-leakage rule requires.
2. **`eagle_jsonl` currently covers one file** (`anon_jobs_2019-12.json`), so it
   holds 2 submit-months rather than the 3 reproduction months. It is exempt from
   folds, so this does not block the splitter — but the table is incomplete
   (see T11).

### Expected output, before any code runs

| Dataset | Locked holdout | Open months | Folds |
|---|---|---|---|
| `eagle_parquet` | 2022-09 → 2023-02 | 46 | ~45 |
| `kestrel` | 2025-07 → 2025-12 | 23 | ~22 |
| `eagle_jsonl` | — | — | refuses (1 fold < 3) |

These become the **golden tests**: if the implementation produces different
counts, something is wrong.

---

## 4. The algorithm

```text
INPUT : one canonical table
OUTPUT: folds.json

1. month(row) = submit_time[0:7]                      # "YYYY-MM"
2. timeline   = months with >= 1000 rows              # (S1)
3. eligible   = users with >= 25 jobs                 # min_history_jobs
4. keep only rows whose user is eligible
5. locked = timeline[-6:]
   open   = timeline[:-6]
6. folds = []
   for i in 1 .. len(open)-1:
       test_month   = open[i]
       train_months = open[:i]                        # expanding, never sliding
       skip if train rows < 100 or test rows < 50
       skip if train positives < 5 or test positives < 5
       folds.append(...)
7. REFUSE if len(folds) < 3
8. sample = 20,000 rows drawn from the open test months, stratified by
            (state x activity quartile), proportional with a floor   (S2, S3)
9. validate the manifest against its own rules, then write it + the audit
```

### 4.1 The sample, precisely

- **Population:** rows in the open *test* months, after the eligibility filter.
- **Strata:** `state` × `activity_quartile` (4 groups).
- **Activity quartile:** computed once over eligible users across the whole
  dataset, on `log1p(jobs_per_user)` (S3).
- **Allocation:** proportional to stratum size, floored at 25 rows (S2).
- **Shortfall:** if a stratum has fewer members than its quota, take all of them
  and record `taken < quota`.
- **Frozen:** seeded; sorted deterministically before selection; ids written into
  the manifest together with a hash of the list.

---

## 5. What the manifest contains

Exactly what `folds_manifest.md` §2 specifies. In brief:

`schema` · `dataset` · `generated_utc` · `generator{name,version,seed}` ·
`snapshot{path,sha256,rows}` · `canonical{path,sha256,rows}` ·
`rules{time_key, month_derivation, reference_window, min_history_jobs, history_rule, label}` ·
`guardrails{...}` · `locked{months,n_rows,n_pos}` ·
`folds[{fold_id, test_month, train_months, n_train, n_test, pos_train, pos_test}]` ·
`sample{n, drawn_from, stratify_by, activity_metric, strata[{state,activity_quartile,quota,taken}], job_ids_sha256, job_ids}` ·
`checksums{manifest_sha256}`

Plus, from S4, an `eligible_users{count, sha256}` block.

Per-fold **counts** are recorded even though the split is deterministic: they are
how a reader detects drift if the canonical table is ever rebuilt.

---

## 6. Task breakdown

Each task is small enough to finish and verify on its own. They are ordered, and
each depends only on the ones above it.

**Status: T1–T10 complete.** Verified numbers and manifest hashes are recorded in
`bench/README.md`. T11 remains open.

### T1 — Scaffold the package

- [ ] Create `bench/splits/` with `__init__.py`, empty modules, and a defaults
      block holding the guardrails and the new constants (`min_month_rows=1000`,
      `sample_n=20000`, `sample_floor=25`, `activity_quartiles=4`).
- [ ] **Verify:** `python -c "import bench.splits"` succeeds.

### T2 — `months.py`

- [ ] Derive `month` from `submit_time`; return the ordered timeline of months
      meeting the 1,000-row floor (S1).
- [ ] **Verify:** `eagle_parquet` → 52 months, first `2018-11`, last `2023-02`;
      `kestrel` → 29 months, last `2025-12` (the 180-row `2026-01` absent).

### T3 — `eligibility.py`

- [ ] Return the set of users with ≥25 jobs, plus `count` and a hash of the
      sorted id list (S4).
- [ ] **Verify:** `eagle_parquet` → 777 users; `kestrel` → 871; `eagle_jsonl` → 120.

### T4 — `folds.py`

- [ ] Build expanding forward-chained folds over the open months, apply the
      guardrails, and refuse when fewer than 3 folds survive.
- [ ] **Verify:** `eagle_parquet` → 45 folds, test months `2018-12` … `2022-08`;
      `kestrel` → 22 folds, test months `2023-09` … `2025-06`;
      a synthetic 3-month table → clean refusal.

### T5 — `sample.py`

- [ ] Stratified sample of 20,000 from the open test months: `state × activity
      quartile`, proportional with a floor of 25, seeded, with `quota`/`taken`
      recorded (S2, S3).
- [ ] **Verify:** the total is exactly 20,000 (or every stratum is exhausted);
      every sampled id lies in an open test month; two runs are identical.

### T6 — `validate.py`

- [ ] Implement the `folds_manifest.md` §8 checks and run them **before** writing.
- [ ] **Verify:** a deliberately corrupted manifest (a test month inside its own
      train set; a locked month inside a fold) is rejected.

### T7 — Manifest writer

- [ ] Assemble the manifest, compute the snapshot / canonical / manifest hashes,
      and write the `.audit.json` companion.
- [ ] **Verify:** the written file round-trips through the validator, and the
      hashes match the files on disk.

### T8 — CLI

- [ ] `python -m bench.splits.cli --dataset <name> --canonical <path> --out <dir>`,
      with `--seed` and a `--dry-run` that reports the timeline, fold count and
      sample size without writing anything.
- [ ] **Verify:** `--dry-run` on each dataset prints the counts in §3.

### T9 — Tests

- [ ] Unit tests for month flooring, eligibility, fold construction and sample
      allocation, on small synthetic canonical tables.
- [ ] Golden tests against the real counts in §3.
- [ ] A determinism test: run twice, compare manifest bytes.
- [ ] A refusal test for a too-short table.
- [ ] **Verify:** the whole suite passes alongside the existing ingest tests.

### T10 — Produce and freeze the real manifests

- [ ] Run on `eagle_parquet` and `kestrel`; store both manifests under
      `data/manifests/`; record their hashes in `bench/README.md`.
- [ ] **Verify:** the golden counts and the determinism check hold on real data.

### T11 — Follow-up (not blocking this component)

- [ ] Ingest the remaining Eagle 3-month files (`anon_jobs_2020-04.json`,
      `anon_jobs_2020-08.json`) so the reproduction table is complete.

---

## 7. Definition of done

The split generator is done when:

- both real datasets produce a manifest that passes every §8 validation;
- the fold counts match §3 exactly (45 and 22);
- the sample is exactly 20,000 ids, all from open test months;
- re-running produces the same manifest digest (the file differs only in its
  `generated_utc` line);
- `eagle_jsonl` refuses cleanly, as an exempt dataset should;
- `bench/README.md` records the manifest paths and hashes.

---

## 8. What this document does not cover

Arms, features, metrics, energy accounting and the scorecard. Those consume the
manifest this component produces; they are defined in their own documents.
