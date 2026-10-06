# canonical_table.md — Canonical Job Table Mapping Spec

**Status:** design (not yet implemented).
**Implements:** `plan.md` §4 (Stage 1).
**Consumed by:** the split generator (`folds_manifest.md`) and every arm.

**Goal.** Turn each raw dataset into one table with identical columns, units and
semantics, so the splitter and the arms never have to know where a row came
from. Format differences are absorbed here, once.

---

## 1. Output and naming

One canonical table per dataset, written as Parquet:

```
data/canonical/<dataset>.parquet      e.g. eagle.parquet, kestrel.parquet
data/canonical/<dataset>.audit.json   row counts, drops, snapshot hash
```

The table is **immutable** once written. If ingestion rules change, the
`ingest_version` changes and the table is rebuilt — it is never edited in place.

---

## 2. The schema (authoritative)

| Column | Type | Null? | Unit | Notes |
|---|---|---|---|---|
| `job_id` | string | no | — | stable within the dataset |
| `cluster` | string | no | — | `"eagle"` \| `"kestrel"` |
| `user_hash` | string | no | — | see §3.1 |
| `account_hash` | string | yes | — | see §3.1 |
| `partition` | string | yes | — | |
| `qos` | string | yes | — | |
| `state` | string | no | — | canonical set, §3.2 |
| `script_hash` | string | yes | — | only if the source already hashes it (used by D2/D4) |
| `submit_time` | timestamp (UTC) | no | — | the scheduling key, §3.3 |
| `start_time` | timestamp (UTC) | yes | — | |
| `end_time` | timestamp (UTC) | no | — | needed for the no-leakage rule |
| `timelimit_s` | float64 | no | seconds | requested wall clock |
| `elapsed_s` | float64 | no | seconds | used wall clock |
| `queue_wait_s` | float64 | yes | seconds | `start_time - submit_time` when both exist |
| `cpus_req` | int64 | yes | count | |
| `cpus_used` | int64 | yes | count | |
| `mem_req` | float64 | yes | source unit | document the unit per source |
| `mem_used` | float64 | yes | source unit | |
| `gpus_req` | int64 | yes | count | |
| `nodes_req` | int64 | yes | count | |
| `nodes_used` | int64 | yes | count | |
| `array_pos` | int64 | yes | index | |
| `dependency` | string | yes | — | |
| `energy_j` | float64 | yes | Joules | `null` iff `energy_tier == "none"` |
| `energy_tier` | string | no | — | `measured` \| `modelled` \| `estimated` \| `none` |
| `source_dataset` | string | no | — | provenance |
| `source_file` | string | no | — | provenance |
| `ingest_version` | string | no | — | provenance |

Columns the raw data lacks are present but `null`. **Never** drop a canonical
column because a source lacks it — the shape is fixed so arms can rely on it.

---

## 3. Global ingestion rules

### 3.1 Identifier hashing

```
user_hash     = sha256(salt_dataset + raw_user)[:16]
account_hash  = sha256(salt_dataset + raw_account)[:16]
script_hash   = source value if already hashed, else hash as above
```

- The salt is **fixed per dataset** and recorded in the ingest config, so hashes
  are **stable across arms and runs** — required, because folds and predictions
  reference the same IDs.
- Trade-off, stated: this differs from `lattice24-assess`, which uses a throwaway
  per-run salt. A stable hash is linkable across runs. That is acceptable here
  because the sources are already anonymised and only aggregates leave the
  machine; the salt and canonical table are never published together.
- **Sources that already ship non-identifying identifiers pass them through.**
  Kestrel ships `user_hash` / `account_hash` / `submit_script_hash`, and the
  Eagle 3-month JSON ships already-hashed `user` / `account`. Those are copied
  unchanged.
- **Sources that ship source-level identifiers are hashed.** The Eagle 11M
  archive carries synthetic ids (`user0001`, `account0001`); they are hashed as
  `sha256(salt + id)[:16]` so the canonical table never carries a source
  identifier verbatim, and so all three datasets agree in shape. The salt is
  per-dataset, supplied at run time (`--salt`, `BENCH_SALT_<DATASET>`, or
  `BENCH_SALT`), and **never written to disk**. Reuse the same salt, or the
  identifiers change and tables stop matching.
- Identifiers are only meaningful *within* a dataset, never across datasets.

### 3.2 State normalisation

1. Uppercase; take the first whitespace-delimited token
   (`"CANCELLED by 1234"` → `"CANCELLED"`).
2. Map to the canonical set:
   `COMPLETED`, `TIMEOUT`, `FAILED`, `CANCELLED`, `OUT_OF_MEMORY`, `NODE_FAIL`,
   `DEADLINE`.
3. Any state outside the "terminal" set — `PENDING`, `RUNNING`, `SUSPENDED`,
   `REQUEUED` — is **dropped** (no outcome to learn from) and counted.
4. Unknown states are **dropped and logged** in the audit file; they are never
   silently coerced.

Note: `FAILED`, `CANCELLED`, `OUT_OF_MEMORY`, `NODE_FAIL` are **kept** — they
are negatives for T1 and the labels for T2.

### 3.3 Time parsing and timezone

- All timestamps are stored as **UTC**.
- `submit_time` is the **scheduling key**: ordering and month bucketing use it
  (see `plan.md` §5 and `folds_manifest.md`).
- Rows with a null `submit_time`, `end_time`, or `state` are dropped.

### 3.4 Duration parsing

Three source encodings occur; a single `parse_duration(value, kind)` handles them:

| Kind | Format | Example |
|---|---|---|
| `seconds` | float, already seconds | `36000.0` |
| `iso8601` | `P{d}DT{h}H{m}M{s}S` | `P0DT0H30M0S` → 1800 |
| `slurm` | `[DD-]HH:MM:SS` | `1-00:00:00` → 86400 |
| `arrow_duration_ns` | Arrow `duration[ns]` | `43_200_000_000_000` → 43200 (÷ 1e9) |

Rules: reject non-positive `timelimit_s`; treat unparseable values as errors
(drop + log), never as zero. Guard explicitly against the classic trap of a bare
integer being read as *minutes* — the source's `kind` is declared per dataset,
not guessed.

### 3.5 Energy derivation and tiers

`energy_j` is populated **only** from the source's declared energy channel, and
`energy_tier` records which tier it is. Summing across tiers is forbidden.

| Source | `energy_j` | `energy_tier` |
|---|---|---|
| Eagle 11M | *(no energy field)* | `none` |
| Eagle 3-month JSON | `avg_power × nodes_used × elapsed_s` | `modelled` |
| Kestrel | `consumed_energy_raw_joules` | `measured` |

**Kestrel decoys — must not be used as measured:** `consumed_energy_joules` (a
formatted string that may carry a `K` suffix) and `cpu_energy_tdp_estimated_*`
(TDP estimates). If used at all, they are `estimated`, never `measured`.

### 3.6 Row filters

Drop, and count in the audit file:

- non-terminal states (§3.2);
- null `submit_time` / `end_time` / `state`;
- `timelimit_s <= 0` or unparseable;
- a null **used** duration (`elapsed_s`), i.e. a job that never ran;
- rows that fail to parse any required field.

### 3.7 Sensitive fields — dropped at parse time

`name`, `work_dir`, `submit_line` (and any command/comment/account strings) are
dropped **before** any other processing and never written to the canonical table.
Accounts are hashed, not dropped (the benchmark's core record keeps
`account_hash`).

---

## 4. Per-dataset mappings

### 4.1 Eagle 11M — `data/eagle_data.parquet`

| canonical | source | transform |
|---|---|---|
| `job_id` | `job_id` | → string |
| `user_hash` | `user` | **hash** `sha256(salt + user)[:16]` (§3.1) |
| `account_hash` | `account` | hash |
| `partition`, `qos` | same | |
| `state` | `state` | normalise (§3.2) |
| `submit_time`, `start_time`, `end_time` | same | → UTC |
| `timelimit_s` | `wallclock_req` | `kind="seconds"` |
| `elapsed_s` | `run_time` | `kind="seconds"` |
| `cpus_req` | `processors_req` | |
| `gpus_req` | `gpus_req` | |
| `nodes_req` | `nodes_req` | |
| `mem_req` | `mem_req` | |
| `energy_j` / `energy_tier` | — | `null` / `"none"` |
| *dropped* | `name`, `work_dir`, `submit_line` | §3.7 |

### 4.2 Eagle 3-month JSON — `21913139/anon_jobs_*.json`

| canonical | source | transform |
|---|---|---|
| `job_id` | `job_id` | → string |
| `user_hash` / `account_hash` | `user` / `account` | hash |
| `script_hash` | `script` | already hashed |
| `partition`, `qos`, `state` | same | normalise state |
| `submit_time`, `start_time`, `end_time` | same | ISO-8601 with `Z` → UTC |
| `timelimit_s` | `wallclock_req` | `kind="iso8601"` |
| `elapsed_s` | `wallclock_used` | `kind="iso8601"` |
| `queue_wait_s` | `queue_wait` | `kind="iso8601"` |
| `cpus_req` / `cpus_used` | `processors_req` / `processors_used` | |
| `nodes_req` / `nodes_used` | `nodes_req` / `nodes_used` | |
| `array_pos` | `array_pos` | |
| `energy_j` | `avg_power × nodes_used × elapsed_s` | **modelled** |
| *dropped* | `nodelist`, `std_power` | not needed downstream |

### 4.3 Kestrel — NLR submission 302 (confirmed)

**Verified against the fetched archive:** 29 monthly files,
`kestrel_jobs_<YYYYMM>_0.parquet`, 2023-08 … 2025-12, **10,559,977 rows**.

| canonical | source | transform |
|---|---|---|
| `job_id` | `job_id` | → string |
| `user_hash` / `account_hash` | `user_hash` / `account_hash` | **pass through** (already hashed) |
| `script_hash` | `submit_script_hash` | pass through |
| `partition`, `qos` | same | |
| `state` | `state_simple` | already normalised: `COMPLETED` / `TIMEOUT` / `FAILED` / `CANCELLED` / `OUT_OF_MEMORY` / `NODE_FAIL` / `DEADLINE` (+ 1 `PENDING` row → drop) |
| `submit_time`, `start_time`, `end_time` | same | **mixed timezones** — convert to UTC per file (see consequence 3) |
| `timelimit_s` | `wallclock_req` | `kind="arrow_duration_ns"` |
| `elapsed_s` | `wallclock_used` | `kind="arrow_duration_ns"`; **null ⇒ drop the row** |
| `queue_wait_s` | `queue_wait` | `kind="arrow_duration_ns"` |
| `cpus_req` / `cpus_used` | `processors_req` / `processors_used` | |
| `nodes_req` / `nodes_used` | `nodes_req` / `nodes_used` | **verified correct** — see below |
| `mem_req` | `memory_req` | **string** (`"500000G"`, `"500000M"`) → parse value + unit |
| `energy_j` | `consumed_energy_raw_joules` | **measured** |
| *unavailable* | `array_pos`, `array_range`, `gpus_requested`, `gpu_nodes_occupied`, `min/max/avg_mem_eff` | **null-typed columns — entirely empty in the archive** |
| *decoys* | `consumed_energy_joules` (string), `consumed_energy_raw_watt_hours`, `cpu_energy_tdp_estimated_*` | never `measured` |

**Consequences to carry forward:**

1. **`array_pos` is unusable on Kestrel** — it and six other columns are
   null-typed. Any detector or feature that relies on array position, GPU
   counts, or memory efficiency is **Eagle-only**; on Kestrel those features
   must be absent rather than imputed.
2. **Energy coverage is not uniform, and not just "four months".**
   Instrumentation effectively begins **2023-12** (25 months, matching the
   write-up), but coverage *within* those months varies a lot — e.g. 2024-04
   ≈ 38% null and 2024-05 ≈ 54% null, while some months are >90% populated.
   2023-08 … 2023-11 are essentially uninstrumented (2023-11 has only 38
   non-zero readings). Therefore every energy metric must be computed **only
   over rows that actually carry `energy_j`**, with the coverage percentage
   reported alongside — never over all rows, and never with imputed values.
3. **Timestamps are not uniformly UTC.** 24 of the 29 files are UTC, but
   **2023-11, 2024-03, 2024-11, 2025-03 and 2025-11 carry −07:00 / −06:00**.
   A naive concatenation fails on the dtype mismatch, so each timestamp column
   is converted to UTC **per file** before combining. (Supersedes the earlier
   "already UTC" assumption.)
4. **`memory_req` is a string and looks like a partition default, not a real
   request.** Values are `"<number><K|M|G>"` (e.g. `256G`, `250000M`,
   `500000G`); the same `500000G` appears against both 2-node and 2048-node
   jobs, and the median `M` value is 250000. It is parsed to MB for
   completeness but **must not be used as a feature** until its meaning is
   resolved.

### 4.3.3 Energy total — OPEN DISCREPANCY

Summing measured energy over TIMEOUT jobs does **not** reproduce the published
Kestrel figure, and this is unresolved:

| Quantity | Computed | Write-up / Gate 1 |
|---|---|---|
| All TIMEOUT, all months | 10,857,813 kWh | **9,370,367 kWh** |
| Window-eligible only (≥25-job history) | 10,830,309 kWh | (same target) |
| Net of checkpoint-restart chains | 10,785,108 kWh | (same target) |
| 2023-12 alone | 222,351 kWh | 219,579 kWh (+1.3%) |

(Figures above are from the raw archive. From the finished canonical table —
i.e. after the never-ran rows are dropped — the all-months TIMEOUT total is
**10,853,422 kWh**, a difference of 4,391 kWh.)

Month subsetting, window eligibility, restart-chain exclusion and the
watt-hour column were each tested and none closes the gap. Because the single
month we can compare against is only **1.3%** high while the 25-month total is
**+15.6%** high, a uniform definitional difference is ruled out: the archive
appears to differ from the one behind the write-up. **Do not adjust the numbers
to match**; record this and investigate separately.

### 4.3.1 `nodes_used` — investigated and cleared

`nodes_used` looked implausible in spot checks (it equalled `processors_req` in
some rows). A full scan of all 29 months settles it:

- **`nodes_used == len(nodelist)` in 100.0% of rows that have a non-empty
  `nodelist`.** The field is genuine.
- The apparent disagreement is **entirely** the rows with an **empty
  `nodelist`** — jobs that were never allocated nodes: overwhelmingly
  `CANCELLED`, plus `DEADLINE` and `FAILED`. These are the same rows that carry a
  **null `wallclock_used`** (99.2–100% overlap), so §3.6 already drops them.
- **One real anomaly remains:** `nodes_used > nodes_req` in four months only —
  2023-08 (26.6%), 2023-09 (13.7%), 2024-04 (15.8%), 2024-05 (30.5%). It is
  ~0% everywhere else. This is a data-quality artefact in those four months. It
  is **harmless here** because Kestrel energy is *measured*, never modelled from
  node counts — but any feature built on `nodes_used` should treat those four
  months with suspicion.

### 4.3.2 New terminal state: `DEADLINE`

The archive contains **32,071 `DEADLINE` jobs** (a Slurm deadline kill), a state
not in the original canonical list. It is terminal and is now kept. Note for the
label: under the published method T1's positive class is **`TIMEOUT` only**, so
`DEADLINE` counts as a *negative*. That is semantically awkward — a deadline
kill is timeout-like — so it is flagged as a decision point rather than silently
baked in. Full archive state counts: COMPLETED 6,991,818 · CANCELLED 1,534,430 ·
FAILED 1,361,924 · TIMEOUT 559,732 · OUT_OF_MEMORY 43,931 · NODE_FAIL 36,070 ·
DEADLINE 32,071 · PENDING 1.

**After the never-ran filter, only 6 `DEADLINE` jobs survive** — the other
32,065 were killed while *queued*, with no `wallclock_used`. So `DEADLINE` is
overwhelmingly a queue deadline rather than a mid-run kill, which makes
treating it as a negative considerably less awkward.

**Resolved (`plan.md` §14.6):** `DEADLINE` stays a **negative** for T1
(`TIMEOUT`-only positive, as published), and the evaluation additionally reports
how many deadline jobs were flagged. Note that Slurm's `DEADLINE` is a separate
absolute cutoff (`--deadline`), not the wall-clock limit — so it is not simply
"exceeded the timeline".

---

## 5. Invariants and edge cases

Flagged (logged), not silently fixed:

- `end_time < start_time` or `start_time < submit_time`;
- `elapsed_s` materially different from `end_time − start_time`;
- `elapsed_s > timelimit_s` (a job can slightly overshoot its limit);
- duplicate `job_id` within a dataset;
- `energy_j < 0`.

Hard errors (drop the row and count it):

- unparseable required fields;
- `timelimit_s <= 0`;
- null `submit_time` / `end_time`.

---

## 6. Validation — acceptance tests for ingestion

The ingest is correct only if all of these hold:

- [ ] **Reconciliation:** `rows_in − dropped == rows_out`, per drop reason.
- [ ] **Shape:** output columns equal the §2 schema exactly, in order.
- [ ] **State set:** every `state` is in the canonical terminal set.
- [ ] **Energy consistency:** `energy_j is null` **iff** `energy_tier == "none"`;
      tier ∈ {measured, modelled, estimated, none}.
- [ ] **No raw identifiers:** a scan finds no `name` / `work_dir` / `submit_line`
      values and no un-hashed user/account strings.
- [ ] **Timezone:** all timestamps parse as UTC; no naive/ambiguous values.
- [ ] **Determinism:** re-running ingest on the same snapshot yields the same
      output hash (recorded in the audit file).
- [ ] **Provenance:** `source_dataset`, `source_file`, `ingest_version` populated
      on every row.

---

## 7. What this document does not cover

The split itself (`folds_manifest.md`), model features per arm (`plan.md` §9),
metric definitions, and energy *type* accounting (consumed / at stake / saved).
