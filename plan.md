# plan.md — Data Splitting Plan

**Scope of this document.** This is the *data layer* of the benchmark suite: how
each raw dataset becomes one canonical job table, how that table is cut into the
train/test folds every arm must share, where the six-month holdout lives, and
what each arm receives. It deliberately stops before models, metrics and
tuning — those are separate documents. Where a decision here is forced by the
benchmark spec, the spec (`prd.md`) is the authority.

**Why split at all.** The benchmark's claim is *"approach X is better than
approach Y"*. That claim is only meaningful if every approach was asked the
identical question on the identical data. The split is what makes the question
identical. Get it wrong and the scorecard measures plumbing, not methods.

---

## 1. Glossary

| Term | Meaning |
|---|---|
| **Job** | One row in the canonical table — one Slurm job record. |
| **Entity** | One user. Their jobs, in submit order, are their "stream". |
| **Window** | The last 24 jobs of a user's stream, used as the input that predicts the *next* job. |
| **Sample / example** | One (window → next-job label) pair. Built by arms, not by the harness. |
| **Month** | `YYYY-MM`, derived from a job's **submit** time. |
| **Fold** | One split: a set of train months and one test month. |
| **Locked holdout** | The final six months of a dataset. Never touched until the last run. |
| **Frozen sample** | The fixed set of 20 000 test job IDs every arm is additionally scored on. |
| **Manifest** | `folds.json` — the frozen record of all of the above. |

---

## 2. The datasets we actually have

Three sources are in play, and they are **not interchangeable**. Each is used
for what it can support; none is merged with another.

| | `data/eagle_data.parquet` | `21913139/anon_jobs_*.json` | Kestrel (NLR 302) |
|---|---|---|---|
| Machine | Eagle (NREL) | Eagle (NREL) | Kestrel (NLR) |
| Source | OEDI 5860 | data.nlr.gov submission 152 | data.nlr.gov submission 302 |
| Rows | 11,030,377 | 181,503 / 112,099 / 118,319 | 10,559,977 (verified) |
| Months | 52 consecutive, 2018-11 → 2023-02 | 3 non-consecutive: 2019-12, 2020-04, 2020-08 | 29 consecutive, 2023-08 → 2025-12 |
| Users | 936 (778 with ≥25 jobs) | ~194 in 2019-12 | — |
| Durations | raw seconds | ISO-8601 (`P0DT0H30M0S`) | parquet fields |
| Energy | **none** | `avg_power` → **modelled** | `consumed_energy_raw_joules` → **measured** |
| Role in the benchmark | **Primary** (full folds, all tasks) | Reproduction only (published Eagle figures) | **Replication** + measured energy |

**Consequences that shape the split:**

1. Only the two *consecutive-month* datasets (Eagle 11M, Kestrel) can support
   monthly forward-chaining. The 3-month Eagle JSON has 4-month gaps and cannot
   form folds; it exists to reproduce the published energy figure and is scored
   per month, not across folds.
2. The Eagle 11M table has **no energy**, so on it the only scoreable T1 metrics
   are AUC / recall / false-flag. Energy-captured needs the other two sources.
3. Energy provenance differs. See §4.3 — tiers are never mixed.

---

## 3. The core principle: one schema, separate datasets, shared splits

There are two different things that could be called "bringing the data to one".
Only one of them is right.

- ✅ **One canonical schema and one split generator.** Every dataset is poured
  into the same column layout, and the same splitting code runs on each.
- ❌ **One merged table.** We do *not* concatenate Eagle and Kestrel into a
  single training set.

Why not merge:

1. **Energy tiers would mix.** Adding a *measured* kWh (Kestrel) to a *modelled*
   kWh (Eagle) produces a number that is neither, and the spec forbids mixing
   tiers.
2. **They are different machines.** Different partitions, QoS, queue policy and
   user populations. The benchmark's question is whether a method *travels*
   between machines; that is only visible if they are scored separately.
3. **They have different shapes.** Different month coverage, different fields,
   different energy availability. A merged table would be mostly empty columns.

So the rule is: **one schema, one splitter, N datasets, scored independently.**

### 3.1 The one thing that *is* shared: the fold boundaries

Within a dataset, the fold boundaries are frozen **before any arm runs**. Every
arm — A, B1, B2, B3, C, D, D−Jev — receives the same training rows and the same
test rows for each fold. Arms may transform those rows internally (that is the
experiment), but they may not change which rows are in which fold.

---

## 4. Stage 1 — the canonical job table

The split works on the canonical table, so the schema is defined first.

### 4.1 Schema

```
job_id        string   stable id within the dataset
cluster       string   "eagle" | "kestrel"
user_hash     string   salted hash of the raw user id (hex)
account_hash  string   salted hash of the raw account id (hex)
partition     string
qos           string
state         string   normalised: COMPLETED|TIMEOUT|FAILED|CANCELLED|
                       OUT_OF_MEMORY|NODE_FAIL|DEADLINE|PENDING|RUNNING
exit_code     string?  if present in the source
submit_time   timestamp
start_time    timestamp
end_time      timestamp
timelimit_s   float    requested wall clock, seconds
elapsed_s     float    used wall clock, seconds
cpus_req      int
cpus_used     int?     if present
mem_req       float?
mem_used      float?
gpus_req      int
nodes_req     int
nodes_used    int?     if present
array_pos     int?     if present
dependency    string?  if present
queue_wait_s  float?   if present
energy_j      float?   per-job energy in Joules, or null
energy_tier   string   "measured" | "modelled" | "estimated" | "none"
```

### 4.2 Per-dataset mapping

**Eagle 11M (`eagle_data.parquet`)**

| canonical | source |
|---|---|
| `job_id` | `job_id` |
| `user_hash` | hash(`user`) |
| `account_hash` | hash(`account`) |
| `partition`, `qos`, `state` | same names |
| `submit_time`, `start_time`, `end_time` | same names |
| `timelimit_s` | `wallclock_req` (**already seconds**) |
| `elapsed_s` | `run_time` (**already seconds**) |
| `cpus_req` | `processors_req` |
| `gpus_req`, `nodes_req`, `mem_req` | same names |
| `energy_j`, `energy_tier` | **absent → null / "none"** |
| *(dropped)* | `name`, `work_dir`, `submit_line` |

**Eagle 3-month JSON (`anon_jobs_*.json`)**

| canonical | source |
|---|---|
| `job_id` | `job_id` |
| `user_hash` / `account_hash` | hash(`user`) / hash(`account`) |
| `submit_time`, `start_time`, `end_time` | same names (ISO timestamps) |
| `timelimit_s` | `iso_seconds(wallclock_req)` |
| `elapsed_s` | `iso_seconds(wallclock_used)` |
| `cpus_req` / `cpus_used` | `processors_req` / `processors_used` |
| `nodes_req` / `nodes_used` | `nodes_req` / `nodes_used` |
| `array_pos` | `array_pos` |
| `queue_wait_s` | `iso_seconds(queue_wait)` |
| `energy_j` | `avg_power × nodes_used × elapsed_s` |
| `energy_tier` | **"modelled"** |
| *(dropped)* | `script`, `nodelist`, `std_power` (kept only if a task needs them) |

**Kestrel (once fetched)**

| canonical | source |
|---|---|
| `energy_j` | `consumed_energy_raw_joules` → tier **"measured"** |
| *(must NOT be used as measured)* | `consumed_energy_joules` (formatted string, may carry a "K" suffix) and `cpu_energy_tdp_estimated_*` (TDP estimates). Those are a different tier. |

Only **25 of the 29 Kestrel months** carry energy; the four earliest have no
`energy_j`. That is a property of the data and must be carried through as
`energy_tier = "none"` for those months, not imputed.

### 4.3 Energy tiers are carried, never mixed

Three tiers exist, and every energy number in every output keeps its label.
Summing across tiers is forbidden. This is the mechanism that lets us *use*
heterogeneous data without pretending it is homogeneous.

| Tier | Where it comes from | Present in |
|---|---|---|
| measured | Slurm `ConsumedEnergyRaw` | Kestrel (25/29 months) |
| modelled | avg power × nodes × elapsed | Eagle 3-month JSON |
| estimated | TDP × cores × elapsed × load factor | not used unless a site needs it |

---

## 5. Time, ordering, and the no-leakage rule

### 5.1 Order by submit time

The benchmark's rule is: **a feature for job *n* may use only jobs that ended
before job *n* was submitted.**

Therefore:

- Each user's jobs are ordered by **`submit_time`**.
- A job's scheduling **month** is derived from its **`submit_time`**.

> ⚠️ **Known discrepancy to fix.** The reference tool (`lattice24_assess/core.py`)
> buckets by the **end** month and sorts by **end** time. That is mildly looser
> than the spec rule. In the harness we split and order by **submit** time, and
> additionally enforce "ended before submitted" *within* a month. This is a
> deliberate, documented deviation from the reference implementation's
> bookkeeping — the method itself (24-job window, four features, logistic
> regression) is unchanged.

### 5.2 What counts as leakage

- Using a test job's own outcome or duration in its own features. (Forbidden.)
- Building a user's history from jobs that ended after the predicted job was
  submitted. (Forbidden.)
- Reusing the same job both to fit and to score within a fold. (Forbidden.)
- Letting history cross datasets (an Eagle history never predicts a Kestrel
  job). (Forbidden — folds are per dataset.)

### 5.3 Minimum history

A user must have **≥ 25 jobs within the dataset** before any of their jobs can
be scored, because a 24-job window plus one predicted job needs 25. Users below
that are dropped *upstream of the split*, so every arm inherits the same
exclusions. (The write-up notes this excludes 74 of 194 users in Eagle 2019-12.)

---

## 6. Stage 2 — the splitting procedure

Run this **once per dataset**, then freeze the result.

```text
INPUT : canonical job table for one dataset
OUTPUT: folds.json  (the manifest, see §9)

1. Drop jobs with a null submit_time, a null state, or timelimit_s <= 0.
2. Drop users with < 25 jobs in the dataset.             # minimum history
3. For every remaining job: month = submit_time[0:7]      # "YYYY-MM"
4. months = sorted(unique(month))
5. locked = months[-6:]                                   # the holdout (§7)
   open    = months[:-6]                                  # everything else
6. folds = []
   for i in range(1, len(open)):                          # need >=1 train month
       m      = open[i]
       train  = jobs where month in open[:i]
       test   = jobs where month == m
       if len(train) < 100:  continue                     # guardrail (§10)
       if len(test)  <  50:  continue
       if positives(train) < 5 or positives(test) < 5: continue
       folds.append({test_month: m, train_months: open[:i],
                     n_train, n_test, pos_train, pos_test})
7. if len(folds) < 3: REFUSE  (dataset too short — see §10)
8. sample = stratified_sample(test_jobs, n=20000, by=[state, user_activity])
                                                    # frozen (§8)
9. write folds.json with: dataset id, snapshot hash, time_key="submit_time",
   reference_window=24, min_history_jobs=25, locked, folds[], sample job_ids, seed
```

### 6.1 Why expanding-window, not sliding

Train on **all** months before *m* (expanding), not a fixed window of recent
months. This matches the published method ("train on months 1..k, score month
k+1") and avoids an arbitrary window-length hyperparameter that would
disadvantage arms differently.

### 6.2 What is reported for a fold set

For each metric, the **median across test months** and the **interquartile
range** across test months — never a single pooled number. Folds are the units;
the median/IQR is the summary.

---

## 7. The locked holdout — where the six months live

**What it is.** The **last six consecutive months** of each dataset's timeline.
For Eagle 11M that is `2022-09 … 2023-02`. For Kestrel it is the final six of its
29 months.

**What it is not.** A six-month development phase. The project plan is eight
weeks; the "six months" is a *window of data*, and separately the method's
minimum-data floor.

**Rules:**

1. No tuning, no model selection, no threshold choice may look at the locked
   months.
2. It is scored **exactly once**, after every arm is frozen.
3. It exists because of the "test-set overuse" risk: repeatedly reporting on the
   same months slowly inflates every score. Locking it removes that lever.

**Why six.** It is long enough that the median and IQR across test months mean
something (one lucky month cannot carry the result) and short enough to leave
the rest of the timeline for training and tuning. The spec does not derive the
number; it is a documented trade-off, not a law.

Placement:

```
 timeline  ─────────────────────────────────────────────► time
           [ ......... open months .......... ][ locked 6 months ]
             fold 1  fold 2  ...  fold k          run ONCE, at the end
             (tuning + validation may use these)  (frozen arms only)
```

---

## 8. The frozen 20 000-job sample

**Purpose.** Arm C (Jev, an API) cannot be run over millions of jobs, so every
arm is additionally scored on the same **20 000 test jobs**, stratified by
**final state** and **user activity**. Arm C is compared to the others only on
this sample; A, B and D report both the sample and the full test sets.

**Construction:**

- Drawn from the **test** rows (not the training rows).
- Stratified by `state` (so the rare TIMEOUT class is represented) and by a
  user-activity bucket (e.g. quartiles of jobs-per-user), so heavy and light
  users both appear.
- The **job IDs are written into the manifest** and frozen before any arm runs,
  so no arm can influence its own sample.

**Decided.** The sample is drawn from the **open test months only** — never the
locked holdout — so it stays usable during development; the locked window is
evaluated separately at the final run. Strata are `state × activity-quartile`,
where activity = jobs per user, bucketed into log-scale quartiles (see §14.2).
The sample job IDs are written into the manifest and frozen before any arm runs.

---

## 9. What each arm receives — same rows in, arm-specific work inside

This is the crux of "same data in, modifications inside".

```
folds.json (frozen)
      │
      ├──  Arm A   ─┐
      ├──  Arm B1  ─┤   each arm is HANDED the same (train rows, test rows)
      ├──  Arm B2  ─┤   per fold, plus the frozen sample ids and the label
      ├──  Arm B3  ─┤   definition. Each returns predictions in ONE shape.
      ├──  Arm C   ─┤
      └──  Arm D   ─┘
                    │
             predictions: {fold_id, job_id, score, probability}
                    │
             shared evaluator (same metrics, same months)
```

### 9.1 The contract every arm must honour

**Every arm receives, per fold:** the training rows, the test rows, the label
definition, and the frozen sample ids. Those row *sets* are identical across
arms.

**Every arm is free to, inside its own pipeline:**

- engineer any features it wants (this is the experiment);
- build its own windows / sequences / aggregates;
- standardise, encode, impute, calibrate;
- choose any model family, within its tuning budget.

**No arm may:**

- add, remove, or move rows between folds;
- change the label definition;
- use test rows to fit or to select features;
- use information that did not exist before a job was submitted;
- borrow history from another dataset;
- choose its own operating point or metric definition.

### 9.2 Feature sets are *meant* to differ

"Same data" does not mean "same features". The benchmark varies features on
purpose:

| Arm | Features it builds from the same rows |
|---|---|
| A | the four Elapsed/Timelimit statistics over the last 24 jobs |
| B | request fields + user-history aggregates; B3 reads the raw last-32 sequence |
| C | a state (current request + last 24 jobs) sent to Jev |
| D | B's best learner **plus** domain features and detectors |

Constraint: **D contains A's features as a subset** and reuses **B's best
architecture**, so D can only lose to them by overfitting. Order of build:
A → B → D.

---

## 10. Guardrails — when the harness refuses

Inherited from the reference method and applied before any arm runs:

| Condition | Threshold | Why |
|---|---|---|
| Scoreable months | ≥ 6 | below this the per-month spread is noise |
| Scoreable windows | ≥ 50 000 | below this the estimate is unstable |
| TIMEOUT events | ≥ 200 | the positive class must exist in quantity |
| Usable folds | ≥ 3 | a single split cannot support a CI |
| Jobs per user | ≥ 25 | the 24-window cannot be formed otherwise |

The harness **refuses and reports which condition failed** rather than emitting
a number it cannot stand behind.

---

## 11. Worked example on the data in hand

**Eagle 11M (`data/eagle_data.parquet`)**

- 52 months, 2018-11 → 2023-02; 936 users, 778 with ≥25 jobs.
- Locked → `2022-09 … 2023-02`.
- Open → 46 months → up to **45 folds**.
- Energy: none, so T1 is AUC/recall only (this is why the report showed
  `n/a (no energy)`).

**Kestrel (NLR 302)** — fetched (`21913139/kestrel/`, 29 files)

- 29 months, 2023-08 → 2025-12.
- Locked → final 6 months.
- Open → 23 months → up to **22 folds**.
- Energy: measured, 25 of 29 months instrumented; within-month coverage varies
  (see §14.3 and `canonical_table.md` §4.3).

**Eagle 3-month JSON**

- Months 2019-12, 2020-04, 2020-08 — **non-consecutive**.
- Cannot form a month-based fold series → **reproduction only** (published
  energy figure), scored per month.

---

## 12. Why we split this way — the reasoning, collected

| Choice | Reason |
|---|---|
| Split by **submit** month | Enforces the no-leakage rule; a job's request is known at submit time. |
| **Expanding** train window | Matches the published method; avoids an arbitrary window parameter. |
| **Forward-chained**, never reversed | Mirrors deployment: you always predict forward in time. |
| **Per-dataset** folds | Preserves energy tiers and makes replication, not pooling, the test. |
| **Last 6 months locked** | Removes the temptation to tune against the test set. |
| **Frozen 20k sample** | Lets API-only Arm C be compared fairly; fixed before results exist. |
| **Same rows, free features** | The question is *which feature philosophy wins*; features are the treatment. |
| **Min 25 jobs/user** | The 24-window is undefined below that. |
| **Median + IQR across months** | A single pooled number hides month-to-month instability. |

---

## 13. Validation checklist for the split layer

Before any arm is built, confirm:

- [ ] Every dataset parses into the canonical schema with the right `energy_tier`.
- [ ] No sensitive field (`name`, `work_dir`, `submit_line`, raw IDs) reaches
      any output; user/account ids are hashed.
- [ ] Month is derived from `submit_time`, and ordering is by `submit_time`.
- [ ] No fold has test rows overlapping train rows; no history crosses a fold.
- [ ] The locked months appear in **no** fold and in **no** tuning set.
- [ ] The 20k sample's job IDs are fixed and reproducible from the seed.
- [ ] `folds.json` verifies against a hash of the input snapshot.
- [ ] Re-running the splitter on the same snapshot produces byte-identical
      manifests.

---

## 14. Decisions (with rationale)

Six of the seven questions are **decided** (§14.1–14.6). One remains open
(§14.7). Each entry states the decision (or the open choice) first, then the
reasoning.

### 14.0 Decision log

| # | Question | Decision |
|---|---|---|
| 1 | Sample scope | ✅ Open test months only |
| 2 | User-activity bucket | ✅ Jobs per user, log-scale quartiles, crossed with final state |
| 3 | Kestrel's 4 energy-less months | ✅ Keep them; AUC/recall only; `energy_tier = "none"` |
| 4 | Submit-time vs end-time bucketing | ✅ Enforce the spec's submit-time rule; document the delta vs published |
| 5 | The 3-month Eagle set | ✅ Exempt from fold evaluation; reproduction only |
| 6 | `DEADLINE` jobs in the T1 label | ✅ `TIMEOUT`-only positive; report the deadline count as a side note |

Four further decisions — made while planning the split generator — are recorded
in `splits.md` §2 (S1–S4): the 1,000-row month floor, proportional-with-floor
sample quotas, whole-dataset activity quartiles, and shipping the eligible-user
set as a count + hash.

### 14.1 Sample scope — ✅ DECIDED

**Decision.** The 20 000-job sample is drawn from the **open test months only**.
The locked six months are evaluated separately at the final run. Rationale: the
sample must be usable during development without peeking at the holdout.

### 14.2 User-activity bucket — ✅ DECIDED

**Decision.** Activity = **jobs per user**, bucketed into **log-scale quartiles**
over the dataset, crossed with **final state** to form the sample strata. ≤4
activity buckets, so every stratum stays populated.

**What the question is.** The sample is stratified "by final state and user
activity". *Final state* is unambiguous (TIMEOUT / COMPLETED / FAILED / …).
*User activity* is not: what property of a user decides which stratum they land
in?

**Why it matters.** Jobs are wildly skewed across users. On Eagle there are only
936 users; the median user has 443 jobs, but the busiest single user has
654 013. If we sample jobs uniformly, a handful of giant accounts dominate the
sample and their behaviour becomes "the result". If we sample users uniformly,
we over-weight one-off users. The activity bucket is what stops the sample from
quietly becoming "mostly user0001".

**Options for defining "activity".**

| Option | What it measures | Trade-off |
|---|---|---|
| (a) jobs per user | how much history exists | simplest; directly tied to whether a 24-window can form; heavy-tailed |
| (b) active days | distinct days with ≥1 job | robust to array-job bursts; ignores intensity |
| (c) requested core-hours | resource weight | closest to energy impact; needs extra fields |
| (d) jobs per month | current engagement | captures recent activity; noisier |

**Recommendation.** (a) jobs per user, split into **quartiles on a log scale**
(because the distribution is heavy-tailed), crossed with final state. Keep to
≤4 activity buckets so every stratum stays populated. Cross-check later against
(c) if it turns out to matter.

### 14.3 The 4 non-instrumented Kestrel months — ✅ DECIDED

**Decision.** Keep the four months; score them for **AUC/recall only**; set
`energy_tier = "none"`; exclude them from every energy metric. Always report the
number of months each metric was computed over.

**Confirmed against the real archive** (`canonical_table.md` §4.3): the 29 files
hold 10,559,977 rows, instrumentation effectively begins **2023-12** (25 months),
and coverage *within* the instrumented months varies widely — 2024-05 is ≈ 54%
null, 2024-04 ≈ 38% null. So "no energy" is a **per-row** property, not only a
per-month one: energy metrics must be computed over rows that actually carry a
reading, and the coverage percentage reported alongside them.

**What the question is.** Kestrel has 29 months, but Slurm energy measurement
only begins in 2023-12. The four earliest months (2023-08 … 2023-11) therefore
have **no energy readings at all**. What do we do with them?

**Why it matters.** T1's *headline* metric is "energy captured at k%", and that
is undefined without energy. If four months carry none, then the AUC metric can
be computed over 29 months while the energy metric can only be computed over 25.
A median/IQR that silently mixes those two month-sets is misleading, and the
energy numbers must state which months they cover.

**Options.**

| Option | Effect |
|---|---|
| (a) Keep them, score AUC/recall only, `energy_tier = "none"`, exclude from energy metrics | uses all the data, invents nothing — **recommended** |
| (b) Drop those four months entirely | loses four months of AUC evidence |
| (c) Impute / interpolate their energy | forbidden — invents a measurement and mixes tiers |

**Recommendation.** (a): score them for AUC/recall, set `energy_tier = "none"`,
and always report the month count each metric used. (This matches the write-up's
own handling. Their absence is a property of the dataset, not a choice.) Note
they do **not** fall in the locked holdout, so the final energy figures are
unaffected.

### 14.4 Submit-time vs end-time bucketing — ✅ DECIDED

**Decision.** The harness **enforces the spec's submit-time rule** (order and
bucket by `submit_time`; a job's history may contain only jobs that *ended*
before it was submitted). The delta against the published figures, which used
the reference tool's end-time bookkeeping, is measured and documented alongside
the Gate-1 reproduction rather than hidden.

**What the question is.** Which timestamp decides a job's place in the timeline:
when it was **submitted**, or when it **ended**?

**What the reference tool does.** `lattice24_assess/core.py` sorts each user's
jobs by **`end`** time and assigns the month from the predicted job's
**`end_time`** (`target.end.strftime("%Y-%m")`).

**What the spec requires** (`prd.md`): *"features for job n use only jobs that
ended before job n was submitted."*

**Why the difference is not academic — a concrete leak.** Suppose one user has:

- **T** — submitted 10:00, runs 6 h, ends 16:00 (the job we want to predict)
- **S** — submitted 11:00, runs 5 min, ends 11:05

Sorted by **end** time, S (11:05) comes *before* T (16:00), so S lands in T's
history. But S was submitted *after* T — at the moment T was submitted, S did
not exist. That is using the future to predict the target.

Sorted by **submit** time, T (10:00) precedes S (11:00), so S can never enter
T's history.

**A second, subtler point.** Even with submit ordering, a job that was submitted
earlier but was *still running* when T was submitted has an unknown final
duration — its ratio is not yet knowable. Strictly, T's history should contain
only jobs that **ended** before T was submitted.

**The tension.** The published figures behind Gate 1 were produced with the
reference tool's end-based bookkeeping, so enforcing the stricter rule may move
the numbers slightly.

| Option | Effect |
|---|---|
| (a) Enforce the spec rule in the harness, document the delta vs published figures | correct per spec — **recommended** |
| (b) Match the reference bookkeeping exactly for Arm A only, to pass Gate 1 | reproduces published numbers, but deviates from the spec rule |
| (c) Compute both and report the difference | most transparent, slightly more work |

**Recommendation.** (a) for the benchmark, plus a side-by-side reproduction note
so any gap against the published figures is explained rather than hidden.

### 14.5 The 3-month Eagle set and fold evaluation — ✅ DECIDED

**Decision.** The 3-month Eagle JSON is **exempt from fold-based evaluation** and
used for **reproduction only** (published per-month table + modelled energy
figure). It is never a harness fold set.

### 14.6 The `DEADLINE` state in the label — ✅ DECIDED

**Decision.** T1's positive class stays **`TIMEOUT` only**; `DEADLINE` counts as
a **negative**, exactly as published. In addition, the evaluation must **report
how many `DEADLINE` jobs were flagged**, so the choice is visible rather than
buried.

**Raised by the Kestrel schema scan** (`canonical_table.md` §4.3.2). Kestrel
contains **32,071 `DEADLINE` jobs** — a Slurm deadline kill, distinct from
`TIMEOUT`. The published method defines T1's positive class as **`TIMEOUT` only**,
so under that definition every `DEADLINE` job is a **negative**.

**Why it matters.** A deadline kill is semantically timeout-like (the job was
stopped by a limit, not completed), so treating it as "finished normally"
arguably mislabels it. But changing the positive class would break comparability
with the published figures and Gate 1.

**Why the effect is tiny.** Slurm's `DEADLINE` is not the wall-clock limit: it is
a separate absolute cutoff (`--deadline`), so a job can reach it having used
almost no runtime. Of Kestrel's 32,071 `DEADLINE` rows, **32,065 never ran at
all** — they were killed while queued — and only **6** survived the never-ran
filter. Whichever way they are classified, they barely move a metric.

**Options.**

| Option | Effect |
|---|---|
| (a) Keep `TIMEOUT` only as positive; `DEADLINE` negative | matches the published method and Gate 1 — **recommended** |
| (b) Treat `TIMEOUT` + `DEADLINE` as positive | arguably more correct, breaks published comparability |
| (c) Keep (a) for T1, and report `DEADLINE` separately as a sensitivity check | most informative, slight extra work |

**Why this option.** (a) keeps the headline directly comparable to the published
method and to Gate 1, which is the whole point of running Arm A. (b) would be
defensible on the physics but would silently change the answer key, making every
comparison with the published figures invalid. The (c)-style side note costs
almost nothing and removes the "did they even consider this?" question.

### 14.7 NEW — Kestrel energy total does not match the published figure — OPEN

**Raised by the Kestrel ingest** (`canonical_table.md` §4.3.3). Summing measured
energy over TIMEOUT jobs gives **10,853,422 kWh**, against the published
**9,370,367 kWh** Gate-1 target — about **+15.6%**.

**Why it matters.** Gate 1 requires Kestrel timeout energy within **±2%** of the
published number, so this check currently **fails**, and it is the one Gate-1
quantity we cannot reproduce.

**What was tested and ruled out:** month subsetting (25 instrumented months vs
all), window eligibility (≥25-job history), checkpoint-restart exclusion, and
the `consumed_energy_raw_watt_hours` column. None closes the gap.

**The telling detail:** on 2023-12 — the only month the write-up publishes a
figure for — we are **+1.3%** (222,351 vs 219,579 kWh), while the 25-month total
is **+15.6%**. A uniform definitional difference would move both by the same
amount, so the archive itself appears to differ from the one used in the
write-up.

**Recommendation.** Do **not** tune the ingest to hit 9,370,367. Record the
discrepancy in the audit and in the write-up, and investigate separately
(candidate causes: a revised data snapshot, a month-set or eligibility rule we
have not identified, or a different energy column upstream). **Needs your
decision on how to report it.**

**What the question is.** The Eagle 3-month JSON (2019-12, 2020-04, 2020-08) is
non-consecutive and cannot produce ≥3 forward-chained folds.

**Recommendation.** Confirm it is **exempt from fold-based evaluation** and used
for **reproduction only** — the published per-month table and the modelled
energy figure. It never appears as a harness fold set. (Same handling as the
write-up, which scores those months individually.)

---

## 15. Explicitly out of scope for this document

- Model families, architectures and tuning budgets (arms B/C/D).
- Metric definitions and CI methodology.
- Energy accounting *types* (consumed / at stake / saved) and the M&V method.
- The scorecard, hypotheses and gates.

Those live in their own documents; this one fixes the data layer they all sit
on.

---

## 16. Backlog

Known work that does **not** block the current stage. None of these is a code
bug: each is recorded deliberately rather than quietly fixed.

### B1 — Ingest the remaining Eagle 3-month files

`data/canonical/eagle_jsonl.parquet` currently covers only one source file
(`anon_jobs_2019-12.json`), so it holds 2 submit-months rather than the 3
reproduction months. The other two files (`anon_jobs_2020-04.json`,
`anon_jobs_2020-08.json`) are present but have never been read.

**Why it does not block:** this dataset is exempt from fold-based evaluation
(§14.5) — its months are non-consecutive — so it is used for reproduction only.
Neither the split generator nor the arms depend on it.
**To close:** run the ingest over all three files and rebuild the table.
**Tracked as:** `splits.md` T11.

### B2 — Reconcile the Kestrel energy total

Our measured timeout energy is 10,853,422 kWh against a published 9,370,367 kWh
(+15.6%), so Gate 1's ±2% energy check currently fails. Month subsetting, window
eligibility, restart-chain exclusion and the watt-hours column were each tested
and ruled out; the single comparable month is only +1.3% off, which points at
the archive differing from the one behind the write-up rather than a
definitional error on our side.

**Why it does not block:** it is a reconciliation question, not a pipeline bug.
The split generator and the arm interface do not depend on the energy figure.
**To close:** decide how to report it, then investigate the archive's provenance.
**Tracked as:** §14.7.

### B3 — Note the `snapshot` deviation in the manifest spec

`folds_manifest.md` §2 asks the manifest to carry `snapshot{path, sha256, rows}`
for the raw sources. The splitter never reads raw files, so it records the raw
sources by reference (taken from the ingest audit) and hashes the canonical
table in full. The spec should say so explicitly rather than implying a raw hash
that was never computed.
**To close:** one paragraph in `folds_manifest.md` §2.
