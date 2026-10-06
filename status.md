# status.md — Where the Project Stands

**Written:** 6 October 2026.
**Scope:** what is built and evidenced, what remains, what is blocked, and the
plan for Arm C (Jev).

---

## 1. In one paragraph

The **data layer is finished and frozen**: all three datasets are ingested into
one canonical schema, and both benchmark splits are written, hash-pinned and
reproducible from a single setup script. **One of the four arms (A, the
published baseline) is implemented** behind the harness interface, and **the
evaluator that scores every arm is implemented and verified**. What remains is
the bulk of the benchmark itself: **arms B, C and D**, the two long-lead
external dependencies they need, and the final scorecard. Roughly stated: the
foundation is done; the comparison has not been run.

---

## 2. What is covered

### 2.1 The data layer — complete

| Piece | State | Evidence |
|---|---|---|
| Ingest: Eagle 3-month | ✅ | 181,503 rows; reproduces the published 6.6% of jobs / 45% of energy / 74-of-194 users |
| Ingest: Kestrel (29 files) | ✅ | 10,559,977 rows in → 9,320,707 kept; deterministic |
| Ingest: Eagle 11M | ✅ | 11,030,377 rows; sensitive columns never decoded |
| Canonical schema | ✅ | one shape, three separate tables; energy tiers never mixed |
| Reproducibility | ✅ | re-running gives byte-identical tables |

### 2.2 The splits — complete and frozen

| Dataset | Months | Folds | Holdout | Sample |
|---|---|---|---|---|
| Eagle 11M | 52 | **45** | 2022-09 → 2023-02 | 20,000 jobs, 24 strata |
| Kestrel | 29 | **22** | 2025-07 → 2025-12 | 20,000 jobs, 25 strata |
| Eagle 3-month | 2 | refuses (exempt, reproduction only) | — | — |

Both manifests are hash-pinned, so every arm reads exactly the same rows.

### 2.3 The arms

| Arm | State |
|---|---|
| **A — Lattice24** (published baseline) | ✅ implemented behind the harness interface, unit-tested |
| **B1 — boosted trees** | ❌ not started |
| **B2 — MLP** | ❌ not started |
| **B3 — sequence model** | ❌ not started |
| **C — Jev** (zero-shot API) | ❌ blocked on access — see §6 |
| **D — Peepalytics** (+ ablated D−Jev) | ❌ not started; depends on B |

### 2.4 The evaluator — complete

Implements every rule in `metrics.md`: energy captured at 1/2/5/10/20%, AUC,
PR-AUC, ECE, recall and false-flag rates, bootstrap confidence ranges, the
label-shuffle gate, and the refusal rules (an invalid run publishes no headline;
energy is "not available" rather than zero). Verified against the document's own
worked examples, not just "it runs".

### 2.5 The written contracts

| Document | Covers |
|---|---|
| `breakdown.md` | plain-language orientation for a newcomer |
| `prd.md` | the benchmark specification: arms, tasks, metrics, gates, timeline |
| `plan.md` | data layer, decision log, backlog |
| `canonical_table.md` | column-by-column ingestion rules |
| `folds_manifest.md` | the frozen split contract |
| `splits.md` | split generator: decisions and tasks |
| `metrics.md` | metric definitions, decisions, implementation plan |

### 2.6 Test coverage

**54 automated tests** across ingest, splits, arm and evaluator. Every described
rule has a test, including the ones that caught real bugs (multi-file ingest,
the windowing rule).

---

## 3. What is left

Ordered by what unblocks the most.

| # | Item | Owner | Blocks |
|---|---|---|---|
| 1 | Run Arm A end-to-end on both frozen splits | us | Gate 1, and proof the whole chain works |
| 2 | Wire the two naive baselines into the result file | us | "does the model beat a dumb rule?" — currently missing from output |
| 3 | Run the evaluator on that output | us | the first real numbers |
| 4 | **Arm B1** (boosted trees) | us | the workhorse; D is built from it |
| 5 | Arm B2 (MLP), B3 (sequence) | us | T1–T4 comparison |
| 6 | **Jev access + Arm C** | TypeSafe + us | one of four arms; §6 |
| 7 | **Arm D** + ablated D−Jev | us | T1–T5; depends on B |
| 8 | **300 hand-labelled findings** | two people, ~2 days | T4 (root cause) for every arm |
| 9 | Scorecard + comparison report | us | the actual deliverable |
| 10 | Gate 3: savings method (M&V) | us | the project's follow-on |

---

## 4. Progress against the eight-week plan

| Week | Planned | Status |
|---|---|---|
| 1 | Data pulled; canonical table; harness; splits; **Arm A reproduced** | ✅ data + harness + splits done; Arm A built and unit-tested, full reproduction run outstanding |
| 2 | Arm B on T1–T2 | ❌ not started — this is the next work |
| 3 | T3 targets; **start 300-finding labelling**; Jev question design | ❌ not started — the labelling has the longest lead time and should begin now |
| 4 | Arm C on the 20k sample; Arm D detectors + features | ❌ blocked on access |
| 5 | Arm D learner + calibration; D with/without Jev; T4 | ❌ |
| 6 | Evidence trail; T5; cost and latency; locked test run | ❌ |
| 7 | M&V method + simulated intervention | ❌ |
| 8 | Comparison report and 2-page brief | ❌ |

**Read plainly:** the data engineering and the measuring instrument exist. The
contest itself — the models — has not been run. Weeks 2–8 of the plan are
essentially ahead, with one dependency (Jev) outside our control.

---

## 5. Blockers and dependencies

| Blocker | Type | Impact |
|---|---|---|
| **Jev access not yet granted** | external, long lead | Arm C cannot start; T4's Jev ablation cannot be measured |
| **300 findings unlabelled** | human, long lead | T4 unscored for every arm |
| **Kestrel energy total is +16% vs published** | unresolved technical | Gate 1's ±2% energy check fails; the headline metric on the only measured-energy dataset is questionable |
| **Compute** | capacity | the window build has to be chunked to fit the machine available; workable, but slower |
| Two of three Eagle 3-month files were not ingested | small | reproduction table incomplete |
| Repo README still describes the upstream tool | cosmetic | misleading for anyone new |

---

## 6. Arm C (Jev): the plan

**What it is.** Jev is a proprietary model from TypeSafe AI, in limited early
access. It does **not** generate text. You send a *state* and *typed questions*
and it returns typed answers with probabilities. It is used **zero-shot** — we
never train it on our data.

### 6.1 Why "it's just API calls" is true, and still not the whole story

The code is small — smaller than any other arm. The work around it is not:

| Piece | Effort |
|---|---|
| Adapter: build the state, call the API, parse answers, emit a score | ~2 days |
| State design: which jobs, which fields, how IDs are hashed | part of the 3 days below |
| Question design: phrasing the typed questions so answers are comparable | the PRD budgets **3 days**, matched to B's feature-engineering time so the comparison is fair |
| Tuning on validation months only | included above |
| Cost and latency measurement, from our own runs | ~0.5 day |
| Governance: confirming retention terms before sending cluster data | before any call |

### 6.2 The interface we build against

| Primitive | Returns | Maps to |
|---|---|---|
| **Noul** | probability a statement is true | **T1** "this job will reach its time limit"; **T2** "this job will fail" |
| **Score** | probability over a rubric | **T3** share of requested wallclock that will be used |
| **Choice** | one option from a list | **T4** root cause and recommended action |

**State sent:** the current job's request plus that user's last 24 jobs (elapsed
ratio, state, request), with **hashed IDs**.

### 6.3 The plan, in order

1. **Get access.** Requested; still the gating item. Nothing can substitute.
2. **Confirm terms before sending anything** — pricing, rate limits, and data
   retention. Our data leaves the site for this arm; that has to be an explicit
   decision, not an accident.
3. **Build the adapter against the documented interface now, with a mock
   client**, so that when the key arrives it is a configuration change rather
   than a build. This is the part we can do without access.
4. **Design the state and the questions** on validation months only.
5. **Run on the 20,000-job sample** — not the full set. API cost makes full-set
   scoring impractical, which is exactly why the sample exists and why every
   arm is scored on it.
6. **Measure cost and latency from our own runs**, never from vendor claims —
   the vendor's speed and price figures are self-benchmarked.
7. **Run it twice**: once as Arm C, and once as the Jev layer inside Arm D, so
   "D with Jev" versus "D without Jev" is a real ablation.

### 6.4 Budget to plan for

- roughly **20,000 jobs × 3 tasks** of API calls, plus **300 findings** for T4;
- the PRD allows **0.5 of an LLM/app engineer from week 3** for this and the
  explanation layer.

### 6.5 Risk

Jev is in early access, its claims are self-benchmarked, and pricing or terms
may change. So it is kept **behind the harness interface**: if access
disappears, the same slot can be filled by an in-house classifier or an LLM with
a constrained schema — which is precisely how the "D without Jev" ablation runs.
The benchmark never depends on one vendor being available.

---

## 7. Immediate next steps

1. **Run Arm A end-to-end** on both frozen splits — the last piece of proof that
   the chain works on real data, and the input to Gate 1.
2. **Wire in the baselines** and produce the first scored result.
3. **Start the 300-finding labelling** — the longest lead time in the project.
4. **Begin Arm B1**, the model everything else is compared against.
5. **Chase Jev access**; build its adapter against a mock in the meantime.

---

## 8. Open items carried from the data layer

| Item | Where |
|---|---|
| Kestrel energy total vs published (+16%) | `plan.md` §14.7 |
| Manifest `snapshot` block records raw sources by reference | `plan.md` §16 B3 |
| Ingest the two remaining Eagle 3-month files | `plan.md` §16 B1 |
