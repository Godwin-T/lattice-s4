# metrics.md — How We Score the Arms (Explained Simply)

**How to read this.** Every idea is explained in plain words first, then given
precisely for whoever has to implement it (those parts are in
`For the technical reader` blocks you can skip). Nothing here is a model; this
is only about **how we decide whether a prediction was any good**.

**Status:** draft, for sign-off. Five decisions are listed at the end.

---

## 1. The problem this file solves

Each approach ("arm") looks at a cluster's job history and says: *"this next job
looks risky, it may hit its time limit."* That is a **flag**.

But a flag is just an opinion. To compare arms we need to answer:

1. When we flagged, were we right?
2. When we flagged, did we catch the jobs that actually waste the most energy?
3. When we stayed quiet, were we right to?
4. And could a much dumber rule have done just as well?

This file fixes the answers to those questions, once, so that every arm is
scored on exactly the same ruler. Without it, each arm could pick a flattering
definition of "we were right" and the comparison would be meaningless.

---

## 2. The words, in plain English

Read this table once and the rest of the document becomes easy.

| Word | What it actually means |
|---|---|
| **Flag** | The arm says "watch this job." |
| **Risk score** | A number the arm gives each job. Higher = the arm thinks a timeout is more likely. |
| **Positive** | A job that really did hit its time limit (`TIMEOUT`). |
| **Negative** | A job that didn't — it completed, failed, was cancelled, and so on. |
| **Recall** | Of all the jobs that *did* time out, what fraction did we flag? "Did we catch them?" |
| **False flag rate** | Of all the jobs that were *fine*, what fraction did we flag anyway? "How much noise did we make?" |
| **AUC** | One number for "does the arm put likely timeouts above likely-fine jobs?" 0.5 = coin flip, 1.0 = perfect. |
| **PR-AUC** | Same idea as AUC, but fairer when the interesting jobs are rare. |
| **Calibration / ECE** | When the arm says "80% likely", does it happen about 80% of the time? |
| **Energy captured at k%** | Out of all the energy sitting in timed-out jobs, how much did our top-k% flags catch? **This is the headline.** |
| **Median / IQR** | The typical month, and the spread of the middle half. Used because months vary. |
| **Bootstrap CI** | A range saying "the true number is very likely inside here". |
| **Baseline** | A deliberately dumb rule. If we can't beat it, we've learned nothing. |
| **Label-shuffle control** | A sanity test: scramble the answers; the arm should become useless. If it doesn't, the pipeline is leaking and the result is fake. |

---

## 3. What we are counting, and the one number that matters

### 3.1 We rank, we don't decide

An arm doesn't say "yes/no". It gives every job a risk score, and we sort them
from riskiest to safest. Then we flag the **top k%** — say the top 5%.

Why? Because a site cannot review everything. It has a budget. So we report what
happens at several budgets: **1%, 2%, 5%, 10%, 20%**.

This also means an arm can't cheat by picking a flattering cut-off — everyone is
judged at the same budgets.

### 3.2 The headline: energy captured at 5%

**Plain question:** of all the energy that ended up in timed-out jobs, how much
sits inside the jobs we flagged?

**A worked example.** Suppose one month has 100 jobs:

* 10 of them will time out, and together they use **1,000 kWh**.
* The arm ranks all 100. We flag the top 5% — that's 5 jobs.
* Of those 5 flagged jobs, 4 really do time out, and those 4 hold **700 kWh**.
* We also wrongly flagged 1 job that was fine (noise).

| Number | Value | Meaning in this example |
|---|---|---|
| Energy captured at 5% | **700 / 1,000 = 70%** | we caught 70% of the wasted energy |
| Recall | 4 / 10 = 40% | we caught 4 of the 10 timeouts |
| False flag rate | 1 / 90 = 1.1% | we bothered 1 job that was fine |

Notice that **recall and energy captured are different numbers.** Catching four
of ten timeouts sounds mediocre; catching 70% of the wasted energy is the thing
that actually matters, because those four jobs were the expensive ones.

That is why energy captured is the headline and recall is only reported
alongside it.

**One caveat we must always state.** If some jobs have no energy reading, we
can't include them. So every report also says **coverage**: what fraction of the
timed-out jobs actually had an energy reading. 70% captured, computed over 60%
of the energy, is a much weaker claim than 70% over 100%.

For the reader who wants the exact definition:

> **For the technical reader.** Rank a test month's jobs by predicted risk,
> descending. Flag the top k%. Then
> `energy_captured(k) = Σ energy_j over (flagged AND positive) / Σ energy_j over (all positives)`.
> Positives with no energy value are excluded from both sums; report
> `coverage_pos = (# positives with energy) / (# positives)`, the total positive
> energy, and the number of months that contributed. A month with zero coverage
> is excluded and counted, never imputed.

### 3.3 The rest of the numbers, one at a time

**Recall** — "did we catch them?" Of the jobs that timed out, how many did we
flag. Easy to inflate by flagging everything, which is why it never appears
alone.

**False flag rate** — "how much noise did we make?" Of the jobs that were fine,
how many did we flag. The cost of the tool: every false flag is someone's time
wasted.

**AUC** — one number for ranking quality. *Plain version:* pick one timed-out
job and one normal job at random; AUC is the chance the arm ranked the timeout
as riskier. 0.5 means the arm is guessing (a coin flip). 1.0 means it never
gets the order wrong. It's useful because the published research reports it, so
we can compare like for like.

**PR-AUC** — the same "does it rank well" idea, but focused on the rare class.
Only about 6–7% of jobs time out, so a lazy arm that says "nothing is risky"
looks accurate while being useless. PR-AUC punishes that.

**Calibration (ECE)** — does the arm's confidence mean anything? If it says "80%
likely to time out" for a hundred jobs, roughly eighty should actually time out.
ECE is the average gap between the promise and reality. 0 is perfectly honest.

> **For the technical reader.** AUC is rank-based with ties averaged. PR-AUC is
> average precision. ECE uses 15 equal-width probability bins:
> `ECE = Σ_b (n_b/N) · |accuracy(b) − mean_predicted_probability(b)|`.
> Isotonic calibration is fitted on **validation months only**, never on the
> test month being scored and never on the locked holdout.

---

## 4. The other four questions the arms are asked

The headline above is question **T1: will this job hit its time limit?** Four
more questions are asked, though only some arms can attempt them.

| Task | The plain question | How we score it |
|---|---|---|
| **T1 Timeout** | Will this job hit its time limit? | energy captured at 5%, AUC, PR-AUC, calibration |
| **T2 Failure** | Will this job crash or run out of memory? | same as T1, with "failure" as the interesting outcome |
| **T3 Right-sizing** | How much of the time it asked for will it really use? | how far off the guess is, plus the idle energy it identifies |
| **T4 Root cause** | *Why*, and what should the site do? | how well it matches 300 hand-labelled answers |
| **T5 Traceability** | Can every claim be traced back to real job records? | fraction of claims a validator can confirm |

Two details that matter:

* **T1 counts only `TIMEOUT` as the interesting outcome.** A `DEADLINE` job
  (killed at an absolute cutoff rather than by its own time limit) counts as
  "not a timeout", exactly as the published method does. We report how many
  deadline jobs were flagged anyway, so the choice stays visible.
* **T2's interesting outcome is a crash**, i.e. `FAILED`, `OUT_OF_MEMORY` or
  `NODE_FAIL`. A *cancellation* is a person's decision, not a failure, so it is
  not counted here.

---

## 5. How we combine many months into one answer

We test month by month, so we get one number per month, not one number overall.
We report three views:

| View | Plain meaning |
|---|---|
| **Per month** | the raw list — what happened in each month |
| **Median ± IQR** | the typical month, plus the spread of the middle half |
| **Pooled** | all months lumped together, scored once |

Why all three? Because one lucky month shouldn't win a benchmark. The median
tells you what to expect in a normal month; the spread tells you how much to
distrust that; the pooled figure is what lets us compare against published
numbers.

### 5.1 "How sure are we?" — the confidence range

We only have one history to look at. So we play a statistical trick: we
**pretend to rerun the experiment** many times by resampling our own months (for
the full data) or our users (for the small sample), and see how much the answer
moves. The spread of those pretend-reruns gives a range.

The practical rule: **one arm only beats another if the range of the difference
excludes zero.** If the range straddles zero, we say "inconclusive", not "A beat
B".

> **For the technical reader.** 95% bootstrap CIs, 1,000 resamples. Resampling
> unit: test months for the full set, users for the 20,000-job sample (that
> sample is correlated within users, so resampling jobs would understate the
> interval). On a multi-million-row test set the default is **200** resamples:
> each resample re-scores the whole set, so 1,000 is quadratic and will stall a
> shared machine. The count actually used is recorded in every result file, and
> the flag raises it when a run can afford the time.

---

## 6. The two safety checks

### 6.1 Could a dumb rule have done just as well?

Every trained arm must beat **two deliberately stupid rules**:

| Silly rule | What it does |
|---|---|
| **"Repeat the last outcome"** | Predict that this user's next job behaves the same as their last one. |
| **"Longest time limits"** | Just flag whichever jobs asked for the most time. |

If a clever model can't beat "flag the longest time limits", it isn't earning
its complexity. Both rules use only information available *when the job is
submitted*, so neither is cheating.

### 6.2 The label-shuffle control

Here's the trick: take the answer key and **scramble it**, so there is nothing
real to learn. Then train the model again.

* A healthy pipeline now scores about **0.5 on AUC** — a coin flip. It has
  nothing to learn, so it learns nothing. Good.
* If it *still* scores well (say 0.7), something is wrong: information is
  leaking from the answers into the inputs. The whole result is fake.

So whenever the scrambled run scores above chance, the report is declared
**invalid** and prints no headline number. Every trained arm must pass this.

The accepted band is **0.45 to 0.55**. (One open decision, M1 below, is whether
to also allow the wider tolerance the published tool uses when data is thin.)

---

## 7. What every arm has to hand in

So that all of this can be computed, every arm — whatever is inside it — submits
a simple table: **one row per job it scored**, with the job's id, which month it
belongs to, its risk score, and its probability.

> **For the technical reader.**
>
> | Column | Meaning |
> |---|---|
> | `run_id` | arm + configuration + calibration variant |
> | `arm` | `A`, `B1`, `B2`, `B3`, `C`, `D`, `D_jev`, `D_nojev` |
> | `task` | `T1` … `T5` |
> | `fold_id`, `test_month` | from the frozen manifest |
> | `job_id` | must be a test row of that fold |
> | `score` | higher = riskier; used for ranking (any real number) |
> | `probability` | calibrated probability in [0,1]; used for calibration |
> | `latency_ms`, `cost_usd` | optional |
>
> The evaluator **refuses the run** if a fold's test rows aren't each scored
> exactly once, or if any test row was seen during training. Calibration is a
> *run variant* (raw vs isotonic), not a second score column.

---

## 8. The things we also record

"Best" isn't only about accuracy. For every arm we record:

| Metric | Plain meaning |
|---|---|
| Training time | how long it took to build |
| Inference speed | how long it takes to score a million jobs |
| Cost per million jobs | measured from our own runs, never from a vendor's claims |
| Runs fully offline | does it need the internet? |
| Data leaves the site | does anything get uploaded? (Arm C: yes) |

---

## 9. When we refuse to print a number

Some results should not be published at all. We stop rather than mislead:

* the label-shuffle control fell outside **0.45–0.55** → the run is invalid;
* any test month wasn't fully scored → hard error;
* a metric that would rest on fewer than 3 months → suppressed, count reported;
* energy metrics on a dataset with no measured energy → shown as **"not
  available"**, never as zero;
* **we never report "energy saved."** We report energy **at stake**. Nothing is
  saved until a site acts, and proving a saving needs a before-and-after
  measurement, which is a separate piece of work.

---

## 10. Where the results go

Each run produces one file, `results/<run_id>.json`, holding:

* the metrics in all three views (per month, median ± IQR, pooled);
* the confidence ranges;
* coverage and contributing-month counts for every energy metric;
* the baseline and control results;
* the system metrics;
* the fingerprint of the split the run used, so a result can be tied to exactly
  one frozen split.

The final scorecard is assembled from these files — it never recomputes anything.

---

## 11. Decisions (all settled)

| # | Decision | Why |
|---|---|---|
| **M1** | Gate the shuffle test on the strict **0.45–0.55** band. Also report the mean and the published tool's wider tolerance alongside it. | The band is the spec's rule and both our datasets have enough months (45 and 22) for it to mean something. Reporting the tolerance as well means a borderline case is visible rather than hidden. |
| **M2** | A month with zero energy coverage is **excluded** from the energy metric and counted as excluded. | Including it would mean either inventing energy or treating it as zero — both mix real measurements with things that were never measured. |
| **M3** | Calibration is reported as **two separate runs** (raw, then isotonic), not an extra column. | Keeps the hand-in table simple, and makes "before vs after calibration" a like-for-like comparison. |
| **M4** | Confidence ranges resample **months** for the full data and **users** for the 20,000-job sample. | Jobs within one user move together, so resampling jobs from the sample would make the range look narrower than it really is. |
| **M5** | The "repeat the last outcome" baseline breaks ties **deterministically, by job id**. | A baseline that answers differently on different runs is not a baseline. |

---

## 12. Implementation plan

**What the evaluator is.** One program. It takes three inputs — an arm's
prediction table, the frozen split manifest, and the canonical job table — and
writes one result file. It never trains anything and never chooses a metric at
run time; every rule in this document is fixed in code, in a single place.

**Status: E1–E12 built** (`bench/eval/`, 11 tests) and verified against the toy
examples in this document. E13 — the end-to-end run — waits on a real arm run.

```
bench/eval/
  config.py        k values, ECE bins, resamples, control band   (E1)
  predictions.py   load a hand-in table; enforce the contract     (E2)
  ranking.py       AUC, PR-AUC, recall/false-flag at k%           (E3)
  energy.py        energy captured, coverage, contributing months (E4)
  calibration.py   ECE + isotonic (validation months only)        (E5)
  baselines.py     the two naive rules                            (E6)
  control.py       shuffle-control gating                         (E7)
  aggregate.py     per-month / median + IQR / pooled + bootstrap  (E8)
  system.py        time, cost, offline flags                      (E9)
  results.py       results/<run_id>.json + refusal rules          (E10)
  cli.py           python -m bench.eval.cli                       (E11)
```

### Tasks

Ordered; each depends only on the ones above it.

**E1 — Scaffold and config.**
- [ ] Create `bench/eval/` with `__init__.py` and a `config.py` holding every
      constant: `K_VALUES = (1, 2, 5, 10, 20)`, `HEADLINE_K = 5`,
      `ECE_BINS = 15`, `BOOTSTRAP_RESAMPLES = 1000`, `ALPHA = 0.05`,
      `CONTROL_BAND = (0.45, 0.55)`.
- [ ] **Verify:** `python -c "import bench.eval"` succeeds.

**E2 — Load a hand-in table and enforce the contract (§2, §7).**
- [ ] Read a predictions file (Parquet or CSV); check the required columns
      exist and the types are sane.
- [ ] Join the frozen manifest; refuse if any fold's test rows are not scored
      **exactly once**, or if extra `job_id`s appear.
- [ ] Attach the label (`state`) and `energy_j` / `energy_tier` from the
      canonical table, by `job_id`.
- [ ] **Verify:** a well-formed synthetic table passes; one with a missing row
      fails; one with an extra row fails; both failures are refused, not warned.

**E3 — Ranking metrics (§3).**
- [ ] AUC (rank-based, ties averaged), PR-AUC (average precision), and
      recall / false-flag rate at each operating point.
- [ ] Operating points are derived **within each test month** from that month's
      own score distribution.
- [ ] **Verify:** a hand-computed toy example (10 jobs, 3 known timeouts)
      reproduces the expected AUC and recall exactly.

**E4 — Energy metrics (§5).**
- [ ] Energy captured at k%, plus `coverage_pos`, total positive energy, and the
      contributing-month count.
- [ ] Months with zero coverage are excluded and counted (M2); a run on a
      dataset with no energy returns `N/A`, never zero.
- [ ] **Verify:** the worked example in §3.2 (700 of 1,000 kWh ⇒ 70%)
      reproduces exactly; a table with no energy yields `N/A`.

**E5 — Calibration (§6).**
- [ ] ECE with 15 equal-width bins, a reliability curve, and isotonic
      calibration fitted on **validation months only**.
- [ ] **Verify:** a synthetic perfectly-calibrated set gives ECE ≈ 0; a
      deliberately over-confident set gives a clearly larger ECE; an attempted
      fit on the test month is refused.

**E6 — Baselines (§6.1).**
- [ ] Score every test job with "repeat the user's last outcome" (ties broken by
      `job_id`) and "flag the longest time limits", from the canonical table.
- [ ] **Verify:** both baselines reproduce on a hand-built toy table, and both
      are deterministic across two runs.

**E7 — Shuffle-control gating (§6.2).**
- [ ] Read the shuffled-fit AUCs an arm reports, compute the mean, and apply the
      **0.45–0.55** gate; also compute and record the tool-style tolerance
      (`max(2·SEM, 0.02)`) for context (M1).
- [ ] A run outside the band is marked **invalid**: no headline is emitted.
- [ ] **Verify:** mean 0.51 passes; mean 0.67 fails and produces no headline.

**E8 — Aggregation and uncertainty (§3, §5).**
- [ ] Produce the three views — per-month, median ± IQR, pooled.
- [ ] 95% bootstrap CIs: resample **months** for the full set, **users** for the
      20,000-job sample (M4).
- [ ] **Verify:** on two arms with identical distributions the CI of the
      difference includes zero; on a clearly separated pair it excludes zero.

**E9 — System metrics (§8).**
- [ ] Record training time, inference time per 1M jobs, cost per 1M jobs, the
      offline flag and the data-leaves-site flag, normalising units as needed.
- [ ] **Verify:** values pass through unchanged when they are already in the
      documented units; missing values stay missing rather than becoming zero.

**E10 — Results writer and refusal rules (§9, §10).**
- [ ] Write `results/<run_id>.json` with all three views, CIs, coverage and
      contributing-month counts, baselines, control, system metrics, and the
      manifest's `checksums.manifest_sha256`.
- [ ] Enforce every refusal rule: invalid control ⇒ no headline; any metric
      resting on fewer than 3 months is suppressed and the count recorded.
- [ ] **Verify:** an invalid run's file contains no headline figure at all; a
      valid run ties to exactly one split hash.

**E11 — CLI.**
- [ ] `python -m bench.eval.cli --predictions P --manifest M --canonical C
      --out results/`, with a `--dry-run` that reports what would be scored.
- [ ] **Verify:** `--dry-run` on a synthetic input prints the folds, the test
      row count and the k values, and writes nothing.

**E12 — Tests.**
- [ ] Unit tests per module (E2–E9), including the toy examples above.
- [ ] A determinism test: the same inputs give a byte-identical result file.
- [ ] A refusal test set: missing rows, extra rows, invalid control, no energy,
      fewer than 3 months.
- [ ] **Verify:** the whole suite passes alongside the ingest and split tests.

**E13 — End-to-end smoke run** *(depends on the Arm A adapter, not on this
document)*.
- [ ] Score a real arm's predictions against a real manifest and confirm the
      result file contains the expected fields for that dataset — for example,
      energy metrics present on Kestrel and `N/A` on Eagle 11M.
- [ ] **Verify:** the run reproduces on a second invocation.

### Definition of done

The evaluator is done when:

- every rule in this document is enforced in code, in one place;
- the toy examples reproduce exactly, so the arithmetic is provably right;
- an invalid run can never emit a headline figure;
- energy metrics are `N/A` rather than zero wherever energy is absent, and every
  energy metric reports its coverage;
- re-running on the same inputs gives a byte-identical result file;
- a real arm has been scored end to end against a real frozen split.

---

## 13. What this document deliberately does not cover

How each arm is built and tuned; how energy is classified as consumed / at stake
/ saved, and how savings are measured after an intervention; how the scorecard
is laid out; and the arms themselves.
