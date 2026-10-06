# breakdown.md — The Project, Explained From Scratch

**Who this is for.** Anyone arriving new to this repository — technical or not.
By the end you should understand what the project is, what exists so far, and
why each choice was made, without needing to read any code.

**How to read it.** Sections 1–3 are the story and need no background. Sections
4–6 get more technical. Section 8 is a table of every decision and its reason —
if you only read one section, read that. Section 10 is a plain-language glossary.

---

## 1. The problem

### 1.1 A job that hits its time limit

A supercomputer (a "cluster") runs many people's jobs at once. When you submit a
job, you must say how long it may run — its **wall-clock limit**. Think of it as
booking a meeting room for 8 hours even if you might only need 1.

The scheduler enforces that booking. If your job is still running when the limit
is reached, the machine **kills it** and records the outcome as `TIMEOUT`.

Here's the awkward part: a job that times out has typically been running the
*whole* time it was allowed. It used the full 8 hours of the room. So it consumed
the maximum energy — and if it did not save its work before being killed, that
energy produced nothing usable.

### 1.2 Why that is worth caring about

On one published cluster, jobs that timed out were about **6.6% of all jobs but
about 45% of all job energy**. A small slice of jobs, holding an outsized share
of the electricity bill. That is the whole reason this project exists.

### 1.3 The idea: notice before it happens

You cannot un-burn energy. But you might be able to **warn before the job
starts**. If a job is very likely to time out, a site could suggest a better
time limit, fix the job, or decline it — whatever the site's policy is.

So the central technical question becomes a prediction question:

> Given how a user's last 24 jobs behaved, will their **next** job hit its time
> limit?

That is the single prediction this project is built around. It is called "T1".

### 1.4 The honest limit: "at stake" is not "saved"

This distinction is the spine of the whole project, and the tool enforces it in
code. The two phrases are easy to confuse, because both are measured in the same
unit (kilowatt-hours) and both are about the same jobs. The difference is not in
the units — it is in **what you have to prove before you are allowed to say the
number**.

- **Energy at stake** is a *diagnosis*: "these are the jobs we flagged, and this
  is how much energy they consumed." It is a statement about **records**, and we
  can compute it today.
- **Energy saved** is an *outcome*: "because someone acted, the cluster actually
  used this much less." It is a statement about **what happened in the world
  afterwards**, and we cannot compute it from records at all.

#### The gap is the word "acted"

A flag changes nothing by itself. The job still ran. It still hit its limit. The
energy was still burned. At the moment a report is written, the energy saved is
**zero** — no matter how large the number at stake is.

At stake is the size of the *opportunity*. Saved is how much of that opportunity
was actually realised.

#### What each number requires

| | Energy **at stake** | Energy **saved** |
|---|---|---|
| Comes from | Summing energy over flagged jobs in the records | Comparing the real world before and after an intervention |
| Needs someone to act? | No | **Yes** |
| Needs a baseline — "what would have happened otherwise"? | No | **Yes** |
| Computable today? | Yes | No — the action has not happened yet |

That third row is the heart of it. To claim a saving you must know the
**counterfactual** — what the cluster would have used if nobody had acted.
Records alone cannot tell you that. Measuring it properly is a separate
discipline called **M&V** (measurement and verification): agree a baseline
period, make a change, then measure the period after, ideally with a comparison
group that was not changed.

#### A worked example

A user requests **48 hours** for a job that historically finishes in about two.

1. The job runs the full 48 hours, hits its limit, and consumes — say —
   **300 kWh**.
2. Our model had flagged it in advance. So **energy at stake = 300 kWh**. That
   is real, and it is in the records.
3. Now the site warns the user. The user re-requests 4 hours, the job finishes in
   2, and it uses about **12 kWh**.
4. The difference — about **288 kWh** — is the *saving*. But it can only be
   claimed if we can show the job would otherwise have run 48 hours again, and
   only if the node it freed was not simply handed to the next waiting job.

Before any of that happens, the energy saved is **zero**. All we have is 300 kWh
*at stake*.

#### Why "saved" is usually much smaller than "at stake"

Three reasons, and they compound:

- **Not every flag is acted on.** A flag is advice, not an instruction.
- **Not every action works.** The user may ignore it, or the job may genuinely
  need the time it asked for.
- **On a shared cluster, freed capacity is often reused.** Cancel a job, free a
  node, and the scheduler hands that node to the next job in the queue. The
  site's total energy may barely move; what improved is throughput. (The project
  treats this as an open question: do savings show up as lower energy use, or as
  more work done?)

#### Why the project is strict about it

Because of everything above, this project reports **at stake** and never claims
savings. The published tool enforces it literally: it **refuses to print** a
report containing the phrase "energy saved", and says so on every run —

> *"That is energy AT STAKE, not energy saved. A flag saves nothing until
> someone acts on it."*

Real savings are reserved for a later, separate step of the benchmark, where a
savings method is tested against a **simulated** intervention — precisely because
you need a before-and-after to measure one.

---

## 2. What this project is

### 2.1 In one sentence

We are running a **fair contest between four different ways of finding wasted
compute**, on the same data, and writing down which one wins.

Strictly, the contest is about *timeout exposure* — finding jobs that will hit
their limit — but the wider ambition is finding wasted compute generally.

### 2.2 The four contestants ("arms")

Each approach is called an **arm**, borrowed from clinical trials. They are
deliberately different in *kind*, not just in settings:

| Arm | Plain description | Trained on our data? | Privacy |
|---|---|---|---|
| **A — Lattice24** | A small, published method: 4 statistics from a user's last 24 jobs, fed to a simple logistic regression. | Yes (tiny) | Fully offline |
| **B — Conventional ML** | What a competent machine-learning team would build without special cluster knowledge: boosted trees, neural nets, a sequence model. | Yes | Fully offline |
| **C — Jev** | A commercial AI model used *zero-shot* — we ask it questions, we do not train it. | No | Data leaves the site (API) |
| **D — Peepalytics** | Everything: domain rules ("detectors"), the best of B, plus explanations and an evidence trail. | Yes | Mostly offline |

Arm A is the **published baseline** — the method someone else already
documented. Arm D is the **full system** the project may eventually build. The
purpose of the contest is to find out whether the extra machinery in B, C and D
actually earns its keep.

### 2.3 The five tasks ("T1–T5")

An "arm" is an approach; a "task" is a question you ask it. Not every approach
can attempt every task — and a blank is recorded as "not applicable", never as a
zero.

| Task | The question, plainly | Who can attempt it |
|---|---|---|
| **T1 Timeout** | Will this job hit its time limit? | A, B, C, D |
| **T2 Failure** | Will this job fail (crash, out of memory, node failure)? | B, C, D |
| **T3 Right-sizing** | How much of the time and cores it asked for will it actually use? | B, C, D |
| **T4 Root cause** | *Why* did this happen, and what should the site do? | B, C, D |
| **T5 Traceability** | Can every claim be traced back to the specific job records? | C, D |

Arm A — the published method — does **only T1**. That is not a criticism; it is
the honest scope of a four-feature model.

---

## 3. The data we are allowed to learn from

### 3.1 Three sources, each good at something different

We have three public datasets. They are **not** the same thing in different
formats — they are different machines, different periods, and different
strengths.

| | **Eagle "11M"** | **Eagle 3-month** | **Kestrel** |
|---|---|---|---|
| Where from | NREL Eagle archive | NREL Eagle (submission 152) | NLR Kestrel (submission 302) |
| Size | 11,030,377 jobs | 3 months (~412k jobs) | 10,559,977 jobs |
| Period | 52 consecutive months (2018-11 → 2023-02) | 3 separate months (2019-12, 2020-04, 2020-08) | 29 consecutive months (2023-08 → 2025-12) |
| Energy data? | **None** | Yes — but *modelled* | Yes — and *measured* |
| Role | The main dataset | Reproduction of a published result | The second machine (replication) |

### 3.2 Why we do not glue them into one big file

It is tempting to concatenate everything into one giant table. We deliberately
do not, for three reasons:

1. **The energy numbers are not the same kind of number.** Kestrel's energy is
   *measured* by the machine. Eagle's is *modelled* — calculated from average
   power × time. Adding a measured number to a modelled number gives a figure
   that is neither. We keep them apart and label each one.
2. **They are different computers.** Different users, different queue policy.
   The interesting question is whether a method *travels* from one machine to
   another. Merging them destroys exactly the evidence we want.
3. **They have different shapes.** Different months, different columns.

So the rule is: **one standard format, three separate datasets, scored
separately.**

### 3.3 Energy tiers — measured, modelled, estimated

Every energy number carries a label saying how it was obtained:

| Tier | Meaning | Where we have it |
|---|---|---|
| **Measured** | The machine reported the actual energy. | Kestrel (25 of 29 months) |
| **Modelled** | Calculated from average power × nodes × time. | Eagle 3-month |
| **Estimated** | Calculated from hardware specifications. | Not used |
| **None** | No energy information at all. | Eagle 11M |

**Tiers are never mixed.** This single rule is why we can use messy, inconsistent
data honestly instead of pretending it is uniform.

---

## 4. Turning raw data into something usable (the "ingest")

### 4.1 The problem

Raw scheduler exports are like a filing cabinet someone emptied onto the floor:
different column names, different date formats, different units. Every one of
the four approaches would otherwise have to re-learn how to read each dataset —
and would probably each do it slightly differently, which would make the contest
unfair.

So we do the cleaning **once**, in a component called **ingest**, and produce a
**canonical table** — one standard layout that everybody downstream reads.

> **The governing rule:** format knowledge lives *only* in ingest. Nothing
> downstream — no arm, no evaluation — ever needs to know which machine a row
> came from.

### 4.2 The canonical table

The canonical table has one row per job, with columns like job id, user (hashed),
partition, state, submit/start/end times, requested and used wall-clock time,
CPU/node/memory counts, energy in joules, and its energy tier. Columns a dataset
does not have are present but empty — never missing — so the shape is always
identical.

### 4.3 The four inconsistencies ingest has to absorb

**1. Identifiers (names).** We must never publish someone's username. Raw names
are turned into one-way codes ("hashes"). Two of our sources already ship codes
rather than names, so for those we simply pass them through.

**2. States.** The scheduler reports outcomes verbosely — `CANCELLED by 1234`
means a person cancelled a job; the "by 1234" is noise. We keep the first word
and discard the rest. We also discovered a state the original list had missed
(`DEADLINE`) and added it.

**3. Durations.** How long a job asked for and how long it ran are stored
**four different ways** across our three datasets:

| Style | Looks like | Means |
|---|---|---|
| Plain seconds | `36000.0` | 36,000 seconds |
| ISO-8601 | `P0DT0H30M0S` | 0 days, 0 hours, 30 minutes, 0 seconds |
| Slurm style | `1-00:00:00` | 1 day |
| Arrow duration | `43200` (nanoseconds) | 43,200 seconds |

There is a classic trap here: a bare integer that should mean *seconds* can be
read as *minutes*, silently multiplying every time limit by 60. Ingest therefore
declares the format per dataset rather than guessing.

**4. Timezones.** Timestamps must all be on the same clock. We found Kestrel's
files are **not** uniform: 24 are UTC, but 5 are at −07:00 or −06:00. Each file
is converted to UTC before anything is combined.

### 4.4 What we drop, and why

| Dropped | Why |
|---|---|
| Jobs still running or waiting | They have no final outcome to learn from. |
| Jobs with no used-time recorded | The job never actually ran — it was cancelled or killed while queued. (This removes 1.24M Kestrel rows — about 12%.) |
| Jobs with no time limit | You cannot compute "how close to the limit did it run" without a limit. |
| Anything identifying | Job names, working directories and command lines are never read. |

Everything dropped is **counted**, so the numbers always add up and nothing
vanishes silently.

### 4.5 The audit trail

Each ingest writes two files: the canonical table, and an **audit file** — a
small JSON record of how many rows went in, how many were dropped and why, the
state and energy breakdowns, and a fingerprint (hash) of the output. Anyone can
check the numbers reconcile, and confirm a re-run produced the identical file.

---

## 5. Splitting the data fairly (the "splits")

### 5.1 Why splitting is a big deal

A prediction question needs a fair test: teach on some jobs, then check the
answers on jobs the model has never seen. If you test on the same jobs you
learned from, you are marking your own homework.

For a contest, the split must be **identical for every arm**. Otherwise you are
comparing plumbing, not methods.

### 5.2 Forward chaining, explained

We never shuffle time. Instead we walk forward:

> Train on January. Test on February. Train on January–February. Test on March.
> And so on.

This mirrors real life — you always predict the future from the past — and it
makes accidental cheating ("leakage") much harder. We report the **median
result across months**, plus how much those months varied, rather than a single
average that could hide a bad month.

### 5.3 The locked six months

The **last six months of each dataset are locked in a safe**.

No tuning, no model selection, no peeking, until every arm is finished. Then
they are scored **once**. The reason is a well-known trap: if you keep checking
your score against the same months, you slowly tune your way into a
flatteringly good result without noticing. Locking the last six months removes
that temptation.

> **Note:** "six months" here is a *window of data*, not a six-month project.
> The project plan is eight weeks.

### 5.4 The 20,000-job sample

One arm (Jev, the commercial AI) is charged per question and cannot be run over
millions of jobs. So every arm is additionally scored on the **same 20,000
job** mini-set. It is not a random handful — it is "stratified", meaning it keeps
the same mix of outcomes (timeouts, failures, successes) and the same mix of
light and heavy users as the full data. The list of those 20,000 jobs is frozen
in advance, before any results exist.

### 5.5 No leakage

The rule, in plain words: *a feature used to predict a job may only come from
jobs that had already **finished** before that job was **submitted**.*

Two subtleties we handle:

- We order and group jobs by **when they were submitted**, not when they ended.
  Ordering by end time can let a job that started *after* our target sneak into
  the target's history — using the future to predict the past.
- A user needs at least **25 jobs** before any of theirs can be scored, because
  the method looks back 24 jobs.

### 5.6 Same rows in, different work inside

This is the heart of the contest, and it is easy to misread:

- Every arm receives **the same rows** — identical training and test sets.
- Every arm is free to build **different features** from those rows.

That difference *is* the experiment. If every arm had to use the same features,
the contest would be pointless. Arm A gets four simple statistics; Arm B gets
richer job details; Arm D gets those plus domain rules. Same raw material,
different thinking.

---

## 6. What we have built so far

### 6.1 The reference tool (Arm A)

The published method already existed as a small Python program —
`lattice24_assess`. It reads a scheduler export, computes the four statistics,
fits the logistic regression, and writes a report.

It is deliberately **left frozen and untouched**. It *is* Arm A, and the contest
requires Arm A to run exactly as published. We did verify it works: on a
12-month slice of the Eagle data it scored very close to the published accuracy.

Its design has one strong opinion we adopted project-wide: it **refuses to
print a number it cannot justify** — too little data, too few timeouts, or a
sanity check failing, and it stops rather than guessing.

### 6.2 The harness

Everything else lives in a new, separate folder, `bench/`. So far it contains:

- **The ingest core** — the shared machinery described in section 4.
- **An Eagle 3-month adapter** — proven by reproducing the published numbers
  (see below).
- **A Kestrel adapter** — all 29 monthly files converted: 10,559,977 rows in,
  9,320,707 clean rows out.
- **An automated test suite** — 18 tests, all passing.

Two checks that matter: re-running the ingest produces a **byte-identical**
file (so results are reproducible), and every dataset is checked against a list
of structural rules before it is accepted.

### 6.3 A high point worth noting

From the raw Eagle month of December 2019, our pipeline produced on its own:

| | Our pipeline | The published write-up |
|---|---|---|
| Timeouts as a share of jobs | 6.60% | 6.6% |
| Timeouts as a share of energy | 45.4% | 45% |
| Users excluded for too little history | 74 of 194 | 74 of 194 |

That is independent confirmation that the pipeline is reading the data
correctly — we reproduced someone else's published result from raw files.

### 6.4 What is not built yet

- The Eagle 11M adapter (the largest dataset).
- The **split generator** — the component that writes the frozen train/test
  plan everybody will use.
- Arms B, C and D themselves.

---

## 7. What the real data taught us

Things we could not have known from documentation, discovered by looking:

1. **Identifiers were already anonymised** in two of the three datasets, so we
   pass them through instead of anonymising again.
2. **`memory_req` (memory requested) on Kestrel looks like a placeholder**, not a
   real number — the same "500000G" appears on both tiny and enormous jobs. We
   parse it but flag it as unusable until resolved.
3. **Some Kestrel columns are completely empty** (array position, GPU counts,
   memory efficiency). We leave them empty rather than inventing values, which
   means a few planned features are Eagle-only.
4. **`nodes_used` looked wrong at first** (it appeared to equal CPU counts), but
   a full check showed it matches the machine's own node list perfectly — the
   oddities were all on jobs that never ran, which we drop anyway.
5. **`DEADLINE` jobs are almost all queue endings, not mid-run kills** — 32,071
   in the raw data, but only 6 survived the "never ran" filter.
6. **Energy coverage is uneven** — some months have no energy readings at all,
   and even within good months up to half the rows are missing energy. Any
   energy number must say which rows it covers.
7. **One number does not match.** Our Kestrel timeout energy total is about
   **16% higher** than the figure in the published write-up. We tested the
   obvious explanations and none fits; on the one month we can compare directly
   we are only 1.3% off, which suggests the archive itself differs from the one
   used in the write-up. We recorded it rather than massaging the numbers to
   match.

---

## 8. Every decision, and why

| # | Decision | What it means in plain terms | Why |
|---|---|---|---|
| 1 | One format, three separate datasets | Clean everything into the same layout, but never merge the datasets. | Mixed energy types cannot be added; different machines must stay separate for a fair "does it travel?" test. |
| 2 | Freeze the six months | Reserve the last six months of each dataset, score once at the end. | Stops us tuning against the test data by accident. |
| 3 | Split by **submit** time | A job's place in time is when it was submitted. | Prevents using jobs that started later to predict an earlier job. |
| 4 | Expand the training window | Train on *all* past months, not a fixed recent slice. | Matches the published method; avoids an arbitrary extra setting. |
| 5 | Keep 20,000-job sample from the *open* months | The mini-set excludes the locked holdout. | Keeps the sample usable during development without touching the sealed data. |
| 6 | Define "user activity" as jobs-per-user | For sample fairness, users are grouped by how many jobs they run, on a log scale. | A few giant accounts would otherwise dominate and become "the result". |
| 7 | Keep the 4 energy-less Kestrel months | Use them for accuracy only; never for energy. | The data is what it is; excluding them throws away evidence, but they cannot contribute energy. |
| 8 | The 3-month Eagle set is reproduction-only | Never used for the main train/test contest. | Its months are not consecutive, so a fair month-by-month test is impossible. |
| 9 | Same rows in, different features inside | Identical data to every arm; each arm engineers its own features. | The feature philosophy *is* what is being compared. |
| 10 | Never mix energy tiers | Measured, modelled and estimated energy stay labelled and separate. | Adding them produces a number that means nothing. |
| 11 | Report "at stake", never "saved" | We report energy in flagged jobs, never claimed savings. | Nothing is saved until someone acts; the tool even refuses to print savings claims. |
| 12 | Refuse rather than guess | With too little data, the tool stops and explains. | A confident wrong number is worse than no number. |
| 13 | Stable, per-dataset anonymising codes | Names become one-way codes; two sources already provide them. | Keeps the same job traceable across the project without storing names. |
| 14 | Everything reproducible | Same input ⇒ identical output file, verified by fingerprint. | Results can be checked by anyone, later. |
| 15 | Keep the reference tool untouched | `lattice24_assess` is never modified. | It is the published baseline; changing it would invalidate the comparison. |
| 16 | Build the harness in a separate folder | New code lives in `bench/`, apart from the reference tool. | Keeps Arm A pristine and the harness clearly ours. |
| 17 | `DEADLINE` counts as a negative | Deadline kills are **not** counted as timeouts; the number of them flagged is reported anyway. | Keeps the headline comparable to the published method; almost all deadline jobs were killed while queued, so the effect is tiny. |

---

## 9. What is still open

One question is deliberately unresolved, and is written down rather than quietly
decided:

- **How to report the Kestrel energy mismatch** (section 7.7). The options are
  to report the discrepancy and investigate it separately, or to investigate
  first. What we must not do is adjust our numbers to match the published one.

The earlier `DEADLINE` question is settled: deadline kills are **not** counted as
timeouts for the headline task (which keeps it comparable to the published
method), but the number of deadline jobs flagged is reported anyway so the
choice is visible.

---

## 10. Plain-language glossary

| Term | What it means |
|---|---|
| **Cluster** | A big shared computer that runs many people's jobs. |
| **Job** | One unit of work submitted to the cluster. |
| **Wall-clock limit** | The maximum time a job is allowed to run. |
| **TIMEOUT** | The job was killed because it reached that limit. |
| **Arm** | One competing approach in the contest (A/B/C/D). |
| **Task (T1–T5)** | One question an arm is asked. |
| **Ingest** | Turning messy raw files into one clean, standard table. |
| **Canonical table** | That standard table — same columns for every dataset. |
| **Feature** | A summary number handed to a model (e.g. "average of the last 24 jobs"). |
| **Label / target** | The thing being predicted (here: did the job time out?). |
| **Window** | The last 24 jobs of a user, used to predict their next one. |
| **Split** | The division into "learn from this" and "test on that". |
| **Forward chaining** | Always learning from the past and testing on the following month. |
| **Leakage** | Accidentally using future information to predict the past. |
| **Holdout / locked months** | Data sealed away, used once at the very end. |
| **Stratified sample** | A small subset that keeps the same mix as the whole. |
| **Energy tier** | How an energy number was obtained: measured, modelled, or estimated. |
| **At stake** | Energy in the flagged jobs — the opportunity, not a saving. |
| **M&V** | Measuring real savings after an intervention; a separate discipline. |
| **Hash** | A one-way code standing in for a name, so nobody can be identified. |
| **Determinism / reproducibility** | Same input always produces the same output. |

---

## 11. Where to read more

| Document | What it covers |
|---|---|
| `prd.md` | The benchmark specification: arms, tasks, metrics, gates, timeline. |
| `plan.md` | The data-layer design: how data is standardised and split, plus the decision log. |
| `canonical_table.md` | The exact column-by-column rules for turning raw data into the standard table. |
| `folds_manifest.md` | The exact contract for the frozen train/test plan. |
| `bench/README.md` | Practical status of the harness code and its verification results. |
| `21913139/WRITEUP.md` | The published research this project builds on. |
