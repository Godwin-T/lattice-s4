# bench — benchmark harness

Two components, built in this order, each with a written contract:

1. **ingest** — raw sources → `data/canonical/<dataset>.parquet`
   (spec: `canonical_table.md`)
2. **splits** — canonical table → `data/manifests/<dataset>.folds.json`
   (spec: `folds_manifest.md`)

The rule that keeps it honest: **format knowledge lives only in ingest.**
Everything downstream sees the canonical table and nothing else.

The reference implementation (`lattice24_assess/`, Arm A) is deliberately left
untouched; this package must never import from it.

## Layout

```
bench/
  ingest/
    schema.py       canonical columns + state/tier constants
    durations.py    four duration encodings -> seconds
    states.py       state normalisation
    hashing.py      salted sha256 for sources that ship raw ids
    energy.py       energy channels -> joules + tier
    filters.py      the row drop rules, with counts
    validate.py     canonical_table.md §6 acceptance tests
    adapters/       one module per source
    cli.py          python -m bench.ingest.cli ...
  tests/
```

## Status

- [x] shared ingest core (schema, durations, states, hashing, energy, filters, validate)
- [x] Eagle 3-month JSONL adapter — 11 tests pass; validated against 2019-12
- [x] Kestrel Parquet adapter — 7 tests pass; 29 files ingested
- [x] Eagle 11M Parquet adapter — streamed (chunked) build, salted id hashing
- [x] split generator — folds + frozen sample; 10 tests pass

### Split generator, verified

```
python3 -m bench.splits.cli --dataset <name> --out data/manifests
```

| Dataset | Months | Open | Folds | Locked holdout | Sample |
|---|---|---|---|---|---|
| `eagle_parquet` | 52 | 46 | **45** | 2022-09 → 2023-02 | 20,000 ids, 24 strata |
| `kestrel` | 29 (the 180-row `2026-01` stub dropped) | 23 | **22** | 2025-07 → 2025-12 | 20,000 ids, 25 strata |
| `eagle_jsonl` | 2 | — | refuses (exempt) | — | — |

Frozen manifests (re-running gives the same digest):

| Dataset | Manifest | sha256 |
|---|---|---|
| eagle_parquet | `data/manifests/eagle_parquet.folds.json` | `6af19a6629f6736081b8468254279ae35759630530befc6641d30b9430978a29` |
| kestrel | `data/manifests/kestrel.folds.json` | `6ccae088e72f876b15d3430eef8f7be25fae60c818d06f47d70442423d4c0dfd` |

### Arms

- [x] Arm A — the published Lattice24 method, run inside the harness (4 tests)
- [ ] Arms B1 / B2 / B3, C, D

```
python3 -m bench.arms.cli --arm A \
    --manifest data/manifests/eagle_parquet.folds.json \
    --canonical data/canonical/eagle_parquet.parquet \
    --out runs --threads 2
```

Writes `runs/A-<dataset>-raw/{predictions.parquet,metadata.json}`. Add
`--calibrate` for the isotonic run variant.

**Defaults are chosen to stay polite on a shared machine**, because an earlier
run stalled one:

* `--threads 2`, set *before* Polars / OpenBLAS load so the cap is honoured;
* the window table is built **once** and cached under `<out>/../windows/`, then
  read one fold at a time — peak memory stays flat instead of holding 11M
  windows and re-filtering them 45 times;
* `--folds N` runs only the last N folds, for a quick check;
* `--control-folds N` limits how many folds the shuffle control refits.

**The history rule** (documented at the top of `bench/arms/common.py`): a target
is scored from the 24 jobs that had most recently *ended* before it was
submitted. The naive equivalent — "the previous 24 jobs by submit order, all of
which must have ended" — rejects almost everything on bursty array-job traffic
(measured: 1.5% of rows survive on Eagle, versus 99.6% with the correct rule).

### Evaluator

- [x] `bench/eval/` — implements `metrics.md` E1–E12 (11 tests)
- [ ] E13 — end-to-end on a real arm run (needs Arm A to have been run)

```
python3 -m bench.eval.cli \
    --predictions runs/A-eagle_parquet-raw/predictions.parquet \
    --manifest data/manifests/eagle_parquet.folds.json \
    --canonical data/canonical/eagle_parquet.parquet \
    --arm-metadata runs/A-eagle_parquet-raw/metadata.json \
    --out results --threads 2
```

Writes `results/<run_id>.json`. It refuses a run whose test rows are not each
scored exactly once, suppresses every metric resting on fewer than three months,
reports energy as "not available" (never zero) where there is no reading, and
publishes **no headline figure** when the shuffle control fails.

`--resamples` defaults to **200**, not the 1,000 named in `metrics.md`: 1,000
resamples over a nine-million-row test set is quadratic in cost and would stall a
shared machine. The number actually used is recorded in every result file, and
the flag raises it when a run can afford it.

**Verified against the documented toy examples**: the 700-of-1,000 kWh capture
example, the hand-computed AUC and average-precision values, ECE ≈ 0 on
perfectly-calibrated input, and the shuffle gate admitting 0.51 while rejecting
0.67.

### Eagle 11M, verified against the full archive

`python3 -m bench.ingest.cli --dataset eagle_parquet --raw data/eagle_data.parquet --out data/canonical`

| Quantity | Result |
|---|---|
| Rows in / out | 11,030,377 → 11,014,689 |
| Dropped: non-terminal state | 15,579 (= PENDING 14,646 + RUNNING 933) |
| Dropped: null end / bad timelimit | 2 / 107 |
| TIMEOUT jobs | 802,997 (7.29%) |
| Energy tier | `none` for every row (this source has no energy column) |
| Wall time / peak RSS | ~20 s / 4.2 GB |

The drop counts reconcile exactly against the raw state counts, and the build
streams in 2M-row chunks so an 11M-row table never has to sit in memory as a
whole. Sensitive columns (`name`, `work_dir`, `submit_line`) are excluded from
the read projection, so they are never decoded at all.

**Identifiers are hashed** (`sha256(salt + id)[:16]`) for uniformity, so a salt
is required:

```
python3 -m bench.ingest.cli --dataset eagle_parquet --raw data/eagle_data.parquet \
        --out data/canonical --salt <your-salt>
# or: export BENCH_SALT_EAGLE_PARQUET=<your-salt>
```

Reusing the same salt is essential — a different salt produces different
identifiers and the table stops matching earlier runs. Hashes are computed per
distinct value (936 users), so chunking cannot change them, and re-running with
the same salt gives a byte-identical file.

### Kestrel, verified against the 29-file archive

`python3 -m bench.ingest.cli --dataset kestrel --raw 21913139/kestrel/*.parquet --out data/canonical`

| Quantity | Result |
|---|---|
| Rows in / out | 10,559,977 → 9,320,707 |
| Dropped: never ran (null elapsed) | 1,238,215 |
| Dropped: non-terminal / bad timelimit | 1 / 1,054 |
| TIMEOUT jobs | 559,585 (6.00%) |
| Energy coverage | 84.6% (measured), 15.4% no reading |
| Wall time / peak RSS | ~6 min / 6.1 GB |

Known open discrepancy: the TIMEOUT energy total (10,853,422 kWh) does not match
the published 9,370,367 kWh. Recorded in `canonical_table.md` §4.3.3, not
adjusted.

### Verified against the 2019-12 Eagle month

Running `python3 -m bench.ingest.cli --dataset eagle_jsonl --raw 21913139/anon_jobs_2019-12.json --out data/canonical`
reproduces the write-up's headline numbers from the raw source:

| Quantity | Ingest | Write-up |
|---|---|---|
| TIMEOUT share of jobs | 6.60% | 6.6% |
| TIMEOUT share of job energy | 45.4% | 45% |
| Users with <25 jobs excluded | 74 of 194 | 74 of 194 |

Re-running produces a byte-identical parquet (determinism check passes).
