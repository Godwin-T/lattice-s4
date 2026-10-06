#!/usr/bin/env bash
#
# fetch_all_data.sh — get every dataset the benchmark needs, from scratch.
#
#   git clone <repo> && cd <repo>
#   bash scripts/fetch_all_data.sh              # download only
#   bash scripts/fetch_all_data.sh --build      # download, then ingest + split
#
# Everything lands outside version control (see .gitignore). Re-running is safe:
# each step is skipped if its output is already present.
#
# Four sources are fetched:
#   1. the published record (Zenodo 21913139) — the write-up, the licence files,
#      and the two fetch scripts this one drives;
#   2. Eagle 3-month job records (data.nlr.gov submission 152);
#   3. Kestrel monthly Parquet (data.nlr.gov submission 302) — pulled member by
#      member over HTTP range requests, so the 697 MB zip is never downloaded;
#   4. Eagle 11M Parquet (OEDI 5860, data.openei.org/files/5860).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
ZENODO_DIR="${ZENODO_DIR:-$REPO_ROOT/21913139}"
ZENODO_RECORD="${ZENODO_RECORD:-21913139}"
EAGLE_11M_URL="${EAGLE_11M_URL:-https://data.openei.org/files/5860/eagle_data.parquet}"
SALT_FILE="$REPO_ROOT/.bench_salt"

log()  { printf '\n== %s\n' "$*"; }
warn() { printf '\n!! %s\n' "$*" >&2; }

mkdir -p "$DATA_DIR" "$ZENODO_DIR"

# ---------------------------------------------------------------------------
# 1. The published record
# ---------------------------------------------------------------------------
if [ -f "$ZENODO_DIR/WRITEUP.md" ]; then
    log "Zenodo record already present in $ZENODO_DIR"
else
    log "Fetching Zenodo record $ZENODO_RECORD"
    python3 - "$ZENODO_RECORD" "$ZENODO_DIR" <<'PY'
import json, os, sys, urllib.request, zipfile

record, outdir = sys.argv[1], sys.argv[2]
api = f"https://zenodo.org/api/records/{record}"
print(f"  querying {api}")
with urllib.request.urlopen(api, timeout=120) as response:
    meta = json.load(response)

files = meta.get("files", [])
if not files:
    sys.exit(f"record {record} lists no files (is it public?)")

for entry in files:
    name = entry["key"]
    url = entry["links"]["self"]
    dest = os.path.join(outdir, name)
    if os.path.exists(dest):
        print(f"  have     {name}")
        continue
    print(f"  fetching {name}")
    urllib.request.urlretrieve(url, dest)
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(dest) as archive:
            archive.extractall(outdir)
        print(f"  unzipped {name}")
PY
fi

# ---------------------------------------------------------------------------
# 2. Eagle 3-month job records (NLR 152)
# ---------------------------------------------------------------------------
if compgen -G "$ZENODO_DIR/anon_jobs_*.json" > /dev/null; then
    log "Eagle 3-month records already present"
else
    log "Fetching Eagle 3-month records (data.nlr.gov submission 152)"
    if [ ! -f "$ZENODO_DIR/fetch_data.sh" ]; then
        warn "fetch_data.sh missing — step 1 did not complete"
        exit 1
    fi
    ( cd "$ZENODO_DIR" && bash fetch_data.sh )
fi

# ---------------------------------------------------------------------------
# 3. Kestrel monthly Parquet (NLR 302)
# ---------------------------------------------------------------------------
if compgen -G "$ZENODO_DIR/kestrel/*.parquet" > /dev/null; then
    log "Kestrel months already present"
else
    log "Fetching Kestrel months (data.nlr.gov submission 302)"
    if [ ! -f "$ZENODO_DIR/kestrel_fetch.py" ]; then
        warn "kestrel_fetch.py missing — step 1 did not complete"
        exit 1
    fi
    python3 "$ZENODO_DIR/kestrel_fetch.py" --list
    python3 "$ZENODO_DIR/kestrel_fetch.py" --all --out "$ZENODO_DIR/kestrel"
fi

# ---------------------------------------------------------------------------
# 4. Eagle 11M Parquet (OEDI 5860)
# ---------------------------------------------------------------------------
# Override EAGLE_11M_URL if the link ever moves, or drop the file in place
# yourself. Everything else still works without this dataset.
if [ -f "$DATA_DIR/eagle_data.parquet" ]; then
    log "Eagle 11M already present"
else
    log "Fetching Eagle 11M (OEDI 5860, ~253 MB)"
    curl -L --fail --retry 4 --retry-all-errors --retry-delay 3 \
         -o "$DATA_DIR/eagle_data.parquet" "$EAGLE_11M_URL"
fi

# ---------------------------------------------------------------------------
# 5. A stable hash salt (generated once per clone, never committed)
# ---------------------------------------------------------------------------
if [ ! -f "$SALT_FILE" ]; then
    log "Generating a local hash salt (kept in .bench_salt, git-ignored)"
    head -c 32 /dev/urandom | base64 | tr -d '\n' > "$SALT_FILE"
    printf '\n' >> "$SALT_FILE"
fi
export BENCH_SALT_EAGLE_PARQUET="$(tr -d '\n' < "$SALT_FILE")"

# ---------------------------------------------------------------------------
# 6. Verify what landed
# ---------------------------------------------------------------------------
log "Checking the downloads"
python3 - "$DATA_DIR" "$ZENODO_DIR" <<'PY'
import glob, os, sys

data_dir, zenodo_dir = sys.argv[1], sys.argv[2]
checks = [
    ("Eagle 11M Parquet",  [os.path.join(data_dir, "eagle_data.parquet")]),
    ("Eagle 3-month JSON", sorted(glob.glob(os.path.join(zenodo_dir, "anon_jobs_*.json")))),
    ("Kestrel Parquet",    sorted(glob.glob(os.path.join(zenodo_dir, "kestrel", "*.parquet")))),
]
missing = 0
for label, files in checks:
    if files:
        total = sum(os.path.getsize(f) for f in files)
        print(f"  ok    {label:20} {len(files):>3} file(s), {total / 1e6:,.1f} MB")
    else:
        missing += 1
        print(f"  MISSING {label}")

# A partial or wrong-parquet download is worse than none, so check the shape.
eagle = os.path.join(data_dir, "eagle_data.parquet")
if os.path.exists(eagle):
    try:
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(eagle).metadata.num_rows
        expected = 11_030_377
        flag = "ok   " if rows == expected else "CHECK"
        print(f"  {flag} Eagle 11M rows       {rows:,} (expected {expected:,})")
    except Exception as exc:                      # noqa: BLE001 - report and continue
        print(f"  CHECK Eagle 11M unreadable: {exc}")

sys.exit(1 if missing == 3 else 0)
PY

# ---------------------------------------------------------------------------
# 7. Optionally build the derived tables and the frozen splits
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--build" ]; then
    failures=0
    step() {
        local label="$1"; shift
        log "$label"
        if ! "$@"; then
            warn "FAILED: $label (continuing with the remaining steps)"
            failures=$((failures + 1))
        fi
    }

    if compgen -G "$ZENODO_DIR/anon_jobs_*.json" > /dev/null; then
        step "Canonical table: Eagle 3-month" \
            python3 -m bench.ingest.cli --dataset eagle_jsonl \
            --raw "$ZENODO_DIR"/anon_jobs_*.json --out "$DATA_DIR/canonical"
    else
        warn "skipping Eagle 3-month ingest: no anon_jobs_*.json present"
    fi

    if compgen -G "$ZENODO_DIR/kestrel/*.parquet" > /dev/null; then
        step "Canonical table: Kestrel" \
            python3 -m bench.ingest.cli --dataset kestrel \
            --raw "$ZENODO_DIR"/kestrel/*.parquet --out "$DATA_DIR/canonical"
    else
        warn "skipping Kestrel ingest: no 21913139/kestrel/*.parquet present"
    fi

    if [ -f "$DATA_DIR/eagle_data.parquet" ]; then
        step "Canonical table: Eagle 11M" \
            python3 -m bench.ingest.cli --dataset eagle_parquet \
            --raw "$DATA_DIR/eagle_data.parquet" --out "$DATA_DIR/canonical"
    else
        warn "skipping Eagle 11M ingest: $DATA_DIR/eagle_data.parquet not present"
    fi

    for dataset in eagle_parquet kestrel; do
        if [ -f "$DATA_DIR/canonical/$dataset.parquet" ]; then
            step "Frozen splits: $dataset" \
                python3 -m bench.splits.cli --dataset "$dataset" \
                --canonical "$DATA_DIR/canonical/$dataset.parquet" \
                --out "$DATA_DIR/manifests"
        fi
    done

    if [ "$failures" -gt 0 ]; then
        warn "$failures build step(s) failed — see the messages above"
        exit 1
    fi
fi

log "Done"
cat <<'EOF'

Next steps:
  # score Arm A on one dataset (minutes; --folds keeps it small)
  python3 -m bench.arms.cli --arm A \
      --manifest data/manifests/kestrel.folds.json \
      --canonical data/canonical/kestrel.parquet \
      --out runs --control-folds 5 --threads 2

  python3 -m bench.eval.cli \
      --predictions runs/predictions.parquet \
      --manifest data/manifests/kestrel.folds.json \
      --canonical data/canonical/kestrel.parquet \
      --arm-metadata runs/metadata.json \
      --out results --threads 2
EOF
