"""
kestrel_fetch.py — pull selected months out of the 697 MB Kestrel zip without
downloading the zip.

This copy lives in the repository and is the one `fetch_all_data.sh` runs. The
deposit at 21913139/ ships an older version with no command line: it only
*lists* the archive, so asking it to download silently does nothing. Keep this
file as the source of truth for fetching.

Zip members are stored contiguously and the central directory sits at the end,
so with HTTP range requests you can:
  1. range-GET the last 64 kB, find the End Of Central Directory record
  2. range-GET the central directory, parse every member's name, compressed
     size and local-header offset
  3. range-GET just the members you want and inflate them

USAGE
-----
    python3 kestrel_fetch.py                        # list the archive (no download)
    python3 kestrel_fetch.py --list
    python3 kestrel_fetch.py --all                  # fetch EVERY .parquet member
    python3 kestrel_fetch.py --month 2023-08 --month 2023-09
    python3 kestrel_fetch.py --all --out kestrel --dry-run
    python3 kestrel_fetch.py --all --overwrite      # re-fetch what already exists

Notes
-----
* Running with no selection flags only *lists* the archive — nothing is
  downloaded. Fetching requires --all or at least one --month.
* --month matches by substring against the member name (e.g. "2023-08"), so it
  is tolerant of naming variations. If nothing matches, the available names are
  printed.
* Existing files are skipped unless --overwrite is given, so an interrupted
  `--all` run can simply be repeated.
* --all downloads roughly the whole archive (the monthly parquet members sum to
  most of the 697 MB). Use --dry-run to see the exact total first.

Data (c) DOE/NLR/Alliance, NLR Data Catalog submission 302. Free use and copying
provided the full licence notice travels with any copy, with credit to
DOE/NLR/ALLIANCE and no implied endorsement. Do not attempt to re-identify
individuals from the hashed fields.
"""
import argparse
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

URL = ("https://data.nlr.gov/system/files/302/"
       "1773544299-esif.hpc.kestrel.job-anon.zip")
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

# curl behaviour shared by every range request: follow redirects (mandatory --
# the host 302s to a presigned S3 URL), retry transient failures, timeout.
CURL_BASE = ["curl", "-sL", "--max-time", "300", "--retry", "4",
             "--retry-all-errors", "--retry-delay", "3", "-A", UA]


def curl_range(start, end):
    """Range-GET bytes [start, end] inclusive. Returns bytes."""
    out = subprocess.run(CURL_BASE + ["-r", f"{start}-{end}", URL],
                         capture_output=True, check=True)
    return out.stdout


def total_size():
    """Read the true size from Content-Range on a 1-byte ranged GET.

    A HEAD request is useless here: data.nlr.gov answers with a 302 whose own
    Content-Length is the ~7 kB HTML redirect page, and following it with -I
    does not reliably HEAD the presigned S3 URL. A ranged GET does follow, and
    Content-Range carries the real total after the slash.
    """
    out = subprocess.run(
        ["curl", "-sL", "-D", "-", "-o", "/dev/null", "--max-time", "180",
         "-A", UA, "-r", "0-0", URL],
        capture_output=True, text=True, check=True)
    for line in out.stdout.splitlines():
        if line.lower().startswith("content-range:"):
            return int(line.split("/")[-1].strip())
    raise RuntimeError("no Content-Range -- server may not honour byte ranges")


def central_directory(total):
    tail = curl_range(max(0, total - 65536), total - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("EOCD not found")
    n_entries = struct.unpack("<H", tail[i + 10:i + 12])[0]
    cd_size = struct.unpack("<I", tail[i + 12:i + 16])[0]
    cd_off = struct.unpack("<I", tail[i + 16:i + 20])[0]
    cd = curl_range(cd_off, cd_off + cd_size - 1)

    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        comp_size = struct.unpack("<I", cd[p + 20:p + 24])[0]
        name_len = struct.unpack("<H", cd[p + 28:p + 30])[0]
        extra_len = struct.unpack("<H", cd[p + 30:p + 32])[0]
        cmt_len = struct.unpack("<H", cd[p + 32:p + 34])[0]
        lh_off = struct.unpack("<I", cd[p + 42:p + 46])[0]
        name = cd[p + 46:p + 46 + name_len].decode("utf-8", "replace")
        entries.append({"name": name, "comp_size": comp_size, "lh_off": lh_off})
        p += 46 + name_len + extra_len + cmt_len
    return entries, n_entries


def fetch_member(e):
    """Range-GET one member and inflate it. Returns the decompressed bytes."""
    blob = curl_range(e["lh_off"], e["lh_off"] + e["comp_size"] + 512)
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {e['name']}")
    nl = struct.unpack("<H", blob[26:28])[0]
    xl = struct.unpack("<H", blob[28:30])[0]
    method = struct.unpack("<H", blob[8:10])[0]
    data = blob[30 + nl + xl: 30 + nl + xl + e["comp_size"]]
    if method == 0:
        return data
    return zlib.decompress(data, -15)


def human(n):
    """Bytes as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def parquet_members(entries):
    """Only the .parquet members of the archive."""
    return [e for e in entries if e["name"].lower().endswith(".parquet")]


def select(members, months, want_all):
    """Choose which members to fetch from --all / --month selections."""
    if want_all:
        return members
    return [e for e in members if any(m in e["name"] for m in months)]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="List or fetch monthly parquet members from the remote "
                    "Kestrel zip using HTTP range requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--all", action="store_true",
                    help="fetch every .parquet member in the archive")
    ap.add_argument("--month", action="append", default=[], metavar="YYYY-MM",
                    help="fetch members whose name contains this string "
                         "(repeatable, e.g. --month 2023-08 --month 2023-09)")
    ap.add_argument("--out", default="kestrel",
                    help="output directory for fetched members (default: ./kestrel)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-fetch members that already exist on disk")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be fetched, download nothing")
    ap.add_argument("--list", action="store_true",
                    help="list archive contents and exit (default when no "
                         "selection is given)")
    args = ap.parse_args(argv)

    # The central directory is needed even for a listing-only run.
    total = total_size()
    print(f"zip total: {total:,} bytes ({human(total)})")
    entries, n = central_directory(total)
    members = parquet_members(entries)
    print(f"central directory: {n} entries, {len(members)} parquet\n")

    # Listing mode: no selection given, or --list explicitly.
    if args.list or (not args.all and not args.month):
        for e in sorted(members, key=lambda x: x["name"]):
            print(f"  {e['name']:<70}{human(e['comp_size']):>12}")
        if not args.list:
            print("\nNothing fetched. Use --all or --month YYYY-MM to download.")
        return 0

    chosen = select(members, args.month, args.all)
    if not chosen:
        print(f"\nNo members matched {args.month!r}. Available names:",
              file=sys.stderr)
        for e in sorted(members, key=lambda x: x["name"]):
            print(f"  {e['name']}", file=sys.stderr)
        return 2

    plan_bytes = sum(e["comp_size"] for e in chosen)
    print(f"selected {len(chosen)} member(s), {human(plan_bytes)} compressed:")
    for e in sorted(chosen, key=lambda x: x["name"]):
        print(f"  {e['name']:<70}{human(e['comp_size']):>12}")

    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    fetched = skipped = failed = 0
    done_bytes = 0
    for e in sorted(chosen, key=lambda x: x["name"]):
        name = os.path.basename(e["name"])
        dst = outdir / name
        if dst.exists() and not args.overwrite:
            print(f"  skip (exists): {name}")
            skipped += 1
            continue
        try:
            data = fetch_member(e)
        except Exception as exc:                      # keep going on one failure
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        dst.write_bytes(data)
        done_bytes += len(data)
        fetched += 1
        pct = 100.0 * done_bytes / max(plan_bytes, 1)
        print(f"  fetched: {name:<60}{human(len(data)):>10}  ({pct:4.0f}%)")

    print(f"\ndone: {fetched} fetched, {skipped} skipped, {failed} failed "
          f"-> {outdir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
