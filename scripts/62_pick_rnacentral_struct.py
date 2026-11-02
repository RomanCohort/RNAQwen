"""Pick ~200k RNAcentral sequences that are NOT exact duplicates of Rfam.

RNAcentral already de-duplicates at the URS level, but a large share of its
sequences are the very same molecules that Rfam curates.  For a structure
probe we want *new* sequences, so this drops anything whose exact sequence
appears in Rfam, then samples per length bin so short ncRNAs do not dominate.

No torch, no GPU -- safe to run next to the codebook trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RFAM = ROOT / "data/processed/rfam_seqs.jsonl"
SRC = ROOT / "data/processed/rnacentral_2m.jsonl"
OUT = ROOT / "data/processed/rnacentral_struct_200k.jsonl"
BINS = [(20, 50), (50, 100), (100, 200), (200, 300), (300, 401)]


def h(s):
    return hashlib.blake2b(s.encode(), digest_size=16).digest()


def bin_of(n):
    for i, (lo, hi) in enumerate(BINS):
        if lo <= n < hi:
            return i
    return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rfam = set()
    with RFAM.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line).get("sequence", "")
            if s:
                rfam.add(h(s.upper().replace("T", "U")))
    print(f"[rfam] hashes={len(rfam):,}", flush=True)

    counts = defaultdict(int)
    total = 0
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s = (r.get("sequence") or "").upper()
            if not s or h(s) in rfam:
                continue
            b = bin_of(len(s))
            if b < 0:
                continue
            counts[b] += 1
            total += 1
    print(f"[pool] non-duplicate={total:,}  per-bin={dict(counts)}", flush=True)
    if total == 0:
        return 1

    quota = {b: max(1, int(a.n * counts[b] / total)) for b in counts}
    print(f"[quota] {quota}", flush=True)

    # reservoir per bin
    res = {b: [] for b in counts}
    seen = defaultdict(int)
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s = (r.get("sequence") or "").upper()
            if not s or h(s) in rfam:
                continue
            b = bin_of(len(s))
            if b < 0 or b not in quota:
                continue
            seen[b] += 1
            k = quota[b]
            if len(res[b]) < k:
                res[b].append(r)
            else:
                j = rng.randrange(seen[b])
                if j < k:
                    res[b][j] = r

    out = Path(a.out)
    n = 0
    with out.open("w", encoding="utf-8") as w:
        for b in sorted(res):
            for r in res[b]:
                r["struct_bin"] = b
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"[save] {out} rows={n:,}  {out.stat().st_size/1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
