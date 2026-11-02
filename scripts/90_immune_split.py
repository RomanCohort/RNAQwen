"""RNA-immunity data plumbing: schema, dedup, and leakage-safe splitting.

Why this exists
---------------
Modification and immune-activation datasets love to leak: the same transcript
appears in several assays, one family shows up in train and test, and a model
that memorised a family scores high while learning nothing.  Every vertical arm
must be split by *family*, not by row.

Record schema (one JSON object per line)
----------------------------------------
    uid       str   stable id; defaults to sha1(sequence)[:16]
    sequence  str   RNA, ACGU (T is normalised to U)
    task      str   e.g. "m6a_site", "m5c_site", "rigi_activation"
    label     any   int / float / str, task specific
    source    str   provenance key, e.g. "GEO:GSE123456"
    family    str   grouping key (Rfam clan, transcript id, or k-mer bucket)
    split     str   optional, forced split ("test" pins a row)

Usage
    python scripts/90_immune_split.py --in data/processed/rnaqa_immune.jsonl \
        --out-dir data/processed/immune --val 0.1 --test 0.1

The splitter is greedy by family size and never puts one family on both sides.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ALPHABET = "ACGU"


def norm_sequence(seq: str) -> str:
    s = "".join(c for c in (seq or "").upper() if c.isalpha())
    return s.replace("T", "U")


def make_uid(seq: str) -> str:
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


def kmer_family(seq: str, k: int = 8, modulo: int = 4096) -> str:
    """Fallback grouping when no curated family exists.

    Hashes the multiset of k-mers so near-duplicates land together; it is not a
    substitute for Rfam clans, only a cheap guard against exact repeats.
    """
    kmers = sorted({seq[i:i + k] for i in range(max(0, len(seq) - k + 1))})
    h = hashlib.sha1("|".join(kmers).encode()).hexdigest()
    return f"kmer{int(h[:8], 16) % modulo:04d}"


def load(path: Path):
    rows, dropped, seen = [], 0, set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        seq = norm_sequence(r.get("sequence", ""))
        if not seq or any(c not in ALPHABET for c in seq):
            dropped += 1
            continue
        r["sequence"] = seq
        r["uid"] = r.get("uid") or make_uid(seq)
        r["family"] = r.get("family") or kmer_family(seq)
        key = (r["uid"], r.get("task"))
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        rows.append(r)
    return rows, dropped


def split(rows, val_frac: float, test_frac: float, seed: int = 13):
    by_family = defaultdict(list)
    pinned = {"train": [], "val": [], "test": []}
    for r in rows:
        forced = r.get("split")
        if forced in pinned:
            pinned[forced].append(r)
        else:
            by_family[r["family"]].append(r)

    rng = random.Random(seed)
    fams = list(by_family)
    rng.shuffle(fams)
    # Greedy: biggest families first, so a huge family cannot swamp one split.
    fams.sort(key=lambda f: len(by_family[f]), reverse=True)
    total = sum(len(v) for v in by_family.values()) or 1
    want_val, want_test = total * val_frac, total * test_frac
    out = {"train": list(pinned["train"]), "val": list(pinned["val"]),
           "test": list(pinned["test"])}
    got = {"val": sum(len(x) for x in out["val"]), "test": sum(len(x) for x in out["test"])}
    for f in fams:
        rows_f = by_family[f]
        if got["test"] < want_test and rng.random() < 0.5:
            out["test"].extend(rows_f); got["test"] += len(rows_f)
        elif got["val"] < want_val:
            out["val"].extend(rows_f); got["val"] += len(rows_f)
        else:
            out["train"].extend(rows_f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    rows, dropped = load(Path(a.src))
    parts = split(rows, a.val, a.test, a.seed)
    outdir = Path(a.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    for name, rs in parts.items():
        p = outdir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fams = {r["family"] for r in rs}
        tasks = defaultdict(int)
        for r in rs:
            tasks[r["task"]] += 1
        print(f"{name:5s} rows={len(rs):6d} families={len(fams):5d} tasks={dict(tasks)}")

    fam_train = {r["family"] for r in parts["train"]}
    for name in ("val", "test"):
        overlap = fam_train & {r["family"] for r in parts[name]}
        uid_overlap = {r["uid"] for r in parts["train"]} & {r["uid"] for r in parts[name]}
        print(f"leak check {name}: family_overlap={len(overlap)} uid_overlap={len(uid_overlap)}")
        if overlap or uid_overlap:
            print("  ** LEAK: refusing to call this split clean")
            return 2
    print(f"dropped rows: {dropped}")
    print("split ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())