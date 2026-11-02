"""m5C sites, cross-gene negatives matched on |distance-to-TSS| and local GC.

Within-gene negatives saturate around 21k pairs because m5C peaks cluster: in a
promoter cluster there is simply no non-peak window at the same distance from
the TSS.  To scale up we keep the two controls that matter and drop the
"same gene" requirement:

  bin = (chromosome, |dts| log bin, GC of the 101 nt window in 0.05 bins)

A candidate from the same bin, from a different gene, not overlapping any peak
and without N, becomes the negative.  Position-in-gene and local composition
are matched; gene identity is not.

Candidates are pooled first so the matching is 1:1 without reuse.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/immune_site_m5c_binmatched.jsonl"
FASTA = ROOT / "data/raw/genome/hg38.fasta"
SRC = ROOT / "data/raw/immune/GSE247936_MeRIP-seq_m5C_peaks.txt.gz"
W = 101
TEST_CHR = {"chr20", "chr21", "chr22", "chrX"}
VAL_CHR = {"chr17", "chr18", "chr19"}
MAIN = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
DTS_EDGES = [0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 10 ** 9]


def dts_bin(d):
    d = abs(d)
    for i in range(len(DTS_EDGES) - 1):
        if DTS_EDGES[i] <= d < DTS_EDGES[i + 1]:
            return i
    return len(DTS_EDGES) - 2


def gc_bin(seq):
    return int(round((seq.count("G") + seq.count("C")) / max(len(seq), 1) * 20))


def load_peaks():
    out = []
    with gzip.open(SRC, "rt", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 30:
                continue
            try:
                chrom, s, e = p[1], int(p[2]), int(p[3])
                strand = p[5] if p[5] in "+-" else "+"
                gchr = p[19] if p[19].startswith("chr") else "chr" + p[19]
                gs, ge = int(p[20]), int(p[21])
                gstr = "-" if str(p[23]).strip() in ("2", "-1") else "+"
                dts = int(float(p[26]))
                ens, sym = p[27], p[28]
            except (ValueError, IndexError):
                continue
            if chrom not in MAIN or gchr != chrom or ge - gs < 400:
                continue
            out.append(dict(chrom=chrom, s=s, e=e, strand=strand, gs=gs, ge=ge,
                            gstr=gstr, dts=dts, ens=ens, sym=sym))
    return out


def load_genes():
    genes = {}
    with gzip.open(SRC, "rt", errors="replace") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 30:
                continue
            try:
                gchr = p[19] if p[19].startswith("chr") else "chr" + p[19]
                gs, ge = int(p[20]), int(p[21])
                gstr = "-" if str(p[23]).strip() in ("2", "-1") else "+"
                ens = p[27]
            except (ValueError, IndexError):
                continue
            if gchr not in MAIN or ge - gs < 400:
                continue
            genes[ens] = (gchr, gs, ge, gstr)
    return genes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import importlib.util
    spec = importlib.util.spec_from_file_location("f93", ROOT / "scripts/93_genome_windows.py")
    f93 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f93)
    fasta = f93.Fasta(FASTA)

    peaks = [p for p in load_peaks() if p["chrom"] in fasta.idx]
    genes = {k: v for k, v in load_genes().items() if v[0] in fasta.idx}
    print(f"[read] peaks={len(peaks)} genes={len(genes)}")

    occ = defaultdict(list)
    for p in peaks:
        occ[p["chrom"]].append((p["s"], p["e"]))
    for c in occ:
        occ[c].sort()
    starts = {c: [x[0] for x in v] for c, v in occ.items()}

    def hits(chrom, s, e):
        v = occ.get(chrom)
        if not v:
            return False
        i = max(bisect.bisect_left(starts[chrom], s) - 1, 0)
        while i < len(v) and v[i][0] <= e:
            if v[i][1] >= s:
                return True
            i += 1
        return False

    half = W // 2
    total_len = sum(v[2] - v[1] for v in genes.values())
    pool = defaultdict(list)          # (chrom, dts_bin, gc_bin) -> [(pos, gene, seq)]
    tries = built = 0
    items = list(genes.items())
    while built < a.pool and tries < a.pool * 6:
        tries += 1
        ens, (chrom, gs, ge, gstr) = items[rng.randrange(len(items))]
        span = ge - gs
        pos = gs + rng.randint(half + 1, max(half + 2, span - W))
        if hits(chrom, pos - half, pos - half + W - 1):
            continue
        s = fasta.get(chrom, pos - half, pos - half + W - 1)
        if len(s) < W or "N" in s:
            continue
        tss = gs if gstr == "+" else ge
        pool[(chrom, dts_bin(pos - tss), gc_bin(s))].append((pos, ens, s))
        built += 1
    print(f"[pool] candidates={built} bins={len(pool)}")
    for k in pool:
        rng.shuffle(pool[k])
    cursor = {k: 0 for k in pool}

    rows, pairs, miss = [], 0, 0
    for p in peaks:
        chrom = p["chrom"]
        mid = (p["s"] + p["e"]) // 2
        w = fasta.get(chrom, mid - half, mid - half + W - 1)
        if len(w) < W or "N" in w:
            continue
        key = (chrom, dts_bin(p["dts"]), gc_bin(w))
        lst = pool.get(key)
        if not lst:
            miss += 1
            continue
        cand = None
        while cursor[key] < len(lst):
            pos, ens, s = lst[cursor[key]]
            cursor[key] += 1
            if ens == p["ens"]:
                continue
            cand = (pos, s)
            break
        if cand is None:
            miss += 1
            continue
        pid = f"{p['ens']}:{chrom}:{mid}"
        split = "test" if chrom in TEST_CHR else ("val" if chrom in VAL_CHR else "train")
        rows.append(dict(uid=pid + "_pos", pair_id=pid, label=1, mark="m5C",
                         gene=p["ens"], sym=p["sym"], chrom=chrom, pos=mid,
                         dts=p["dts"], strand=p["strand"], split=split,
                         width=p["e"] - p["s"] + 1, seq=w))
        np_, ns = cand
        rows.append(dict(uid=pid + "_neg", pair_id=pid, label=0, mark="m5C",
                         gene=ens, sym=p["sym"], chrom=chrom, pos=np_,
                         dts=None, strand=p["strand"], split=split,
                         width=p["e"] - p["s"] + 1, seq=ns))
        pairs += 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            r["task"] = "immune_site_m5c_binmatched"
            r["source"] = "GSE247936"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[done] pairs={pairs} rows={len(rows)} unmatched={miss} "
          f"split={dict(Counter(r['split'] for r in rows))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
