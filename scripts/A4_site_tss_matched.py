"""m5C sites, negatives matched on BOTH gene and distance-to-TSS.

A3 matched the relative position inside the gene.  distance-to-TSS is the
single strongest positional driver of m5C, and it is not the same thing: a
gene of 50 kb can have two positions at relpos 0.1 that sit 5 kb apart.
Here the negative must sit within +-300 nt of the positive's distance-to-TSS
and on the same side, in the same gene, outside every m5C peak.

If a model still separates these, it is reading the local sequence.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/immune_site_m5c_tss_matched.jsonl"
FASTA = ROOT / "data/raw/genome/hg38.fasta"
SRC = ROOT / "data/raw/immune/GSE247936_MeRIP-seq_m5C_peaks.txt.gz"
W = 101
TEST_CHR = {"chr20", "chr21", "chr22", "chrX"}
VAL_CHR = {"chr17", "chr18", "chr19"}
MAIN = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def load():
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
                gchr = p[19]
                if not gchr.startswith("chr"):
                    gchr = "chr" + gchr
                gs, ge = int(p[20]), int(p[21])
                dts = int(float(p[26]))
                sym, ens = p[28], p[27]
            except (ValueError, IndexError):
                continue
            if chrom not in MAIN or gchr != chrom:
                continue
            if ge - gs < 400:
                continue
            out.append(dict(chrom=chrom, s=s, e=e, strand=strand, gs=gs, ge=ge,
                            dts=dts, sym=sym, ens=ens))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=int, default=150)
    ap.add_argument("--min-dts", type=int, default=250)
    ap.add_argument("--max-per-gene", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import importlib.util
    spec = importlib.util.spec_from_file_location("f93", ROOT / "scripts/93_genome_windows.py")
    f93 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f93)
    fasta = f93.Fasta(FASTA)

    peaks = [p for p in load() if p["chrom"] in fasta.idx]
    print(f"[read] m5C peaks usable={len(peaks)}")

    by_gene = defaultdict(list)
    for p in peaks:
        by_gene[(p["ens"], p["chrom"], p["gs"], p["ge"])].append(p)
    print(f"[read] genes={len(by_gene)}")

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
    rows, pairs = [], 0
    for (ens, chrom, gs, ge), plist in by_gene.items():
        rng.shuffle(plist)
        plist = plist[: a.max_per_gene]
        for p in plist:
            if abs(p["dts"]) < a.min_dts:
                continue
            mid = (p["s"] + p["e"]) // 2
            w = fasta.get(chrom, mid - half, mid - half + W - 1)
            if len(w) < W or "N" in w:
                continue
            d0 = p["dts"]
            cand = None
            for tol in (a.tol, a.tol * 2, a.tol * 4):
                for _ in range(300):
                    delta = rng.randint(-tol, tol)
                    d1 = d0 + delta
                    if d1 == 0 or (d1 * d0 < 0):
                        continue
                    if abs(abs(d1) - abs(d0)) > tol:
                        continue
                    c = mid + delta
                    if not (gs + half < c < ge - half):
                        continue
                    ws, we = c - half, c - half + W - 1
                    if ws < 1 or hits(chrom, ws, we):
                        continue
                    nw = fasta.get(chrom, ws, we)
                    if len(nw) < W or "N" in nw:
                        continue
                    cand = (c, nw, d0 + delta)
                    break
                if cand:
                    break
            if not cand:
                continue
            pid = f"{ens}:{chrom}:{mid}"
            split = "test" if chrom in TEST_CHR else ("val" if chrom in VAL_CHR else "train")
            rows.append(dict(uid=pid + "_pos", pair_id=pid, label=1, mark="m5C",
                             gene=ens, sym=p["sym"], chrom=chrom, pos=mid, dts=d0,
                             strand=p["strand"], split=split, width=p["e"] - p["s"] + 1,
                             seq=w))
            nc, nw, nd = cand
            rows.append(dict(uid=pid + "_neg", pair_id=pid, label=0, mark="m5C",
                             gene=ens, sym=p["sym"], chrom=chrom, pos=nc, dts=nd,
                             strand=p["strand"], split=split, width=p["e"] - p["s"] + 1,
                             seq=nw))
            pairs += 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            r["task"] = "immune_site_m5c_tss_matched"
            r["source"] = "GSE247936"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[done] pairs={pairs} rows={len(rows)} split={dict(Counter(r['split'] for r in rows))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
