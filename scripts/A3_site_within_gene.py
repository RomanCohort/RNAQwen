"""m5C site task with WITHIN-GENE negatives.

v1 (A2_site_task.py) drew negatives 200-2000 nt away on the same chromosome,
which still lets the model key on "which region of the genome is this".
Here every negative comes from the SAME gene as its positive, at a similar
relative position within that gene, and outside every m5C peak.  A model that
beats the positional baseline on this task is reading the local sequence, not
the neighbourhood.

m6A is not included: that peak file carries only a numeric geneID, no gene
coordinates, and the v1 analysis already showed its signal is composition.
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
OUT = ROOT / "data/processed/immune_site_m5c_within_gene.jsonl"
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
                gstrand = "-" if str(p[23]).strip() in ("2", "-1") else "+"
                sym = p[28]
                ens = p[27]
            except (ValueError, IndexError):
                continue
            if gchr not in MAIN or chrom not in MAIN or chrom != gchr:
                continue
            if ge - gs < 400:
                continue
            out.append(dict(chrom=chrom, s=s, e=e, strand=strand,
                            gs=gs, ge=ge, gstrand=gstrand, sym=sym, ens=ens))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
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
    print(f"[read] m5C peaks with gene coords={len(peaks)}")

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
    rows = []
    kept_pairs = 0
    npos_used = 0
    for (ens, chrom, gs, ge), plist in by_gene.items():
        rng.shuffle(plist)
        plist = plist[: a.max_per_gene]
        span = max(ge - gs, 1)
        for p in plist:
            mid = (p["s"] + p["e"]) // 2
            w = fasta.get(chrom, mid - half, mid - half + W - 1)
            if len(w) < W or "N" in w:
                continue
            relpos_p = (mid - gs) / span
            cand = None
            for tol in (0.25, 0.5, 1.0):
                for _ in range(400):
                    off = rng.randint(half + 1, max(half + 2, span - W))
                    c = gs + off
                    rel = (c - gs) / span
                    if abs(rel - relpos_p) > tol:
                        continue
                    ws, we = c - half, c - half + W - 1
                    if ws < 1 or hits(chrom, ws, we):
                        continue
                    nw = fasta.get(chrom, ws, we)
                    if len(nw) < W or "N" in nw:
                        continue
                    cand = (c, nw)
                    break
                if cand:
                    break
            if not cand:
                continue
            npos_used += 1
            pid = f"{ens}:{chrom}:{mid}"
            split = "test" if chrom in TEST_CHR else ("val" if chrom in VAL_CHR else "train")
            rows.append(dict(uid=pid + "_pos", pair_id=pid, label=1, mark="m5C",
                             gene=ens, sym=p["sym"], chrom=chrom, pos=mid,
                             strand=p["strand"], split=split, relpos=round(relpos_p, 4),
                             width=p["e"] - p["s"] + 1, seq=w))
            nc, nw = cand
            rows.append(dict(uid=pid + "_neg", pair_id=pid, label=0, mark="m5C",
                             gene=ens, sym=p["sym"], chrom=chrom, pos=nc,
                             strand=p["strand"], split=split,
                             relpos=round((nc - gs) / span, 4),
                             width=p["e"] - p["s"] + 1, seq=nw))
            kept_pairs += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            r["task"] = "immune_site_m5c_within_gene"
            r["source"] = "GSE247936"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[done] pairs={kept_pairs} rows={len(rows)} genes={npos_used} "
          f"split={dict(Counter(r['split'] for r in rows))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
