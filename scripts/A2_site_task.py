"""Site-level m6A / m5C task from GSE247936.

positive : peak centre, 101 nt window on the peak's strand
negative : same chromosome, offset 200-2000 nt away, window must not overlap
           ANY peak of either mark -- this keeps local GC / regional context
           comparable instead of sampling random genome (which is trivially
           separable by composition).

split    : by chromosome (test = chr20/21/22/X, val = chr17/18/19) so no locus
           is shared between train and test.
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
OUT = ROOT / "data/processed/immune_site_rna_mod.jsonl"
FASTA = ROOT / "data/raw/genome/hg38.fasta"
MARK_FILES = {
    "m6A": ROOT / "data/raw/immune/GSE247936_MeRIP-seq_m6A_peaks.txt.gz",
    "m5C": ROOT / "data/raw/immune/GSE247936_MeRIP-seq_m5C_peaks.txt.gz",
}
W = 101
TEST_CHR = {"chr20", "chr21", "chr22", "chrX"}
VAL_CHR = {"chr17", "chr18", "chr19"}


class Intervals:
    def __init__(self):
        self.by = defaultdict(list)

    def add(self, chrom, s, e):
        self.by[chrom].append((s, e))

    def finish(self):
        for c in self.by:
            self.by[c].sort()
        self.starts = {c: [x[0] for x in v] for c, v in self.by.items()}

    def hits(self, chrom, s, e):
        v = self.by.get(chrom)
        if not v:
            return False
        i = bisect.bisect_left(self.starts[chrom], s) - 1
        j = max(i, 0)
        while j < len(v) and v[j][0] <= e:
            if v[j][1] >= s:
                return True
            j += 1
        return False


def read_peaks(path, mark):
    rows = []
    with gzip.open(path, "rt", errors="replace") as fh:
        head = fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            try:
                chrom = p[1] if mark == "m6A" else p[1]
                s = int(p[2]); e = int(p[3])
                strand = p[5]
            except (IndexError, ValueError):
                continue
            if not chrom.startswith("chr"):
                continue
            rows.append((chrom, s, e, strand if strand in "+-" else "+"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-mark", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = random.Random(a.seed)

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("f93", ROOT / "scripts/93_genome_windows.py")
    f93 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f93)
    fasta = f93.Fasta(FASTA)

    peaks = {m: read_peaks(p, m) for m, p in MARK_FILES.items()}
    for m, v in peaks.items():
        print(f"[read] {m} peaks={len(v)}")

    occ = Intervals()
    for m, v in peaks.items():
        for c, s, e, _st in v:
            occ.add(c, s, e)
    occ.finish()

    rows = []
    for mark, v in peaks.items():
        v = [x for x in v if x[0] in fasta.idx]
        rng.shuffle(v)
        v = v[: a.per_mark]
        kept = 0
        for c, s, e, strand in v:
            mid = (s + e) // 2
            half = W // 2
            w = fasta.get(c, mid - half, mid - half + W - 1)
            if len(w) < W or "N" in w:
                continue
            pid = f"{mark}_{c}_{mid}"
            split = "test" if c in TEST_CHR else ("val" if c in VAL_CHR else "train")
            rows.append(dict(uid=f"{pid}_pos", pair_id=pid, label=1, mark=mark,
                             chrom=c, pos=mid, strand=strand, split=split,
                             width=e - s + 1, seq=w))
            nmid = None
            for _ in range(60):
                off = rng.choice([-1, 1]) * rng.randint(200, 2000)
                cand = mid + off
                ws, we = cand - half, cand - half + W - 1
                if ws < 1:
                    continue
                if occ.hits(c, ws, we):
                    continue
                nw = fasta.get(c, ws, we)
                if len(nw) < W or "N" in nw:
                    continue
                nmid = cand
                break
            if nmid is None:
                rows.pop()
                continue
            nseq = fasta.get(c, nmid - half, nmid - half + W - 1)
            rows.append(dict(uid=f"{pid}_neg", pair_id=pid, label=0, mark=mark,
                             chrom=c, pos=nmid, strand=strand, split=split,
                             width=e - s + 1, seq=nseq))
            kept += 1
        print(f"[make] {mark} pairs={kept}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            r["task"] = "immune_site_rna_mod"
            r["source"] = "GSE247936"
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n1 = sum(1 for r in rows if r["label"] == 1)
    from collections import Counter
    print(f"[done] rows={len(rows)} pos={n1} neg={len(rows) - n1} "
          f"split={dict(Counter(r['split'] for r in rows))} "
          f"mark={dict(Counter(r['mark'] for r in rows))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
