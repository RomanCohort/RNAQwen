"""Local genomic window extraction (needs hg38 fasta, no network).

Builds a .fai-style index once, then slices 101 nt windows around arbitrary
(chrom, pos) coordinates.  Same task shapes as 92_immune_sites.py but ~10^5
windows/second instead of one HTTP request per window.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
from pathlib import Path

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


class Fasta:
    def __init__(self, path: Path):
        self.path = path
        self.idx = {}   # name -> (seq_offset, length, line_bases, line_width)
        with path.open("rb") as fh:
            name = None; seq_len = 0; line_bases = 0; line_width = 0; offset = 0
            while True:
                pos = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if name:
                        self.idx[name] = (offset, seq_len, line_bases, line_width)
                    name = line[1:].split()[0].decode()
                    seq_len = 0; line_bases = 0; line_width = 0; offset = fh.tell()
                else:
                    if line_bases == 0:
                        line_bases = len(line.strip()); line_width = len(line)
                    seq_len += len(line.strip())
            if name:
                self.idx[name] = (offset, seq_len, line_bases, line_width)
        self.fh = open(path, "rb")

    def get(self, chrom: str, start: int, end: int) -> str:
        """1-based inclusive start/end."""
        key = chrom if chrom in self.idx else ("chr" + chrom if "chr" + chrom in self.idx else None)
        if key is None:
            return ""
        off, length, lb, lw = self.idx[key]
        start = max(1, start); end = min(length, end)
        if end < start:
            return ""
        s0 = off + (start - 1) // lb * lw + (start - 1) % lb
        n = end - start + 1
        self.fh.seek(s0)
        raw = self.fh.read(n + (n // lb + 1) * 2).decode(errors="replace")
        return "".join(raw.split())[:n].upper()


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default="data/raw/genome/hg38.fasta")
    ap.add_argument("--peaks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--half", type=int, default=50)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--task", default="mod_site")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--score-col", type=int, default=0)
    a = ap.parse_args()

    fa = Fasta(Path(a.fasta))
    print("chroms:", len(fa.idx), "| first:", list(fa.idx)[:3])

    peaks = []
    with open(a.peaks, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 4 or f[0].lower() in ("chr", "seqnames"):
                continue
            try:
                chrom = f[0].replace("chr", "")
                s, e = int(float(f[1])), int(float(f[2]))
                strand = f[3] if f[3] in "+-" else "+"
            except Exception:
                continue
            peaks.append((chrom, s, e, strand))
    print("peaks:", len(peaks))

    known = {}
    for c, s, e, _ in peaks:
        known.setdefault(c, []).append((s + e) // 2)
    for c in known:
        known[c].sort()

    def near(c, pos, d=200):
        arr = known.get(c, [])
        if not arr:
            return False
        i = bisect.bisect_left(arr, pos - d)
        return i < len(arr) and arr[i] <= pos + d

    rng = random.Random(13)
    out = open(a.out, "w", encoding="utf-8")
    pos_n = neg_n = 0
    for chrom, s, e, strand in peaks:
        mid = (s + e) // 2
        w = fa.get(chrom, mid - a.half, mid + a.half)
        if len(w) != 2 * a.half + 1:
            continue
        if strand == "-":
            w = revcomp(w)
        out.write(json.dumps({"uid": f"{chrom}:{mid}:{strand}", "sequence": w,
                              "task": a.task, "label": 1, "source": Path(a.peaks).name,
                              "family": chrom}, ensure_ascii=False) + "\n")
        pos_n += 1
        for _ in range(a.neg_per_pos):
            for _try in range(8):
                p = mid + rng.randint(-50000, 50000)
                if p < a.half + 1 or near(chrom, p):
                    continue
                nw = fa.get(chrom, p - a.half, p + a.half)
                if len(nw) != 2 * a.half + 1:
                    continue
                if strand == "-":
                    nw = revcomp(nw)
                out.write(json.dumps({"uid": f"{chrom}:{p}:{strand}", "sequence": nw,
                                      "task": a.task, "label": 0, "source": Path(a.peaks).name,
                                      "family": chrom}, ensure_ascii=False) + "\n")
                neg_n += 1
                break
    out.close()
    print(f"wrote pos={pos_n} neg={neg_n} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())