"""Build site-level m5C windows from GSE174374 (human A549, +/- infection, NSUN2-KO).

No genome download: Ensembl REST serves the 101 nt window around each site.

Positives  = reported m5C sites (sheet + row provenance kept)
Negatives  = random cytosines on the same chromosome, >=200 bp from any known
             site, so the model cannot win by recognising the locus alone.

Output: data/processed/rnaqa_immune.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES_CSV = ROOT / "data/raw/immune/GSE174374_host_m5c_sites.csv"
OUT = ROOT / "data/processed/rnaqa_immune.jsonl"
REST = "https://rest.ensembl.org"
HALF = 50
BATCH = 20
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _one(chrom: str, start: int) -> str:
    """Single-region GET through curl.exe (the POST batch endpoint drops us)."""
    import subprocess
    url = (f"{REST}/sequence/region/human/{chrom}:{start}..{start + 2 * HALF}"
           "?content-type=text/plain")
    for i in range(4):
        p = subprocess.run(["curl.exe", "-s", "-m", "90", "-w", "\n%{http_code}", url],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        body, _, code = p.stdout.rpartition("\n")
        if p.returncode == 0 and code.strip() == "200" and len(body.strip()) == 2 * HALF + 1:
            return body.strip()
        time.sleep(1.5 * (i + 1))
    return ""


def fetch(windows: list[tuple[str, int]], workers: int = 6) -> list[str]:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        seqs = list(pool.map(lambda w: _one(w[0], w[1]), windows))
    return seqs

def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    df_all = pd.read_csv(SITES_CSV)
    sites = []
    for sheet in df_all["sheet"].unique():
        df = df_all[df_all["sheet"] == sheet]
        for _, r in df.iterrows():
            chrom = str(r["chr"]).replace("chr", "").replace("M", "MT") if str(r["chr"]).strip() in ("chrM", "M") else str(r["chr"]).replace("chr", "")
            if not chrom.replace("X", "").replace("Y", "").replace("M", "").isdigit() and chrom not in ("X", "Y", "M"):
                continue
            pos = int(r["pos"]); strand = str(r["strand"]).strip()
            gene = None
            for col in ("name", "Annotation", "refseq"):
                if col in df.columns and pd.notna(r.get(col)):
                    gene = str(r[col]); break
            ratios = [float(r[c]) for c in df.columns if "WT rep" in str(c) and pd.notna(r[c])]
            sites.append(dict(chrom=chrom, pos=pos, strand=strand, sheet=sheet,
                              gene=gene, wt=sum(ratios) / len(ratios) if ratios else None))
    print(f"site rows: {len(sites)}")

    known = {}
    for s in sites:
        known.setdefault(s["chrom"], []).append(s["pos"])

    rng = random.Random(13)
    negs = []
    for s in sites:
        tries = 0
        made = 0
        while made < a.neg_per_pos and tries < 6:
            tries += 1
            off = rng.randint(-50000, 50000)
            pos = s["pos"] + off
            if pos < HALF + 1:
                continue
            near = [p for p in known.get(s["chrom"], []) if abs(p - pos) < 200]
            if near:
                continue
            negs.append(dict(chrom=s["chrom"], pos=pos, strand=s["strand"], sheet=s["sheet"],
                             gene=s["gene"], wt=None))
            made += 1
    print(f"candidate negatives: {len(negs)}")

    wins = [(s["chrom"], s["pos"] - HALF) for s in sites] + [(n["chrom"], n["pos"] - HALF) for n in negs]
    seqs = fetch(wins)
    if len(seqs) != len(wins):
        print(f"** fetch returned {len(seqs)} of {len(wins)}")
        return 3

    rows, kept_neg = [], 0
    for i, s in enumerate(sites):
        w = seqs[i]
        if len(w) != 2 * HALF + 1:
            continue
        if s["strand"] == "-":
            w = revcomp(w)
        rows.append(dict(sequence=w, task="m5c_site", label=1,
                         source="GSE174374", family=s["gene"] or f"chr{s['chrom']}", 
                         meta={"chrom": s["chrom"], "pos": s["pos"], "strand": s["strand"],
                               "sheet": s["sheet"], "wt_ratio": s["wt"]}))
    base = len(sites)
    for j, n in enumerate(negs):
        w = seqs[base + j]
        if len(w) != 2 * HALF + 1:
            continue
        if n["strand"] == "-":
            w = revcomp(w)
        if w[HALF].upper() != "C":      # negatives must be cytosines to be comparable
            continue
        kept_neg += 1
        rows.append(dict(sequence=w, task="m5c_site", label=0,
                         source="GSE174374", family=n["gene"] or f"chr{n['chrom']}",
                         meta={"chrom": n["chrom"], "pos": n["pos"], "strand": n["strand"],
                               "sheet": n["sheet"], "wt_ratio": None}))

    for r in rows:
        r["uid"] = hashlib.sha1(r["sequence"].encode()).hexdigest()[:16]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(1 for r in rows if r["label"] == 1); neg = len(rows) - pos
    fams = len({r["family"] for r in rows})
    print(f"wrote {len(rows)} rows (pos={pos} neg={neg}) families={fams} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())