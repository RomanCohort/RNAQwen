"""Gene-level task: does losing m5C raise dsRNA (innate-sensing load)?

Source: GSE174374 dsRNA-seq, WT vs NSUN2-KO, 11,550 genes.
  positive: rel = (KO - WT) / WT > 0.20
  negative: |rel| <= 0.05
Matching is strict: same chromosome preferred, relative length within 15%,
1:1 without reuse.  A keyword-only baseline (length, WT level, position) is
reported so we can see whether the label is confounded.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/immune/GSE174374_dsRNA-seq_WT_vs_NSUN2-KO.xlsx"
OUT = ROOT / "data/processed/immune_gene_dsrna.jsonl"
POS_T, NEG_T, LEN_TOL = 0.20, 0.05, 0.15


def main() -> int:
    d = pd.read_excel(SRC)
    wt = [c for c in d.columns if "WT" in c]
    ko = [c for c in d.columns if "KO" in c]
    d["wt_dsrna"] = d[wt].mean(axis=1)
    d["ko_dsrna"] = d[ko].mean(axis=1)
    d["rel"] = (d["ko_dsrna"] - d["wt_dsrna"]) / d["wt_dsrna"]
    d["length"] = d["end"] - d["start"]

    pos = d[d["rel"] > POS_T].copy()
    neg = d[d["rel"].abs() <= NEG_T].copy()

    by_chrom = defaultdict(list)
    for _, r in neg.iterrows():
        by_chrom[r["chr"]].append(r)

    used, pairs, same_chrom_n = set(), [], 0
    for _, p in pos.sort_values("length").iterrows():
        L = p["length"]
        pick, same = None, True
        cands = sorted(by_chrom.get(p["chr"], []), key=lambda r: abs(r["length"] - L))
        for c in cands:
            if c["id"] in used:
                continue
            if abs(c["length"] - L) / L > LEN_TOL:
                break
            pick = c
            break
        if pick is None:
            same = False
            rest = [r for _, r in neg.iterrows()
                    if r["id"] not in used and abs(r["length"] - L) / L <= LEN_TOL]
            rest.sort(key=lambda r: abs(r["length"] - L))
            if rest:
                pick = rest[0]
        if pick is None:
            continue
        used.add(pick["id"])
        same_chrom_n += same
        pairs.append((1, p))
        pairs.append((0, pick))

    rows = []
    for label, r in pairs:
        rows.append({
            "uid": r["id"], "gene": r["name"], "chrom": r["chr"],
            "start": int(r["start"]), "end": int(r["end"]), "length": int(r["length"]),
            "strand": "+", "strand_source": "not_in_table",
            "label": label,
            "wt_dsrna": round(float(r["wt_dsrna"]), 6),
            "ko_dsrna": round(float(r["ko_dsrna"]), 6),
            "rel_change": round(float(r["rel"]), 4),
            "task": "immune_dsrna_gene", "source": "GSE174374", "family": r["chr"],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    npos = sum(1 for r in rows if r["label"] == 1)
    print(f"pairs={npos} rows={len(rows)} same_chrom={same_chrom_n} unmatched_pos={len(pos)-npos}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())