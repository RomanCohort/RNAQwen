"""Matched-capacity k-mer baselines for the A6 task.

The logistic-regression 4-mer baseline is not a fair fight against a
2048 -> 2 linear head fed by ERNIE-RNA + FSQ.  Here the SAME head
architecture (dropout + linear, z-scored, identical splits) is trained on
k-mer count vectors, with k chosen so the input width is comparable.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/immune_gene_cls"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/immune_site_m5c_binmatched.jsonl")
    ap.add_argument("--ks", nargs="+", type=int, default=[4, 5, 6])
    a = ap.parse_args()

    recs = [json.loads(l) for l in (ROOT / a.data).open(encoding="utf-8")]
    print(f"[read] rows={len(recs)}")
    for k in a.ks:
        alpha = "ACGU"
        kmers = ["".join(p) for p in itertools.product(alpha, repeat=k)]
        idx = {s: i for i, s in enumerate(kmers)}
        dim = len(kmers)
        X = torch.zeros(len(recs), dim, dtype=torch.float32)
        for i, r in enumerate(recs):
            s = r["seq"]
            n = max(len(s) - k + 1, 1)
            row = X[i]
            for j in range(len(s) - k + 1):
                q = idx.get(s[j:j + k])
                if q is not None:
                    row[q] += 1.0
            row /= n
        meta = [dict(uid=r["uid"], gene=r.get("gene", ""), chrom=r["chrom"],
                     pair_id=r["pair_id"], label=int(r["label"]), win=0,
                     length=len(r["seq"]), wt_dsrna=0.0, rel_change=0.0,
                     gc=(r["seq"].count("G") + r["seq"].count("C")) / len(r["seq"]),
                     split=r.get("split", ""), mark=r.get("mark", ""), st={})
                for r in recs]
        path = OUT / f"feats_kmer{k}_k3_n{len(recs)}.pt"
        torch.save(dict(X=X, Xmean=X.clone(), meta=meta, tag=f"kmer{k}", k=0,
                        win=len(recs[0]["seq"]), question=""), path)
        print(f"[save] {path.name}  dim={dim}  {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
