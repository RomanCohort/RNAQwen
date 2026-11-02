"""Composition-only baseline for the structure targets (no torch).

If 4-mer/GC/length already explain most of mfe_per_nt, then any "the code
channel predicts structure" claim has to beat THIS, not beat zero.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rfam_struct.jsonl"
TARGETS = ["mfe_per_nt", "paired_frac", "longest_stem", "acc_mean", "mfe"]


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    sx = x[order]; i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def ridge(Xtr, ytr, Xte, lam=10.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], 1)
    B = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], 1)
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    return B @ w


def main() -> int:
    recs = []
    with DATA.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("struct") and r.get("sequence"):
                recs.append(r)
    print(f"[data] {len(recs):,}")

    fams = defaultdict(int)
    for r in recs:
        fams[r.get("family", "")] += 1
    order = sorted(fams, key=lambda f: hashlib.md5(f.encode()).hexdigest())
    tot = sum(fams.values()); vt = tt = 0; mp = {}
    for f in order:
        if vt < tot * 0.05:
            mp[f] = "val"; vt += fams[f]
        elif tt < tot * 0.10:
            mp[f] = "test"; tt += fams[f]
        else:
            mp[f] = "train"
    tr = np.array([mp[r["family"]] == "train" for r in recs])
    te = np.array([mp[r["family"]] == "test" for r in recs])

    alpha = "ACGU"
    km = ["".join(p) for p in itertools.product(alpha, repeat=4)]
    ki = {k: i for i, k in enumerate(km)}
    X4 = np.zeros((len(recs), 256), np.float32)
    gc = np.zeros((len(recs), 1), np.float32)
    ln = np.zeros((len(recs), 1), np.float32)
    for i, r in enumerate(recs):
        s = r["sequence"].upper().replace("T", "U")
        ln[i, 0] = len(s)
        gc[i, 0] = (s.count("G") + s.count("C")) / max(len(s), 1)
        n = max(len(s) - 3, 1)
        for j in range(len(s) - 3):
            q = ki.get(s[j:j + 4])
            if q is not None:
                X4[i, q] += 1
        X4[i] /= n
    feats = {"gc": gc, "len": ln, "gc+len": np.concatenate([gc, ln], 1),
             "4mer": X4, "4mer+gc+len": np.concatenate([X4, gc, ln], 1)}

    print(f"\n{'target':14s}" + "".join(f"{k:>14s}" for k in feats))
    res = {}
    for t in TARGETS:
        y = np.array([float(r["struct"][t]) for r in recs])
        row = f"{t:14s}"
        for name, X in feats.items():
            p = ridge(X[tr], y[tr], X[te])
            rho = spearman(y[te], p)
            res[f"{t}|{name}"] = rho
            row += f"{rho:>14.3f}"
        print(row)
    (ROOT / "runs/struct_probe").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs/struct_probe/kmer_baseline.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print("\n[save] runs/struct_probe/kmer_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
