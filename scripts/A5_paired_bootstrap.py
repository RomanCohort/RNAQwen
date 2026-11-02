"""Paired bootstrap of AUC(real) - AUC(shuffled) for the m5C site tasks.

The two heads see the identical rows, so the two AUCs are strongly correlated
and the standard error of the DIFFERENCE is much smaller than the standard
error of either AUC.  This is the right way to ask "does shuffling hurt?".
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/immune_gene_cls"


def mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_xy(fname):
    d = torch.load(OUT / fname, map_location="cpu", weights_only=False)
    return d["Xmean"].float(), d["meta"]


def fit(X, meta, epochs=60, seed=0):
    pairs = sorted({m["pair_id"] for m in meta if m.get("split") in ("train", "val")})
    tr = [i for i, m in enumerate(meta) if m["split"] == "train"]
    va = [i for i, m in enumerate(meta) if m["split"] == "val"]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-6)
    Xn = (X - mu) / sd
    y = torch.tensor([meta[i]["label"] for i in range(len(meta))])
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(X.shape[1], 2))
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    torch.manual_seed(seed)
    best, best_state = -1, None
    for ep in range(epochs):
        head.train()
        idx = torch.tensor(tr)[torch.randperm(len(tr))]
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            loss = F.cross_entropy(head(Xn[b]), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            p = head(Xn[torch.tensor(va)]).softmax(-1)[:, 1]
        a = mod("a1", "scripts/A1_gene_cls.py").auc(
            [meta[i]["label"] for i in va], p.tolist())
        if a > best:
            best = a
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state); head.eval()
    return head, Xn, y


def scores(head, Xn, idx):
    with torch.no_grad():
        return head(Xn[torch.tensor(idx)]).softmax(-1)[:, 1].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="feats_real_k3_n94286.pt")
    ap.add_argument("--shuf", default="feats_shuffle_k3_n94286.pt")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    a1 = mod("a1", "scripts/A1_gene_cls.py")

    Xr, meta = load_xy(a.real)
    Xs, meta2 = load_xy(a.shuf)
    same = all(p["uid"] == q["uid"] for p, q in zip(meta, meta2))
    print(f"[check] identical row order: {same}  n={len(meta)}")

    hr, Xrn, y = fit(Xr, meta)
    hs, Xsn, _ = fit(Xs, meta2)

    te = [i for i, m in enumerate(meta) if m["split"] == "test"]
    pr, ps = scores(hr, Xrn, te), scores(hs, Xsn, te)
    yy = np.array([meta[i]["label"] for i in te])
    ar, as_ = a1.auc(yy.tolist(), pr.tolist()), a1.auc(yy.tolist(), ps.tolist())
    print(f"[auc] real={ar:.4f} shuffle={as_:.4f} delta={ar - as_:+.4f}  n_test={len(te)}")

    rng = np.random.default_rng(a.seed)
    deltas = []
    for _ in range(a.boot):
        b = rng.integers(0, len(yy), len(yy))
        deltas.append(a1.auc(yy[b].tolist(), pr[b].tolist()) -
                      a1.auc(yy[b].tolist(), ps[b].tolist()))
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"[boot] delta 95% CI = [{lo:+.4f}, {hi:+.4f}]  "
          f"P(delta<=0) = {(d <= 0).mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
