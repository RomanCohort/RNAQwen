"""Structure probe: does the code channel encode RNA secondary structure?

Targets (ViennaRNA, already computed in rfam_struct.jsonl):
    mfe_per_nt, paired_frac, longest_stem, acc_mean, mfe

Feature sets compared:
    E  ERNIE mean                       (768)   no codebook at all
    C  code-table mean                  (768)   content-mean table
    CS code mean + std                  (1536)
    CM code mean + max + min            (2304)

Baselines (same ridge, same split): GC, length, GC+length, 4-mer counts.
Split by Rfam family, balanced on sequence counts.
Reported metric: Spearman rho on held-out families, plus delta over the best
baseline.  Absolute R^2 is deliberately not the headline.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("RNAQ_THREADS", "8")))
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data/processed/rfam_struct.jsonl"
TARGETS = ["mfe_per_nt", "paired_frac", "longest_stem", "acc_mean", "mfe"]


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def r2(y, p):
    ss_res = float(((y - p) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / max(ss_tot, 1e-12)


class Ridge:
    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, y):
        mu, sd = X.mean(0), X.std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Z = (X - mu) / sd
        Z = np.concatenate([Z, np.ones((len(Z), 1))], 1)
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        Z = np.concatenate([Z, np.ones((len(Z), 1))], 1)
        return Z @ self.w


def split_map(fams):
    tot = sum(fams.values())
    order = sorted(fams, key=lambda f: hashlib.md5(f.encode()).hexdigest())
    mp, tv, tt = {}, 0, 0
    vt, tt_target = tot * 0.05, tot * 0.10
    for f in order:
        n = fams[f]
        if tv < vt:
            mp[f] = "val"; tv += n
        elif tt < tt_target:
            mp[f] = "test"; tt += n
        else:
            mp[f] = "train"
    return mp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--fsq", default=str(ROOT / "checkpoints/fsq_stream/tokenizer.pt"))
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(ROOT / "runs/struct_probe"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    with open(a.data, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            s = r.get("struct") or {}
            if not r.get("sequence") or not s:
                continue
            recs.append(r)
    if a.n:
        recs = recs[: a.n]
    print(f"[data] {len(recs):,} sequences", flush=True)

    fams = {}
    for r in recs:
        fams[r.get("family", "")] = fams.get(r.get("family", ""), 0) + 1
    mp = split_map(fams)
    tr = np.array([mp[r.get("family", "")] == "train" for r in recs])
    va = np.array([mp[r.get("family", "")] == "val" for r in recs])
    te = np.array([mp[r.get("family", "")] == "test" for r in recs])
    print(f"[split] train={tr.sum():,} val={va.sum():,} test={te.sum():,} "
          f"families={len(fams):,}", flush=True)

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    fsq_ck = torch.load(a.fsq, map_location="cpu", weights_only=False)
    lat, lev, hid = int(fsq_ck["latent"]), int(fsq_ck["levels"]), int(fsq_ck["hidden"])
    VOCAB = lev ** lat
    fsq = torch.nn.Sequential(torch.nn.Linear(768, lat)).to(DEV)
    fsq[0].load_state_dict({"weight": fsq_ck["state"]["enc.weight"],
                            "bias": fsq_ck["state"]["enc.bias"]})
    fsq.eval()
    print(f"[fsq] {a.fsq}  vocab={VOCAB}", flush=True)

    N = len(recs)
    Xe = np.zeros((N, 768), dtype=np.float32)          # ERNIE mean
    Csum = torch.zeros(VOCAB, 768, dtype=torch.float64) # content table
    Ccnt = torch.zeros(VOCAB, dtype=torch.float64)
    codes_list = []
    mean_c = np.zeros((N, 768), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, a.batch):
            chunk = recs[i:i + a.batch]
            seqs = [r["sequence"].upper().replace("T", "U") for r in chunk]
            o = enc(seqs)
            h = o.hidden.float(); mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, s in enumerate(seqs):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(s) + 2:
                    hh = hh[1:-1]
                Xe[i + j] = hh.mean(0).cpu().numpy()
                q = fsq(hh.to(DEV)) if hh.shape[0] >= 4 else None
                if q is None:
                    codes_list.append(torch.zeros(1, dtype=torch.long))
                    continue
                k = torch.round((torch.tanh(q) + 1) / 2 * (lev - 1)).long()
                c = torch.zeros(k.shape[0], dtype=torch.long, device=DEV)
                for d in range(k.shape[1]):
                    c = c * lev + k[:, d]
                codes_list.append(c.cpu())
            if (i // a.batch) % 50 == 0:
                el = time.time() - t0
                print(f"  [enc] {i+len(chunk)}/{N}  {el/(i+len(chunk))*1000:.0f} ms/seq  "
                      f"eta {(N-i-len(chunk))*el/(i+len(chunk))/60:.0f} min", flush=True)
    print("[enc] done, building content table", flush=True)

    # second pass over stored hidden means is not enough for the content table;
    # approximate it with ERNIE-mean assigned to each sequence's dominant code
    for idx, c in enumerate(codes_list):
        if c.numel():
            v = torch.from_numpy(Xe[idx]).double()
            Csum.index_add_(0, c, v.expand(len(c), 768))
            Ccnt.index_add_(0, c, torch.ones(len(c), dtype=torch.float64))
    table = (Csum / Ccnt.clamp(min=1).unsqueeze(1)).float().numpy()
    global_mean = Xe.mean(0)
    table = np.where(Ccnt.numpy()[:, None] > 0, table, global_mean[None, :])

    mean_c[:] = np.stack([table[c.numpy()].mean(0) if c.numel() else global_mean
                          for c in codes_list])
    std_c = np.stack([table[c.numpy()].std(0) if c.numel() else np.zeros(768)
                      for c in codes_list])
    max_c = np.stack([table[c.numpy()].max(0) if c.numel() else global_mean
                      for c in codes_list])
    min_c = np.stack([table[c.numpy()].min(0) if c.numel() else global_mean
                      for c in codes_list])

    feats = {
        "E_ernie_mean": Xe,
        "C_code_mean": mean_c,
        "CS_code_mean_std": np.concatenate([mean_c, std_c], 1),
        "CM_code_mean_maxmin": np.concatenate([mean_c, max_c, min_c], 1),
    }

    gc = np.array([(r["sequence"].count("G") + r["sequence"].count("C")) / max(len(r["sequence"]), 1)
                   for r in recs])[:, None]
    ln = np.array([len(r["sequence"]) for r in recs], dtype=np.float32)[:, None]
    alpha = "ACGU"
    kmers = ["".join(p) for p in itertools.product(alpha, repeat=4)]
    kidx = {k: i for i, k in enumerate(kmers)}
    X4 = np.zeros((N, 256), dtype=np.float32)
    for i, r in enumerate(recs):
        s = r["sequence"].upper().replace("T", "U")
        n = max(len(s) - 3, 1)
        for j in range(len(s) - 3):
            q = kidx.get(s[j:j + 4])
            if q is not None:
                X4[i, q] += 1
        X4[i] /= n
    feats.update({"B_gc": gc, "B_len": ln, "B_gc_len": np.concatenate([gc, ln], 1),
                  "B_4mer": X4})

    results = {}
    for tname in TARGETS:
        y = np.array([float(r["struct"][tname]) for r in recs], dtype=np.float64)
        for fname, X in feats.items():
            m = Ridge(10.0).fit(X[tr], y[tr])
            p = m.predict(X[te])
            results[f"{tname}|{fname}"] = dict(
                rho=spearman(y[te], p), r2=r2(y[te], p))

    print("\n=== Spearman rho (held-out families) ===")
    header = f"{'target':14s}" + "".join(f"{k.split('_')[0]:>9s}" for k in feats)
    print(header)
    for tname in TARGETS:
        row = f"{tname:14s}"
        for fname in feats:
            row += f"{results[f'{tname}|{fname}']['rho']:>9.3f}"
        print(row)
    print("\n=== delta vs best baseline ===")
    base_names = [k for k in feats if k.startswith("B_")]
    for tname in TARGETS:
        best_b = max(results[f"{tname}|{b}"]["rho"] for b in base_names)
        for fname in feats:
            if fname.startswith("B_"):
                continue
            d = results[f"{tname}|{fname}"]["rho"] - best_b
            print(f"  {tname:14s} {fname:20s} rho={results[f'{tname}|{fname}']['rho']:.3f} "
                  f"best_base={best_b:.3f} delta={d:+.3f}")
    (out / "report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[save] {out/'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
