"""Where does the information die: encoder/FSQ, or the LLM?

B1 showed the code histogram barely beats a length+GC baseline on the 3-class
task.  That number alone cannot say *why*, because the three classes differ
enormously in length (tRNA median 73 nt, rRNA 406 nt), so almost any feature
that leaks length wins for free.

This script puts the same task through a ladder of readouts, from cheapest to
richest, so the gap between any two rungs is attributable:

    len          length alone
    gc           GC% alone
    len+gc
    kmer4        raw-sequence 4-mer histogram        (256 dims)
    kmer6        raw-sequence 6-mer histogram        (4096 dims)
    code_hist    FSQ codes, histogram                (4096 dims)
    code_ngram   FSQ codes, position-sensitive 3-gram

Interpretation:
  * code_hist << kmer6   -> the encoder/FSQ already threw the signal away
  * code_hist ~= kmer6   -> the code is fine; the loss is downstream (LLM)

Two controls that make the comparison honest:
  * length-stratified accuracy: within 100-nt length bins, so length cannot
    explain the result
  * shuffled labels: must land at chance, otherwise there is leakage

Usage:
    python scripts/B2_probe_controls.py --per-class 3000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402

HID = ROOT / "data/processed/_rfa_full_hidden.f16"
IDX = ROOT / "data/processed/_rfa_full_index.pt"
SEQS = ROOT / "data/processed/rfam_seqs.jsonl"
FSQ = ROOT / "runs/B_seq8/artifacts/fsq.pt"
OUT = ROOT / "reports/family_class"

POOL = 4
CLASSES = ["rRNA", "other", "tRNA"]


def coarse(family: str, name: str) -> str:
    n = (name or "").lower()
    if family == "unassigned":
        return "other"
    if n.startswith("trna"):
        return "tRNA"
    if "ssu_rrna" in n or "lsu_rrna" in n or n in ("5s_rrna", "5_8s_rrna", "5.8s_rrna"):
        return "rRNA"
    return "other"


def split_of(family: str, uid: str, cls: str) -> str:
    """Family-level holdout; tRNA falls back to sequence level (2 families only).

    A family-level split on tRNA would put both RF00005 and RF01852 on one side,
    leaving the other side with zero tRNA — the mistake B1 made on its first run.
    """
    if cls == "tRNA":
        h = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16)
    else:
        h = int(hashlib.md5(family.encode()).hexdigest()[:8], 16)
    return "train" if h % 10 < 7 else "test"


def kmer_feat(seq: str, k: int) -> np.ndarray:
    """Fixed-length nucleotide k-mer histogram, A/C/G/U only."""
    idx = {"A": 0, "C": 1, "G": 2, "U": 3}
    v = np.zeros(4 ** k, dtype=np.float32)
    codes = np.array([idx.get(c, -1) for c in seq], dtype=np.int64)
    if len(codes) < k:
        return v
    n = len(codes) - k + 1
    base = 4 ** np.arange(k - 1, -1, -1)
    for i in range(n):
        w = codes[i:i + k]
        if (w < 0).any():
            continue
        v[int(w @ base)] += 1
    v /= max(1, n)
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3000)
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="controls")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- load metadata --------------------------------------------------- #
    meta = {}
    with open(SEQS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = d["uid"]
            meta[u[4:] if u.startswith("seq:") else u] = d

    idx = torch.load(IDX, map_location="cpu", weights_only=False)
    uids, lens, offs = idx["uids"], idx["lens"].tolist(), idx["offsets"].tolist()
    n_pos, dim = int(idx["n_pos"]), int(idx["dim"])

    pools = defaultdict(list)
    for i, u in enumerate(uids):
        k = u[4:] if u.startswith("seq:") else u
        m = meta.get(k)
        if not m:
            continue
        c = coarse(m["family"], m["family_name"])
        pools[c].append((i, k, m["family"], c, m["sequence"]))

    rng = np.random.default_rng(7)
    keep = []
    for c in CLASSES:
        pool = pools[c]
        n = min(args.per_class, len(pool))
        keep.extend(pool[i] for i in rng.choice(len(pool), n, replace=False))
    print(f"[sample] {len(keep):,} = {args.per_class}/class (capped)", flush=True)

    y = np.array([CLASSES.index(k[3]) for k in keep])
    tr = np.array([split_of(k[2], k[1], k[3]) == "train" for k in keep])
    te = ~tr
    print(f"[split] train={int(tr.sum()):,} test={int(te.sum()):,}", flush=True)
    for c in CLASSES:
        print(f"  {c:6s} tr={int(((y == CLASSES.index(c)) & tr).sum()):5d} "
              f"te={int(((y == CLASSES.index(c)) & te).sum()):5d}", flush=True)

    # ---- features -------------------------------------------------------- #
    L = np.array([len(k[4]) for k in keep], dtype=np.float32)
    GC = np.array([100.0 * (k[4].count("G") + k[4].count("C")) / max(1, len(k[4]))
                   for k in keep], dtype=np.float32)

    t0 = time.time()
    k4 = np.stack([kmer_feat(k[4], 4) for k in keep])
    k6 = np.stack([kmer_feat(k[4], 6) for k in keep])
    print(f"[kmer] 4/6 done {time.time()-t0:.0f}s  {k4.shape} {k6.shape}", flush=True)

    # ---- codes ----------------------------------------------------------- #
    ck = torch.load(FSQ, map_location="cpu", weights_only=False)
    dim_enc = int(ck["state"]["compress.weight"].shape[1]) // POOL
    import torch.nn as nn

    class Coder(nn.Module):
        def __init__(self):
            super().__init__()
            self.compress = nn.Linear(POOL * dim_enc, dim_enc, bias=False)
            self.enc = nn.Linear(dim_enc, int(ck["latent"]), bias=True)
            self.compress.weight.data.copy_(ck["state"]["compress.weight"])
            self.enc.weight.data.copy_(ck["state"]["enc.weight"])
            self.enc.bias.data.copy_(ck["state"]["enc.bias"])
            self.levels = int(ck["levels"])
            self.latent = int(ck["latent"])

        @torch.no_grad()
        def codes(self, x):
            n = (x.shape[0] // POOL) * POOL
            if n == 0:
                return torch.zeros(0, dtype=torch.long)
            z = self.enc(self.compress(x[:n].reshape(n // POOL, POOL * dim_enc)))
            step = 2.0 / (self.levels - 1)
            q = torch.clamp(torch.round((z + 1) / step), 0, self.levels - 1).long()
            pw = self.levels ** torch.arange(self.latent, dtype=torch.long,
                                             device=q.device)
            return (q * pw).sum(1)

    coder = Coder().to(args.dev).eval()
    vocab = coder.levels ** coder.latent
    mm = np.memmap(HID, dtype=np.float16, mode="r", shape=(n_pos, dim))
    H = np.zeros((len(keep), vocab), dtype=np.float32)
    NG = np.zeros((len(keep), 1 << 16), dtype=np.float32)
    n_codes = []
    t0 = time.time()
    for j, (i, u, fam, c, seq) in enumerate(keep):
        l = int(lens[i])
        x = torch.from_numpy(
            np.array(mm[int(offs[i]): int(offs[i]) + l], dtype=np.float32)).to(args.dev)
        cc = coder.codes(x).cpu().numpy().astype(np.int64)
        n_codes.append(len(cc))
        if len(cc):
            np.add.at(H[j], cc, 1.0)
            H[j] /= len(cc)
        for t in range(max(0, len(cc) - 2)):
            key = (t, cc[t], cc[t + 1], cc[t + 2])
            b = hashlib.md5(str(key).encode()).digest()
            NG[j][int.from_bytes(b[:4], "little") & 0xFFFF] += 1
        if len(cc) > 2:
            NG[j] /= (len(cc) - 2)
        if (j + 1) % 3000 == 0:
            print(f"  codes {j+1:,}/{len(keep):,} {time.time()-t0:.0f}s", flush=True)
    del mm
    print(f"[codes] done {time.time()-t0:.0f}s  codes/seq="
          f"{np.mean(n_codes):.1f}", flush=True)

    # ---- classifiers ----------------------------------------------------- #
    from sklearn.linear_model import LogisticRegression

    def run(name, X, m=1.0, balanced=True, y_use=None):
        yy = y if y_use is None else y_use
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        clf = LogisticRegression(max_iter=300, C=0.5,
                                 class_weight="balanced" if balanced else None)
        clf.fit(Xs[tr], yy[tr])
        p = clf.predict(Xs[te])
        acc = float((p == yy[te]).mean())
        present = [c for c in range(len(CLASSES)) if (yy[te] == c).any()]
        bal = float(np.mean([(p[yy[te] == c] == c).mean() for c in present]))
        print(f"  {name:12s} acc={acc:.4f} bal={bal:.4f}", flush=True)
        return acc, bal, p, clf, Xs

    res = {"n": len(keep), "per_class": args.per_class,
           "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "codes_per_seq_mean": float(np.mean(n_codes)),
           "readouts": {}}

    print("\n[raw readouts]", flush=True)
    for name, X in (("len", L[:, None]), ("gc", GC[:, None]),
                    ("len+gc", np.stack([L, GC], 1)), ("kmer4", k4),
                    ("kmer6", k6), ("code_hist", H), ("code_ngram", NG)):
        a, b, _, _, _ = run(name, X)
        res["readouts"][name] = {"acc": a, "bal": b}

    # ---- control 1: shuffled labels ------------------------------------- #
    print("\n[control] shuffled labels (must be near chance)", flush=True)
    rng2 = np.random.default_rng(11)
    y_sh = y[tr].copy()
    y_sh = rng2.permutation(y_sh)
    y_full = y.copy()
    y_full[tr] = y_sh
    a, b, _, _, _ = run("code_hist/shuf", H, y_use=y_full)
    res["shuffled"] = {"acc": a, "bal": b}

    # ---- control 2: length-stratified ----------------------------------- #
    print("\n[control] length-stratified (code_hist vs kmer6 vs len)", flush=True)
    band = np.digitize(L, [80, 120, 160, 220, 300, 420, 600])
    strata = {}
    for b_ in np.unique(band):
        sel_te = te & (band == b_)
        sel_tr = tr & (band == b_)
        if sel_te.sum() < 60 or len(np.unique(y[sel_te])) < 3:
            continue
        # refit inside the stratum so length is constant by construction
        row = {"n_test": int(sel_te.sum()), "len_range":
               [float(L[band == b_].min()), float(L[band == b_].max())]}
        for name, X in (("code_hist", H), ("kmer6", k6), ("len+gc",
                                                        np.stack([L, GC], 1))):
            mu, sd = X[sel_tr].mean(0), X[sel_tr].std(0) + 1e-6
            clf = LogisticRegression(max_iter=300, C=0.5, class_weight="balanced")
            clf.fit(((X - mu) / sd)[sel_tr], y[sel_tr])
            p = clf.predict(((X - mu) / sd)[sel_te])
            row[name] = round(float(np.mean([
                (p[y[sel_te] == c] == c).mean()
                for c in range(len(CLASSES)) if (y[sel_te] == c).any()])), 4)
        strata[str(b_)] = row
        print(f"  band {int(b_)} len {row['len_range']} n={row['n_test']}  "
              f"code_hist={row['code_hist']:.3f} kmer6={row['kmer6']:.3f} "
              f"len+gc={row['len+gc']:.3f}", flush=True)
    res["length_strata"] = strata

    lp = OUT / f"B2_probe_{args.tag}.json"
    lp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[out] {lp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
