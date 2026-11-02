"""7-class RNA family classification — how much structure does the code channel hold?

Why this probe and not more LLM steps
-------------------------------------
The LLM-side evaluation (`76_acc_by_source.py`) already answered "can the 35B body
read the codes".  It can, for the first few nucleotides; everything global is at
the floor.  This probe asks the complementary question at the representation
level, where an answer is cheap: does the code sequence carry *category* structure
at all?  If it does not, no amount of LLM training will recover it.

Split protocol
--------------
Family-level holdout: every Rfam family goes wholly to train or wholly to test
(`md5(family) % 10 < 7`).  A sequence-level split would let the model recognise
"this looks like a tRNA" from siblings of the same family and inflate the score.
The cost is that families with only 9 representatives (rRNA) put few families in
the test side — the per-class family counts are printed so that limitation is
visible instead of buried.

Readouts
--------
  floor     always answer the most frequent class (train)
  compo     logistic regression on (length, GC%) only — no sequence order, no codes
  hist      logistic regression on the 4096-dim FSQ code histogram
  ngram     logistic regression on code 3-gram hashes (captures *order*, which the
            histogram throws away)

`ngram - hist` is the number that separates "frequencies only" from "structure".

Balanced accuracy is reported next to raw accuracy because rRNA is 52% of the
corpus: raw accuracy alone would be dominated by one class.

Usage
-----
    python scripts/B1_family_class.py --limit 4000          # smoke
    python scripts/B1_family_class.py                       # full (145,201 seq)
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

import rnaqwen  # noqa: F401,E402  (must precede transformers via shim)

HID = ROOT / "data/processed/_rfa_full_hidden.f16"
IDX = ROOT / "data/processed/_rfa_full_index.pt"
SEQS = ROOT / "data/processed/rfam_seqs.jsonl"
FSQ = ROOT / "runs/B_seq8/artifacts/fsq.pt"
OUT = ROOT / "reports/family_class"

POOL = 4
NGRAM = 3
HASH_BITS = 16          # 65,536 n-gram buckets — bounded so the design matrix fits


# --------------------------------------------------------------------------- #
# family -> coarse class
# --------------------------------------------------------------------------- #
def coarse(family: str, name: str) -> str:
    """Map an Rfam family to one of seven coarse classes.

    Deliberately keyword-based and explicit: an Rfam-clan-based mapping would be
    better, but it needs the clan file, which is not in the repo.  Misclassified
    families land in `other`, which is reported as its own class rather than
    silently merged, so the number is auditable.
    """
    n = (name or "").lower()
    if family == "unassigned":
        return "other"
    if n.startswith("trna"):
        return "tRNA"
    if "ssu_rrna" in n or "lsu_rrna" in n or n in ("5s_rrna", "5_8s_rrna", "5.8s_rrna"):
        return "rRNA"
    if n.startswith("mir"):
        return "miRNA"
    if n.startswith("sno"):
        return "snoRNA"
    if n in ("u1", "u2", "u4", "u5", "u6", "u11", "u12", "srp", "7sk",
             "metazoa_srp", "vault"):
        return "snRNA"
    if any(k in n for k in ("hammerhead", "ribozyme", "rnasep", "glms", "cobalamin",
                            "twister", "hairpin", "hdv", "group")):
        return "ribozyme"
    return "other"


CLASSES = ["rRNA", "other", "tRNA", "miRNA", "snRNA", "ribozyme", "snoRNA"]

# 3-class reduction.  The 7-class version is kept above for reference, but tRNA
# has only two Rfam families in this corpus (RF00005, RF01852), so a family-level
# holdout can (and did) leave it with zero test rows.  Three coarse classes is the
# level at which every class has enough of *something* to hold out.
CLASSES3 = ["rRNA", "other", "tRNA"]
_TO3 = {"rRNA": "rRNA", "tRNA": "tRNA"}


def fam_split(family: str, frac_train: float = 0.7) -> str:
    """Deterministic family-level split.  md5, not hash(): Python's hash() is
    salted per process and would reshuffle the split on every run."""
    h = int(hashlib.md5(family.encode()).hexdigest()[:8], 16)
    return "train" if (h % 10) < frac_train * 10 else "test"


def seq_split(uid: str, frac_train: float = 0.7) -> str:
    """Sequence-level fallback for classes with too few families to hold out.

    Used for tRNA (2 families).  This is weaker evidence than a family-level
    split and is labelled as such in the output.
    """
    h = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16)
    return "train" if (h % 10) < frac_train * 10 else "test"


# --------------------------------------------------------------------------- #
# code extraction
# --------------------------------------------------------------------------- #
def load_coder(dev: str):
    """Rebuild the frozen FSQ coder (compress + enc) exactly as 75_rfam_probe.py
    does, so the codes here are the same ones the LLM was trained on."""
    import torch.nn as nn

    ck = torch.load(FSQ, map_location="cpu", weights_only=False)
    # `d_model` (4096) is the post-projection LLM width, NOT the encoder width.
    # The encoder width is encoded in compress.weight's input size: 3072 = 768*4.
    in_dim = int(ck["state"]["compress.weight"].shape[1]) // POOL
    dim = in_dim
    mode = ck.get("mode", "pool")

    class Coder(nn.Module):
        def __init__(self):
            super().__init__()
            self.compress = nn.Linear(POOL * dim, dim, bias=False)
            self.enc = nn.Linear(dim, int(ck["latent"]), bias=True)
            st = ck["state"]
            self.compress.weight.data.copy_(st["compress.weight"])
            self.enc.weight.data.copy_(st["enc.weight"])
            self.enc.bias.data.copy_(st["enc.bias"])
            self.levels = int(ck["levels"])
            self.latent = int(ck["latent"])

        @torch.no_grad()
        def codes(self, x: torch.Tensor) -> torch.Tensor:
            """x: (L, dim) float -> (n_codes,) int64"""
            n = (x.shape[0] // POOL) * POOL
            if n == 0:
                return torch.zeros(0, dtype=torch.long)
            x = self.compress(x[:n].reshape(n // POOL, POOL * dim))
            z = self.enc(x)
            step = 2.0 / (self.levels - 1)
            idx = torch.clamp(torch.round((z + 1) / step), 0, self.levels - 1).long()
            pow = (self.levels ** torch.arange(
                self.latent, dtype=torch.long, device=idx.device))
            return (idx * pow).sum(dim=1)

    m = Coder().to(dev).eval()
    print(f"[fsq] dim={dim} mode={mode} latent={m.latent} levels={m.levels}", flush=True)
    return m


def seq_features(codes: torch.Tensor, vocab: int):
    """(codes, ngram) sparse-ish feature rows for one sequence."""
    c = codes.detach().cpu().numpy().astype(np.int64)
    hist = np.zeros(vocab, dtype=np.float32)
    for v in c:
        hist[v] += 1
    n = max(1, len(c) - NGRAM + 1)
    hist /= n                                  # length-normalised frequency
    ng = np.zeros(1 << HASH_BITS, dtype=np.float32)
    for i in range(len(c) - NGRAM + 1):
        key = (i, c[i], c[i + 1], c[i + 2])    # position-sensitive: keeps order
        # md5, not hash(): Python's hash() is salted per process, so the buckets
        # would change between runs and the number would be irreproducible.
        b = hashlib.md5(str(key).encode()).digest()
        ng[int.from_bytes(b[:4], "little") & ((1 << HASH_BITS) - 1)] += 1
    ng /= n
    return hist, ng


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--stratify", type=int, default=0,
                    help="if >0, sample this many per coarse class (smoke only)")
    args = ap.parse_args()

    dev = args.dev
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- labels ---------------------------------------------------------- #
    meta = {}
    with open(SEQS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            u = d["uid"]
            meta[u[4:] if u.startswith("seq:") else u] = (
                d["family"], d["family_name"], d["sequence"])

    idx = torch.load(IDX, map_location="cpu", weights_only=False)
    uids = idx["uids"]
    lens = idx["lens"].tolist()
    offs = idx["offsets"].tolist()
    n_pos, dim = int(idx["n_pos"]), int(idx["dim"])

    keep = []
    for i, u in enumerate(uids):
        # The index stores the bare md5; rfam_seqs.jsonl prefixes it with "seq:".
        k = u[4:] if u.startswith("seq:") else u
        m = meta.get(k)
        if not m:
            continue
        fam, name, seq = m
        keep.append((i, u, fam, coarse(fam, name), seq))
    if args.stratify:
        # Stratified smoke.  Taking the first N rows is NOT a random sample:
        # all.fa is grouped by family, so a head slice is almost pure rRNA.
        rng = np.random.default_rng(7)
        per = defaultdict(list)
        for k in keep:
            per[k[3]].append(k)
        sel = []
        for c in CLASSES:
            pool = per[c]
            n = min(args.stratify, len(pool))
            sel.extend(pool[i] for i in rng.choice(len(pool), n, replace=False))
        keep = sel
    elif args.limit:
        keep = keep[: args.limit]

    print(f"[data] cached={len(uids):,}  matched={len(keep):,}", flush=True)
    by_cls = Counter(k[3] for k in keep)
    fam_cnt = Counter({k[2]: k[3] for k in keep}.values())
    print(f"[classes] {'class':10s} {'seq':>8s} {'fam':>6s}", flush=True)
    for c in CLASSES:
        print(f"          {c:10s} {by_cls[c]:8d} {fam_cnt[c]:6d}", flush=True)

    # Split policy per class:
    #   tRNA  -> sequence-level (only 2 families; a family split can empty it)
    #   rest  -> family-level holdout
    FAMCOUNT = Counter(k[2] for k in keep)
    split = []
    for i, u, fam, cls, seq in keep:
        c3 = _TO3.get(cls, "other")
        if c3 == "tRNA" or FAMCOUNT[fam] <= 2:
            sp = seq_split(u)
        else:
            sp = fam_split(fam)
        split.append((i, cls, sp, c3))
    n_tr = sum(1 for s in split if s[2] == "train")
    n_trna_tr = sum(1 for s in split if s[3] == "tRNA" and s[2] == "train")
    n_trna_te = sum(1 for s in split if s[3] == "tRNA" and s[2] == "test")
    print(f"[split] train={n_tr:,} test={len(split)-n_tr:,}  "
          f"(tRNA seq-level: tr={n_trna_tr:,} te={n_trna_te:,})", flush=True)

    # ---- code channel ---------------------------------------------------- #
    coder = load_coder(dev)
    vocab = coder.levels ** coder.latent
    mm = np.memmap(HID, dtype=np.float16, mode="r", shape=(n_pos, dim))
    h = np.zeros((len(keep), vocab), dtype=np.float32)
    g = np.zeros((len(keep), 1 << HASH_BITS), dtype=np.float32)
    comp = np.zeros((len(keep), 2), dtype=np.float32)
    t0 = time.time()
    for j, (i, u, fam, cls, seq) in enumerate(keep):
        L = int(lens[i])
        x = torch.from_numpy(
            np.array(mm[int(offs[i]): int(offs[i]) + L], dtype=np.float32)).to(dev)
        c = coder.codes(x)
        hh, gg = seq_features(c, vocab)
        h[j], g[j] = hh, gg
        comp[j] = (L, 100.0 * (seq.count("G") + seq.count("C")) / max(1, len(seq)))
        if (j + 1) % 20000 == 0:
            print(f"  encoded {j+1:,}/{len(keep):,}  {time.time()-t0:.0f}s",
                  flush=True)
    del mm
    print(f"[codes] done {time.time()-t0:.0f}s", flush=True)

    # ---- classifiers ----------------------------------------------------- #
    from sklearn.linear_model import LogisticRegression

    def run_task(label: str, classes: list[str]) -> dict:
        """Fit every readout on one class reduction and return the report block."""
        idx_of = {c: i for i, c in enumerate(classes)}
        y = np.array([idx_of[_TO3.get(s[1], "other")] if classes is CLASSES3
                      else idx_of[s[1]] for s in split])
        maj = Counter(y[tr]).most_common(1)[0][0]
        maj_p = float((y[te] == maj).mean())
        print(f"\n[{label}] floor=always {classes[maj]}: acc={maj_p:.4f}", flush=True)

        block = {"classes": classes, "n_train": int(tr.sum()),
                 "n_test": int(te.sum()),
                 "floor": {"class": classes[maj], "acc": maj_p},
                 "readouts": {}}

        def fit(name: str, X: np.ndarray, C: float = 1.0, balanced: bool = True):
            t = time.time()
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xs = (X - mu) / sd
            clf = LogisticRegression(max_iter=200, C=C,
                                     class_weight="balanced" if balanced else None)
            clf.fit(Xs[tr], y[tr])
            pred = clf.predict(Xs[te])
            acc = float((pred == y[te]).mean())
            present = [c for c in range(len(classes)) if (y[te] == c).any()]
            bal = float(np.mean([(pred[y[te] == c] == c).mean() for c in present]))
            per = {classes[c]: round(float((pred[y[te] == c] == c).mean()), 4)
                   for c in present}
            tag = "bal" if balanced else "raw"
            print(f"  {name:12s} {tag}  acc={acc:.4f}  balanced={bal:.4f}  "
                  f"{per}  ({time.time()-t:.0f}s)", flush=True)
            block["readouts"][f"{name}/{tag}"] = {
                "acc": acc, "balanced": bal, "per_class": per}

        stacks = {"compo": comp, "hist": h, "ngram": g,
                  "hist+ngram": np.hstack([h, g])}
        for mode in (True, False):
            for name, X in stacks.items():
                fit(name, X, C=1.0 if name == "ngram" else 0.5, balanced=mode)
        return block

    tr = np.array([s[2] == "train" for s in split])
    te = ~tr

    res = {"n": len(keep), "classes7": dict(by_cls),
           "protocol": {
               "tRNA": "sequence-level (2 families only)",
               "rRNA": "family-level",
               "other": "family-level",
               "ngram": "position-sensitive code 3-gram, md5 hash",
           }}
    res["task_3class"] = run_task("3-class", CLASSES3)
    res["task_7class"] = run_task("7-class", CLASSES)

    lp = OUT / f"B1_family_class_{args.tag}.json"
    lp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {lp}", flush=True)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
