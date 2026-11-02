"""Position-aware readout for the SOS code channel.

Every readout built so far pools the code sequence with mean/std/max/min, and
all of those are permutation-invariant: they cannot tell ABCDE from EDCBA.
Comparing such a readout against a 4-mer bag is comparing bag to bag, which is
exactly why A7 ties (0.5685 vs 0.5663).

This script keeps coarse position by splitting the code sequence into B
equal-relative-width bins and mean-pooling each bin.  B=1 reproduces the old
bag readout, so the two come from the same file:

    X      = flattened bin means   (B * d_model)  <- position-aware
    Xmean  = global mean           (d_model)      <- old bag readout

No 35B body: the code table alone is the object under test.

Requires the code table that matches the codebook.  For a new codebook, build
it first with 51_content_init_codebook.py --fsq ... --out ...
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def window_of(r) -> str:
    """Accept both schemas: RNAQA (`sequence`) and site tasks (`seq`)."""
    s = r.get("sequence") or r.get("seq") or ""
    if r.get("strand") == "-":
        s = revcomp(s)
    return s


def bin_pool(v: torch.Tensor, bins: int) -> torch.Tensor:
    """(n, d) -> (bins, d), equal relative widths, always exactly `bins` rows."""
    n = v.shape[0]
    out = torch.zeros(bins, v.shape[1], dtype=v.dtype, device=v.device)
    if n == 0:
        return out
    for b in range(bins):
        lo = (n * b) // bins
        hi = (n * (b + 1)) // bins
        if hi > lo:
            out[b] = v[lo:hi].mean(0)
        else:
            out[b] = v[min(lo, n - 1)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fsq", default=str(ROOT / "checkpoints/fsq_stream/tokenizer.pt"))
    ap.add_argument("--code-emb", required=True,
                    help="code table matching --fsq (see 51_content_init_codebook.py)")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--tag", default="real", choices=["real", "shuffle"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(int(__import__("os").environ.get("RNAQ_THREADS", "8")))

    recs = []
    with open(a.data, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    if a.limit:
        recs = recs[: a.limit]
    print(f"[data] {len(recs):,} records", flush=True)

    ck = torch.load(a.fsq, map_location="cpu", weights_only=False)
    lat, lev = int(ck["latent"]), int(ck["levels"])
    vocab = lev ** lat
    print(f"[fsq] {Path(a.fsq).name}  latent={lat} levels={lev} vocab={vocab}", flush=True)

    ce = torch.load(a.code_emb, map_location="cpu", weights_only=False)
    table = ce["code_emb"] if isinstance(ce, dict) else ce
    table = table.float().to(DEV)
    print(f"[code-emb] {Path(a.code_emb).name}  {tuple(table.shape)}", flush=True)
    assert table.shape[0] == vocab, "code table does not match codebook vocabulary"

    fsq_w = ck["state"]["enc.weight"].float().to(DEV)
    fsq_b = ck["state"]["enc.bias"].float().to(DEV)

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    feats, means, meta = [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(recs), a.bs):
            chunk = recs[i:i + a.bs]
            seqs = []
            for r in chunk:
                w = window_of(r).upper().replace("T", "U")
                if a.tag == "shuffle":
                    rr = random.Random(zlib.crc32(r["uid"].encode()))
                    w = "".join(rr.sample(list(w), len(w)))
                seqs.append(w)

            o = enc(seqs)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)

            for j, (r, w) in enumerate(zip(chunk, seqs)):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(w) + 2:
                    hh = hh[1:-1]
                if hh.shape[0] < 4:
                    continue
                k = torch.round((torch.tanh(hh @ fsq_w.T + fsq_b) + 1) / 2 * (lev - 1)).long()
                c = torch.zeros(k.shape[0], dtype=torch.long, device=DEV)
                for d in range(k.shape[1]):
                    c = c * lev + k[:, d]
                v = table[c]                       # (ncode, d_model)
                feats.append(bin_pool(v, a.bins).reshape(-1).float().cpu())
                means.append(v.mean(0).float().cpu())
                meta.append(dict(
                    uid=r["uid"], gene=r.get("gene", r.get("mark", "")),
                    chrom=r.get("chrom", ""), pair_id=r.get("pair_id", -1),
                    label=int(r.get("label", 0)), win=0,
                    length=int(r.get("length") or len(w)),
                    wt_dsrna=float(r.get("wt_dsrna", 0.0)),
                    rel_change=float(r.get("rel_change", 0.0)),
                    gc=(w.count("G") + w.count("C")) / max(len(w), 1),
                    split=r.get("split", ""), mark=r.get("mark", ""), st={},
                    ncode=int(v.shape[0])))

            if (i // a.bs) % 100 == 0:
                el = time.time() - t0
                done = i + len(chunk)
                print(f"  [enc] {done}/{len(recs)}  {el/done*1000:.0f} ms/rec  "
                      f"eta {(len(recs)-done)*el/done/60:.0f} min", flush=True)

    X = torch.stack(feats)
    Xm = torch.stack(means)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(X=X, Xmean=Xm, meta=meta, tag=a.tag, k=a.bins, win=0,
                    question="", ckpt=str(a.fsq), code_emb=str(a.code_emb)), out)
    print(f"[save] {out}  X={tuple(X.shape)} ({out.stat().st_size/1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
