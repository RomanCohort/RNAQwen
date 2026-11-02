"""Streaming stage-1: train the FSQ codebook on a large RNAcentral sample.

48_stage1_full_tokenizer.py caches every position hidden state to disk.  That
is fine for 4,481 sequences (5.9 GB) but 2M sequences would be ~1.5 TB.  Here
the encoding and the training are interleaved: encode a chunk, take a few
optimiser steps on it, throw it away.

What is kept:
  - a small held-out set (never trained on) for mse / usage / transfer
  - an incremental code -> 6-mer count table for the transfer metric

The transfer metric is the only non-circular check (46_): do unseen sequences
map the same code onto the same local 6-mer?  Baseline to beat is lift 2.63.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

import os
import torch
import torch.nn as nn

_t = int(os.environ.get("RNAQ_THREADS", "8"))
torch.set_num_threads(_t)
torch.set_num_interop_threads(1)
print(f"[threads] torch={_t} interop=1", flush=True)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RNACENTRAL = ROOT / "data/raw/rnacentral/rnacentral_active.fasta.gz"
TOTAL_SEQS = 48_053_536
K = 6


def fsq(z, levels):
    t = (torch.tanh(z) + 1) / 2
    s = t * (levels - 1)
    k = torch.round(s)
    return (k / (levels - 1) * 2 - 1) + (s - s.detach()), k.long()


class FSQTokenizer(nn.Module):
    def __init__(self, dim, latent, levels, hidden=256):
        super().__init__()
        self.dim, self.latent, self.levels = dim, latent, levels
        self.enc = nn.Linear(dim, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        q, k = fsq(self.enc(x), self.levels)
        c = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            c = c * self.levels + k[:, d]
        return self.dec(q), c


def iter_fasta(path):
    name, buf = None, []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line[0] == ">":
                if name is not None and buf:
                    yield name, "".join(buf)
                name = line[1:].strip()
                buf = []
            else:
                buf.append(line.strip())
    if name is not None and buf:
        yield name, "".join(buf)


def mer_ids(win, k=K):
    b = win.encode()
    out = torch.full((len(b),), -1, dtype=torch.long)  # aligned to positions, -1 pad
    if len(b) < k:
        return out
    a = 0
    for i, ch in enumerate(b):
        a = ((a * 4) + {65: 0, 67: 1, 71: 2, 84: 3, 85: 0}.get(ch, 0)) & (4 ** k - 1)
        if i >= k - 1:
            out[i - k + 1] = a
    return out


def encode_windows(enc, wins, batch=64):
    """Returns (list of (L,768) tensors, list of source indices that survived)."""
    xs, keep = [], []
    for i in range(0, len(wins), batch):
        chunk = wins[i:i + batch]
        o = enc(chunk)
        h = o.hidden.float()
        mk = o.mask.squeeze(-1) > 0
        if mk.shape[1] != h.shape[1]:
            mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for j, s in enumerate(chunk):
            hh = h[j][mk[j]]
            if hh.shape[0] == len(s) + 2:
                hh = hh[1:-1]
            if hh.shape[0] == len(s):
                xs.append(hh.cpu()); keep.append(i + j)
        del o, h, mk
    return xs, keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", default=str(RNACENTRAL))
    ap.add_argument("--jsonl", default="", help="converted jsonl; skips FASTA parsing")
    ap.add_argument("--n-seq", type=int, default=2_000_000)
    ap.add_argument("--total", type=int, default=TOTAL_SEQS)
    ap.add_argument("--min-nt", type=int, default=20)
    ap.add_argument("--max-nt", type=int, default=512)
    ap.add_argument("--window", type=int, default=101)
    ap.add_argument("--latent", type=int, default=12)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--chunk", type=int, default=2048, help="sequences per chunk")
    ap.add_argument("--steps-per-chunk", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "checkpoints/fsq_stream"))
    ap.add_argument("--limit-chunks", type=int, default=0, help="pilot mode")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    VOCAB = a.levels ** a.latent
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stride = 1 if a.jsonl else max(1, a.total // a.n_seq)
    print(f"[cfg] vocab={VOCAB} stride={stride} target={a.n_seq:,} "
          f"window={a.window} chunk={a.chunk}", flush=True)

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    tok = FSQTokenizer(768, a.latent, a.levels, a.hidden).to(DEV)
    opt = torch.optim.AdamW(tok.parameters(), lr=a.lr)

    # ---- held-out first: the LAST `val` sampled records, never trained on
    val_start = a.n_seq - a.val
    val_wins = []
    train_iter = []
    t0 = time.time()
    seen = kept = 0
    if a.jsonl:
        src = []
        with open(a.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                s_ = r.get("sequence") or ""
                if a.min_nt <= len(s_) <= a.max_nt:
                    src.append((r.get("uid", ""), s_))
        print(f"[data] jsonl records={len(src):,}", flush=True)
    else:
        src = None
    for name, seq in (src if src is not None else iter_fasta(a.fasta)):
        if seen % stride:
            seen += 1
            continue
        seen += 1
        if not (a.min_nt <= len(seq) <= a.max_nt):
            continue
        w = a.window
        if len(seq) <= w:
            win = seq
        else:
            r = random.Random(kept)
            s = r.randrange(0, len(seq) - w + 1)
            win = seq[s:s + w]
        win = win.upper().replace("T", "U")
        if any(c not in "ACGU" for c in win):
            continue
        if kept >= val_start:
            val_wins.append(win)
        else:
            train_iter.append(win)
        kept += 1
        if kept >= a.n_seq:
            break
    print(f"[data] kept={kept:,} train={len(train_iter):,} val={len(val_wins):,} "
          f"({time.time()-t0:.0f}s)", flush=True)

    print("[val] encoding held-out", flush=True)
    val_xs, val_keep = encode_windows(enc, val_wins)
    val_wins = [val_wins[i] for i in val_keep]
    val_x = torch.cat(val_xs) if val_xs else torch.zeros(0, 768)
    print(f"[val] positions={val_x.shape[0]:,}", flush=True)

    # Preallocate the (code, 6-mer) table once and scatter-add in place.
    # The previous `cnt += bincount(..., minlength=16.7M)` allocated a 134 MB
    # int64 tensor plus a 67 MB int32 copy on EVERY chunk; over ~500 chunks the
    # CPU allocator degraded and per-chunk time drifted from 15 s to 30 s.
    cnt_flat = torch.zeros(VOCAB * 4 ** K, dtype=torch.int32)
    step = 0
    start_chunk = 0
    resume_path = out / "state_latest.pt"
    if resume_path.exists() and not a.no_resume:
        st = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(st.get("vocab", 0)) == VOCAB:
            tok.load_state_dict(st["state"])
            step = int(st.get("step", 0))
            start_chunk = int(st.get("chunk", -1)) + 1
            print(f"[resume] from chunk {start_chunk} (step {step})", flush=True)
    nchunk = math.ceil(len(train_iter) / a.chunk)
    if a.limit_chunks:
        nchunk = min(nchunk, a.limit_chunks)
    for ci in range(start_chunk, nchunk):
        chunk = train_iter[ci * a.chunk:(ci + 1) * a.chunk]
        xs, keep = encode_windows(enc, chunk)
        if not xs:
            continue
        pos = torch.cat(xs)
        mers = torch.cat([mer_ids(chunk[i]) for i in keep])
        assert pos.shape[0] == mers.numel()
        tok.train()
        for _ in range(a.steps_per_chunk):
            p = torch.randint(0, pos.shape[0], (min(a.batch, pos.shape[0]),))
            xb = pos[p].to(DEV)
            rec, code = tok(xb)
            loss = F.mse_loss(rec, xb)
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
        with torch.no_grad():
            _, code_all = tok(pos.to(DEV))
        code_all = code_all.cpu()
        m = mers >= 0
        idx = code_all[m].long() * (4 ** K) + mers[m]
        cnt_flat.index_add_(0, idx, torch.ones(idx.numel(), dtype=torch.int32))
        if ci % 10 == 0 or ci == nchunk - 1:
            el = time.time() - t0
            done = ci + 1
            print(f"[train] chunk {done}/{nchunk}  steps={step}  loss={float(loss):.4f}  "
                  f"{el/done:.1f}s/chunk  eta {(nchunk-done)*el/done/60:.0f} min", flush=True)
        if ci % 20 == 0:
            torch.save(dict(state=tok.state_dict(), step=step, chunk=ci,
                            latent=a.latent, levels=a.levels, hidden=a.hidden,
                            vocab=VOCAB, dim=768), out / "state_latest.pt")

    # ---- metrics on held-out
    tok.eval()
    with torch.no_grad():
        rec, vcode = tok(val_x.to(DEV))
        mse = float(F.mse_loss(rec, val_x.to(DEV)))
        vcode = vcode.cpu()
    vcnt = torch.bincount(vcode, minlength=VOCAB)
    used = int((vcnt > 0).sum())
    once = float((vcnt == 1).sum()) / max(used, 1)
    med = float(vcnt[vcnt > 0].float().median()) if used else 0.0

    cnt = cnt_flat.reshape(VOCAB, 4 ** K)
    lk = cnt.argmax(dim=1).long()
    vmers = torch.cat([mer_ids(w) for w in val_wins])
    m = vmers >= 0
    pred = lk[vcode[m]]
    acc = float((pred == vmers[m]).float().mean()) if m.any() else 0.0
    g = torch.Generator().manual_seed(99)
    perm = torch.randperm(int(m.sum()), generator=g)
    ctrl = float((lk[vcode[m]][perm] == vmers[m]).float().mean())
    base = float(torch.bincount(vmers[m], minlength=4 ** K).max()) / int(m.sum())
    lift = acc / max(ctrl, 1e-9)

    print(f"\n[metrics] val_mse={mse:.4f} used={used}/{VOCAB} ({100*used/VOCAB:.1f}%) "
          f"once={100*once:.2f}% median_reuse={med:.0f}", flush=True)
    print(f"[transfer] acc={100*acc:.2f}% ctrl={100*ctrl:.2f}% base={100*base:.2f}% "
          f"**lift={lift:.2f}x** (baseline 2.63x)", flush=True)

    torch.save(dict(state=tok.state_dict(), dim=768, latent=a.latent, levels=a.levels,
                    hidden=a.hidden, vocab=VOCAB), out / "tokenizer.pt")
    rep = dict(args=vars(a), vocab=VOCAB, n_trained=len(train_iter), n_val=len(val_wins),
               steps=step, mse=mse, used=used, used_pct=used / VOCAB,
               once_frac=once, median_reuse=med,
               transfer=dict(acc=acc, ctrl=ctrl, base=base, lift=lift),
               minutes=(time.time() - t0) / 60)
    (out / "report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
