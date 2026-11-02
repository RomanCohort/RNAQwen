"""Finalise a streaming codebook run whose metric step crashed.

49_stream_tokenizer.py computes its validation split as
`a.n_seq - a.val`, but the loader drops windows containing N/R/Y.  When the
requested `--n-seq` equals the whole file, `kept` never reaches `val_start`
and `val_wins` is empty -> torch.cat([]) -> exit 1 *after* all chunks trained.

This script takes the surviving `state_latest.pt` and:
  1. writes `tokenizer.pt` in the format 50_routeC_e2e.py expects
  2. reports mse / usage / reuse on a freshly sampled window set
  3. reports a transfer number, explicitly labelled in-sample when the run
     had no clean holdout
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 6


class FSQTokenizer(nn.Module):
    def __init__(self, dim, latent, levels, hidden=256):
        super().__init__()
        self.dim, self.latent, self.levels = dim, latent, levels
        self.enc = nn.Linear(dim, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        t = torch.tanh(self.enc(x))
        s = (t + 1) / 2 * (self.levels - 1)
        k = torch.round(s)
        q = k / (self.levels - 1) * 2 - 1
        c = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            c = c * self.levels + k[:, d].long()
        return self.dec(q), c


def mer_ids(win, k=K):
    b = win.encode()
    out = torch.full((len(b),), -1, dtype=torch.long)
    a = 0
    for i, ch in enumerate(b):
        a = ((a * 4) + {65: 0, 67: 1, 71: 2, 84: 3, 85: 0}.get(ch, 0)) & (4 ** k - 1)
        if i >= k - 1:
            out[i - k + 1] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(ROOT / "checkpoints/fsq_stream/state_latest.pt"))
    ap.add_argument("--jsonl", default=str(ROOT / "data/processed/rnacentral_2m.jsonl"))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--window", type=int, default=101)
    ap.add_argument("--out", default=str(ROOT / "checkpoints/fsq_stream"))
    a = ap.parse_args()

    torch.set_num_threads(int(__import__("os").environ.get("RNAQ_THREADS", "8")))
    st = torch.load(a.state, map_location="cpu", weights_only=False)
    lat, lev, hid = int(st["latent"]), int(st["levels"]), int(st["hidden"])
    vocab = lev ** lat
    tok = FSQTokenizer(768, lat, lev, hid).to(DEV)
    tok.load_state_dict(st["state"])
    tok.eval()
    print(f"[state] chunk={st.get('chunk')} step={st.get('step')} "
          f"latent={lat} levels={lev} vocab={vocab}", flush=True)

    rng = random.Random(0)
    wins = []
    with open(a.jsonl, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if len(wins) >= a.n:
                break
            r = json.loads(line)
            s = (r.get("sequence") or "").upper().replace("T", "U")
            if not s:
                continue
            if len(s) <= a.window:
                w = s
            else:
                j = rng.randrange(0, len(s) - a.window + 1)
                w = s[j:j + a.window]
            if any(c not in "ACGU" for c in w):
                continue
            wins.append(w)
    print(f"[data] sampled windows={len(wins):,}", flush=True)

    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    xs, mer_list = [], []
    with torch.no_grad():
        for i in range(0, len(wins), 64):
            chunk = wins[i:i + 64]
            o = enc(chunk)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, w in enumerate(chunk):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(w) + 2:
                    hh = hh[1:-1]
                if hh.shape[0] >= 4:
                    xs.append(hh.cpu())
                    mer_list.append(mer_ids(w))
            if (i // 64) % 50 == 0:
                print(f"  [enc] {i+len(chunk)}/{len(wins)}", flush=True)
    x = torch.cat(xs)
    mers = torch.cat(mer_list)
    print(f"[enc] positions={x.shape[0]:,}", flush=True)

    with torch.no_grad():
        rec, code = tok(x.to(DEV))
        mse = float(F.mse_loss(rec, x.to(DEV)))
        code = code.cpu()

    cnt = torch.bincount(code, minlength=vocab)
    used = int((cnt > 0).sum())
    once = float((cnt == 1).sum()) / max(used, 1)
    med = float(cnt[cnt > 0].float().median()) if used else 0.0

    # code -> most frequent 6-mer, evaluated on a 20% split of THESE windows.
    # NOTE: these sequences were part of training, so this is in-sample.
    g = torch.Generator().manual_seed(7)
    nseq = len(mer_list)
    seq_id = torch.repeat_interleave(torch.arange(nseq),
                                     torch.tensor([len(m) for m in mer_list]))
    test_ids = set(torch.randperm(nseq, generator=g)[: max(1, nseq // 5)].tolist())
    tr = torch.tensor([s not in test_ids for s in seq_id.tolist()])
    m = mers >= 0
    trp = torch.nonzero(tr & m).squeeze(1)
    tep = torch.nonzero(~tr & m).squeeze(1)
    lk = torch.full((vocab,), -1, dtype=torch.long)
    order = torch.argsort(code[trp])
    cs, ms = code[trp][order], mers[trp][order]
    uniq, ccount = torch.unique_consecutive(cs, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(ccount, 0)[:-1]])
    for u, s0, c0 in zip(uniq.tolist(), starts.tolist(), ccount.tolist()):
        lk[u] = int(torch.bincount(ms[s0:s0 + c0], minlength=4 ** K).argmax())
    ec, em = code[tep], mers[tep]
    h = lk[ec] >= 0
    acc = float((lk[ec][h] == em[h]).float().mean()) if h.any() else 0.0
    perm = torch.randperm(int(h.sum()), generator=torch.Generator().manual_seed(99))
    ctrl = float((lk[ec][h][perm] == em[h]).float().mean()) if h.any() else 0.0

    print(f"[metrics] mse={mse:.4f} used={used}/{vocab} ({100*used/vocab:.1f}%) "
          f"once={100*once:.2f}% median_reuse={med:.0f}", flush=True)
    print(f"[transfer] acc={100*acc:.2f}% ctrl={100*ctrl:.2f}% "
          f"lift={acc/max(ctrl,1e-9):.2f}x  (IN-SAMPLE, not a clean holdout)", flush=True)

    out = Path(a.out)
    torch.save(dict(state=tok.state_dict(), dim=768, latent=lat, levels=lev,
                    hidden=hid, vocab=vocab), out / "tokenizer.pt")
    rep = dict(source_state=a.state, chunk=st.get("chunk"), step=st.get("step"),
               vocab=vocab, latent=lat, levels=lev, hidden=hid,
               n_windows=len(wins), n_positions=int(x.shape[0]),
               mse=mse, used=used, used_pct=used / vocab, once_frac=once,
               median_reuse=med,
               transfer=dict(acc=acc, ctrl=ctrl, lift=acc / max(ctrl, 1e-9),
                             note="IN-SAMPLE: run had no clean holdout"))
    (out / "report_finalized.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[save] {out/'tokenizer.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
