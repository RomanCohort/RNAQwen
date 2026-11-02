"""路线 C · 第一阶段（全量）：训练全量压缩码表（FSQ tokenizer）

与 47_ 的区别：47_ 只在 1500 条 RNA 上训码表（只用到 2754/4096 个码字）。
这里用**全部 4481 条**带注释的 RNA，产出正式的第一阶段产物：

    checkpoints/fsq_full/tokenizer.pt    FSQ 权重（enc + dec）
    checkpoints/fsq_full/codes.pt        uid -> 码序列（下游直接用，不必再跑 encoder）
    reports/fsq_full.json                重建 / 利用率 / 跨序列迁移三项指标

**为什么要带上「跨序列迁移」这一项**：46_ 已证明重建误差和利用率都能骗人
（重建是循环的 —— 我们就是照它训的；利用率在词表远小于样本数时必然好看）。
唯一非循环的判据是：在**没见过的序列**上，同一个码是否指向同一类局部 6-mer。
所以这个脚本训完必须把它一起报出来，不然不知道码表是不是真货。

用法:
    python scripts/48_stage1_full_tokenizer.py
    python scripts/48_stage1_full_tokenizer.py --latent 12 --levels 2 --steps 6000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NT = 1024
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
ERNIE = ROOT / "models/ernierna"
OUT = ROOT / "checkpoints/fsq_full"
POS_CACHE = ROOT / "data/processed/_fsq_full_pos.pt"
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


def load_all():
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence") or ""
            if s and len(s) <= MAX_NT:
                recs.append(r)
    return recs


def encode_all(recs, batch=16):
    """分块过 ERNIE-RNA；丢掉 [CLS]/[SEP]。返回 (位置张量, 每条序列的位置数, 每条序列的位置起始)"""
    from rnaqwen.models.encoder import RnaEncoder
    if POS_CACHE.exists():
        d = torch.load(POS_CACHE, map_location="cpu", weights_only=False)
        if d.get("n") == len(recs):
            print(f"[cache] {POS_CACHE.name}")
            return d["x"], d["lens"], d["offsets"]
    enc = RnaEncoder(str(ERNIE), dtype=torch.float32).to(DEV).eval()
    xs, lens = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(recs), batch):
            chunk = [r["sequence"] for r in recs[i:i + batch]]
            o = enc(chunk)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, s in enumerate(chunk):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(s) + 2:      # [CLS] ... [SEP]
                    hh = hh[1:-1]
                if hh.shape[0] == len(s):
                    xs.append(hh.cpu()); lens.append(len(s))
            del o, h, mk
            if (i // batch) % 40 == 0:
                el = time.time() - t0
                done = i + len(chunk)
                print(f"    [enc] {done}/{len(recs)}  {done/max(el,1e-9):.1f} seq/s  "
                      f"ETA {(len(recs)-done)/max(done/max(el,1e-9),1e-9)/60:.0f} min", flush=True)
    x = torch.cat(xs)
    lens = torch.tensor(lens)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), lens]), 0)[:-1]
    torch.save(dict(x=x, lens=lens, offsets=offsets, n=len(recs)), POS_CACHE)
    del enc
    import gc; gc.collect(); torch.cuda.empty_cache()
    return x, lens, offsets


def mer_of(seqs, K=K):
    """每条序列的局部 K-mer 编号（不足 K 的位置记 -1）。与 codes 对齐（无 [CLS]/[SEP]）。"""
    out = []
    for s in seqs:
        b = s.encode()
        L = len(b)
        v = torch.full((L,), -1, dtype=torch.long)
        for j in range(L - K + 1):
            a = 0
            for ch in b[j:j + K]:
                a = a * 4 + ({65: 0, 67: 1, 71: 2, 84: 3, 85: 0}.get(ch, 0))
            v[j] = a
        out.append(v)
    return torch.cat(out)


def transfer_metric(codes, mers, seq_id, n_seq, vocab, hold=0.2, seed=7):
    """46_ 的那个非循环判据：训练序列上学 code->最常见 6-mer，考没见过序列。"""
    g = torch.Generator().manual_seed(seed)
    test_ids = set(torch.randperm(n_seq, generator=g)[:max(1, int(n_seq * hold))].tolist())
    tr = torch.tensor([s not in test_ids for s in seq_id.tolist()])
    m = mers >= 0
    tr_p = torch.nonzero(tr & m).squeeze(1)
    te_p = torch.nonzero(~tr & m).squeeze(1)
    if tr_p.numel() == 0 or te_p.numel() == 0:
        return None
    tc, tm = codes[tr_p], mers[tr_p]
    ec, em = codes[te_p], mers[te_p]

    def acc_of(c_src, m_src, c_eval, m_eval):
        lk = torch.full((vocab,), -1, dtype=torch.long)
        order = torch.argsort(c_src)
        cs, ms = c_src[order], m_src[order]
        uniq, cnt = torch.unique_consecutive(cs, return_counts=True)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)[:-1]])
        for u, st, cn in zip(uniq.tolist(), starts.tolist(), cnt.tolist()):
            lk[u] = int(torch.bincount(ms[st:st + cn], minlength=4 ** K).argmax())
        h = lk[c_eval] >= 0
        return (float((lk[c_eval][h] == m_eval[h]).float().mean()) if h.any() else 0.0,
                float(h.float().mean()))

    acc, cov = acc_of(tc, tm, ec, em)
    g2 = torch.Generator().manual_seed(99)
    ctrls = []
    for _ in range(3):
        perm = torch.randperm(tc.numel(), generator=g2)
        ctrls.append(acc_of(tc[perm], tm, ec, em)[0])
    ctrl = sum(ctrls) / len(ctrls)
    base = float(torch.bincount(tm, minlength=4 ** K).max()) / tm.numel()
    return dict(acc=acc, ctrl=ctrl, base=base, lift=acc / max(ctrl, 1e-9),
                cov=cov, n_train_pos=int(tr_p.numel()), n_test_pos=int(te_p.numel()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=12)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()
    VOCAB = args.levels ** args.latent
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"=== 路线 C 第一阶段 · 全量 ===  vocab {VOCAB} "
          f"(latent {args.latent}, levels {args.levels})")
    recs = load_all()
    print(f"[data] {len(recs)} 条 RNA（全部带注释序列，len<={MAX_NT}）")
    seqs = [r["sequence"] for r in recs]

    x, lens, offsets = encode_all(recs)
    seq_id = torch.repeat_interleave(torch.arange(len(lens)), lens)
    print(f"[pos] {x.shape[0]} 个位置  dim {x.shape[1]}  中位序列长 {int(lens.median())}")
    assert x.shape[0] == int(lens.sum()) == seq_id.numel()

    print(f"\n=== 训练 FSQ（{args.steps} 步, batch {args.batch}）===")
    torch.manual_seed(0)
    tok = FSQTokenizer(x.shape[1], args.latent, args.levels, args.hidden).to(DEV)
    opt = torch.optim.AdamW(tok.parameters(), lr=args.lr)
    # 位置张量留在 CPU：2.4M x 768 fp32 约 7.4 GB，放在共享显存上会挤到 Qwen
    xs = x
    N = xs.shape[0]
    for it in range(args.steps):
        p = torch.randint(0, N, (args.batch,))
        xb = xs[p].to(DEV, non_blocking=True)
        rec, _ = tok(xb)
        loss = F.mse_loss(rec, xb)
        loss.backward(); opt.step(); opt.zero_grad()
        if it % max(1, args.steps // 12) == 0:
            print(f"  step {it:>5}  mse {float(loss):.4f}  {time.time()-t0:.0f}s", flush=True)
    tok.eval()

    print("\n=== 码表统计 ===")
    codes = torch.empty(N, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, N, 200000):
            _, c = tok(xs[i:i + 200000].to(DEV))
            codes[i:i + 200000] = c.cpu()
            del c
        xb = xs[:200000].to(DEV)
        rec, _ = tok(xb)
        mse = float(F.mse_loss(rec, xb))
        tot_var = float(xb.var(dim=0).sum())
        err_norm = mse * xb.shape[1] / tot_var
        del xb, rec
    cnt = torch.bincount(codes, minlength=VOCAB)
    used = int((cnt > 0).sum())
    med = float(cnt[cnt > 0].float().median())
    once = float((cnt == 1).sum()) / max(used, 1)
    print(f"  重建 mse {mse:.4f}  归一化 err {err_norm:.4f}")
    print(f"  用码 {used}/{VOCAB} ({100*used/VOCAB:.1f}%)  位置/用码 {N/used:.1f}  "
          f"中位重用 {med:.0f}  仅1次 {100*once:.1f}%")

    print("\n=== 跨序列迁移（唯一非循环的判据，见 46_）===")
    mers = mer_of(seqs)
    tm = transfer_metric(codes, mers, seq_id, len(seqs), VOCAB)
    if tm:
        print(f"  训练位置 {tm['n_train_pos']}  测试位置 {tm['n_test_pos']}")
        print(f"  按码预测 6-mer 准确率 {100*tm['acc']:.2f}%  vs 打乱对照 "
              f"{100*tm['ctrl']:.2f}%  vs 常数 {100*tm['base']:.2f}%")
        print(f"  **提升 {tm['lift']:.2f}x**（46_ 在 1500 条上测到 2.88x）")

    # --- 存盘 ---
    torch.save(dict(state=tok.state_dict(), dim=x.shape[1], latent=args.latent,
                    levels=args.levels, hidden=args.hidden, vocab=VOCAB),
               OUT / "tokenizer.pt")
    # uid -> 码（下游直接用，不必再跑 encoder）
    cmap = {}
    for i, r in enumerate(recs):
        a, b = int(offsets[i]), int(offsets[i]) + int(lens[i])
        cmap[r["uid"]] = codes[a:b].to(torch.int32)
    torch.save(dict(codes=cmap, vocab=VOCAB, latent=args.latent, levels=args.levels),
               OUT / "codes.pt")

    rep = dict(args=vars(args), vocab=VOCAB, n_rna=len(recs), n_pos=N,
               mse=mse, err_norm=err_norm, used=used, used_pct=used / VOCAB,
               pos_per_code=N / used, median_reuse=med, once_frac=once,
               transfer=tm, minutes=(time.time() - t0) / 60)
    (ROOT / "reports/fsq_full.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT/'tokenizer.pt'}  {OUT/'codes.pt'}  reports/fsq_full.json")
    print(f"       用时 {(time.time()-t0)/60:.1f} min")

    print("\n" + "=" * 72)
    if tm and tm["lift"] >= 2.0:
        print(f"  ✅ 码表可用于下游：迁移 {tm['lift']:.2f}x，用码 {100*used/VOCAB:.0f}%，"
              f"中位重用 {med:.0f}")
    elif tm:
        print(f"  ⚠️ 迁移只有 {tm['lift']:.2f}x，比 46_ 的 2.88x 弱。")
        print("     下游可以跑，但先别期待结果；考虑加复用压力再重训。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
