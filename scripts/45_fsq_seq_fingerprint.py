"""决定性对照：FSQ 的码编码的是「这条序列」还是「这个局部结构」？

背景：44_ 测出跨序列覆盖率**低于** iid 基线（词表 4096: 95.8% vs 97.5%）。
低于基线只有一个解释 —— 码的使用在序列内成簇，每条序列偏爱自己的一小撮码。

本脚本把这件事定死，两个对照：
  (a) 同序列内的位置留出：把训练序列的位置对半切，一半当"训练"、一半当"测试"。
      两者来自同样的序列 -> 若码是序列指纹，覆盖率应**远高于**基线。
  (b) 跨序列留出：整条序列留出 -> 44_ 已测，**低于**基线。

  (a) >> 基线 > (b)  ==> 码识别的是序列身份，不是可复用的局部结构。
                          接进 LLM 等于给每个位置发一个"哪条序列"的指纹，
                          换一条新序列就全是没见过的码 —— embedding 学不到东西。

  附：直接量 burstiness —— 同一个码在一条序列里连续出现的游程长度分布。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LATENT, LEVELS = 12, 2          # vocab 4096（44_ 里覆盖率最好的候选之一）
VOCAB = LEVELS ** LATENT
STEPS = 1500


def fsq(z, levels):
    t = (torch.tanh(z) + 1) / 2
    s = t * (levels - 1)
    k = torch.round(s)
    return (k / (levels - 1) * 2 - 1) + (s - s.detach()), k.long()


class AE(nn.Module):
    def __init__(self, dim, latent, levels, hidden=256):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.levels = levels
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        q, k = fsq(self.enc(x), self.levels)
        code = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            code = code * self.levels + k[:, d]
        return self.dec(q), code


def base_rate(tc: torch.Tensor, n: int, k: int, reps: int = 8) -> float:
    """iid 零假设下，从训练边缘分布独立抽 n 个码，其中训练计数>=k 的比例。"""
    p = tc / tc.sum()
    g = torch.Generator().manual_seed(1234)
    tot = 0.0
    for _ in range(reps):
        s = torch.multinomial(p.cpu(), n, replacement=True, generator=g).to(tc.device)
        tot += float((tc[s] >= k).float().mean())
    return tot / reps


def main() -> int:
    CACHE = ROOT / "data/processed/_fsq_probe_x.pt"
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    x, sid = d["x"].to(DEV), d["sid"].to(DEV)
    N, D = x.shape
    print(f"[data] {N} 位置 / {int(sid.max())+1} 条序列（缓存）")

    g = torch.Generator().manual_seed(7)
    test_ids = set(torch.randperm(int(sid.max()) + 1, generator=g)[:64].tolist())
    seq_tr = torch.tensor([s not in test_ids for s in sid.tolist()], device=DEV)

    # 只在「训练序列」上训 AE（与 44_ 一致）
    torch.manual_seed(0)
    ae = AE(D, LATENT, LEVELS).to(DEV)
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    idx = torch.nonzero(seq_tr).squeeze(1)
    for _ in range(STEPS):
        p = idx[torch.randint(0, idx.numel(), (512,), device=DEV)]
        rec, _ = ae(x[p])
        F.mse_loss(rec, x[p]).backward()
        opt.step(); opt.zero_grad()
    ae.eval()
    with torch.no_grad():
        _, code = ae(x)

    print()
    print("=" * 78)
    print("burstiness：同一个码在某条序列内的连续游程长度")
    print("=" * 78)
    runs = []
    order = torch.argsort(sid, stable=True)
    cs, ss = code[order], sid[order]
    run = 1
    for i in range(1, len(cs)):
        if ss[i] == ss[i - 1] and cs[i] == cs[i - 1]:
            run += 1
        else:
            runs.append(run); run = 1
    runs.append(run)
    rt = torch.tensor(runs, dtype=torch.float)
    print(f"  游程数 {len(runs)}（位置 {N}）平均长 {rt.mean():.2f}  中位 {rt.median():.0f}  "
          f"p90 {torch.quantile(rt, 0.9):.0f}  max {int(rt.max())}")
    print(f"  若码在序列内 iid，平均游程应 ≈ 1。实测 {rt.mean():.2f} —— "
          f"{'显著成簇' if rt.mean() > 1.5 else '接近 iid'}")

    print()
    print("=" * 78)
    print("决定性对照：同序列内留出 (a) vs 跨序列留出 (b)")
    print("=" * 78)
    # (a) 训练序列的位置对半切
    idx_tr = torch.nonzero(seq_tr).squeeze(1)
    g2 = torch.Generator().manual_seed(11)
    half = torch.randperm(idx_tr.numel(), generator=g2)
    A_tr, A_te = idx_tr[half[:half.numel() // 2]], idx_tr[half[half.numel() // 2:]]
    # (b) 跨序列
    B_tr = idx_tr
    B_te = torch.nonzero(~seq_tr).squeeze(1)

    print(f"{'对照':<26} {'测试位':>8} | {'>=1次':>8} {'基线':>8} {'超出':>8} | "
          f"{'>=5次':>8} {'基线':>8} {'超出':>8}")
    res = {}
    for name, tr_i, te_i in (("(a) 同序列内留出", A_tr, A_te),
                             ("(b) 跨序列留出", B_tr, B_te)):
        tc = torch.bincount(code[tr_i], minlength=VOCAB).float()
        ec = torch.bincount(code[te_i], minlength=VOCAB).float()
        pe = ec / ec.sum()
        n = int(te_i.numel())
        line = f"{name:<26} {n:>8} |"
        vals = {}
        for k in (1, 5):
            cov = float(pe[tc >= k].sum())
            b = base_rate(tc, n, k)
            vals[k] = (cov, b)
            line += f" {100*cov:>7.1f}% {100*b:>7.1f}% {100*(cov-b):>+7.1f}% |"
        print(line)
        res[name] = vals

    a1 = res["(a) 同序列内留出"][5][0] - res["(a) 同序列内留出"][5][1]
    b1 = res["(b) 跨序列留出"][5][0] - res["(b) 跨序列留出"][5][1]
    print()
    print("=" * 78)
    print("判定")
    print("=" * 78)
    print(f"  (a) 同序列内留出: >=5次覆盖超出基线 {100*a1:+.1f}pp")
    print(f"  (b) 跨序列留出  : >=5次覆盖超出基线 {100*b1:+.1f}pp")
    print()
    if a1 > 0.10 and b1 < 0.0:
        print("  ❌ **码是「序列指纹」，不是可复用的局部词表。**")
        print("     同一条序列内部，码高度自洽（覆盖率远高于基线）；")
        print("     换一条没见过的序列，码就全落在训练罕见区（覆盖率低于基线）。")
        print("     这说明每条序列偏爱自己的一小撮码 —— 码区分的是「这是哪条序列」，")
        print("     不是「这里是什么结构」。")
        print()
        print("     对 LLM 的含义：新序列进来时给的是一串没怎么见过的码，")
        print("     embedding 表里那些条目只见过少数样本，学不到可迁移的东西。")
        print("     **不是把 FSQ 的输出直接接到 Qwen 就能解决的问题。**")
    elif a1 > 0.10 and b1 >= -0.02:
        print("  ⚠️ 混合：序列内自洽，跨序列也不亏。码有部分可复用性，值得小规模试。")
    else:
        print("  ✅ 码没有明显的序列特异性，跨序列可复用。可以接 LLM。")

    out = ROOT / "reports" / "fsq_seq_fingerprint.json"
    out.write_text(json.dumps(dict(
        vocab=VOCAB, latent=LATENT, levels=LEVELS,
        run_mean=float(rt.mean()), run_p90=float(torch.quantile(rt, 0.9)),
        within_holdout_excess_5=a1, cross_holdout_excess_5=b1),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
