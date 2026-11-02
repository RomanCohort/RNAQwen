"""路线 C 的定方向实验：修公式 + 换解码器。

两个改动：
  ① 修 FSQ 公式（v3 的 l=2 恒等输出零，是 bug 不是发现）
     v3:  q = round(tanh(z) * (l-1)/2) / ((l-1)/2)      -> l=2 时 tanh(z)*0.5 ∈(-0.5,0.5)，round 恒 0
     v4:  k = round((tanh(z)+1)/2 * (l-1))              -> k ∈ {0..l-1}，正好 l 档
          q = k/(l-1)*2 - 1                              -> 落在 [-1,1]
  ② 解码器 Linear -> MLP
     假设："err 0.64 是解码器太弱，不是词表不够大"
     若换 MLP 后 err 大降 -> 瓶颈是解码器（好消息，词表可以小）
     若几乎不动       -> 真的是词表容量不足，路线要重估

判据（决定路线成不成）：
    err  —— 能不能留住信息（< 1 才比"输出均值"好）
    词内方差 —— 同一个码对应的输入有多像（越低 = 码字越有意义）
    最大码占比 —— 塌缩检测（RiboSphere: 有选择性的稀疏是好的，塌是坏的）
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rnaqwen.models.encoder import RnaEncoder  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SEQ, STEPS = 64, 2000
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"

CONFIGS = [
    (4096, 6, 4), (4096, 4, 8),
    (65536, 8, 4), (65536, 4, 16),
]


def fsq(z, levels):
    """v4 公式：正好 levels 档，l=2 时是 {0,1}，不退化。"""
    k = torch.round((torch.tanh(z) + 1) / 2 * (levels - 1))
    q = k / (levels - 1) * 2 - 1
    q = q + ((torch.tanh(z) + 1) / 2 * (levels - 1) - k).detach() * 0  # 纯 STE
    q = q + (((torch.tanh(z) + 1) / 2 * (levels - 1)) - ((torch.tanh(z) + 1) / 2 * (levels - 1)).detach())
    return q, k.long()


class AE(nn.Module):
    def __init__(self, dim, latent, levels, dec="linear"):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.levels = levels
        if dec == "linear":
            self.dec = nn.Linear(latent, dim)
        else:
            self.dec = nn.Sequential(
                nn.Linear(latent, 128), nn.GELU(),
                nn.Linear(128, 256), nn.GELU(),
                nn.Linear(256, dim))

    def forward(self, x):
        q, k = fsq(self.enc(x), self.levels)
        return self.dec(q), k


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    recs.sort(key=lambda r: len(r.get("sequence") or ""))
    pick = [recs[int(i * (len(recs) - 1) / (N_SEQ - 1))] for i in range(N_SEQ)]
    seqs = [r["sequence"] for r in pick if r.get("sequence")]
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()
    with torch.no_grad():
        o = enc(seqs)
        x = o.hidden.float().to(DEV)[o.mask.to(DEV).squeeze(-1) > 0]
    N, D = x.shape
    var_tot = float(x.var(dim=0).sum())
    print(f"[v4] {len(seqs)} 条序列，位置 {N}，维度 {D}")

    res = []
    for vocab, latent, levels in CONFIGS:
        assert levels ** latent == vocab
        for dec in ("linear", "mlp"):
            torch.manual_seed(0)
            ae = AE(D, latent, levels, dec).to(DEV)
            opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
            t0 = time.time()
            for _ in range(STEPS):
                p = torch.randperm(N, device=DEV)[:512]
                xb = x[p]
                rec, _ = ae(xb)
                loss = F.mse_loss(rec, xb) * D
                opt.zero_grad(); loss.backward(); opt.step()
            ae.eval()
            with torch.no_grad():
                rec, k = ae(x)
                err = float(((rec - x) ** 2).sum(1).mean() / var_tot)
                code = torch.zeros(N, dtype=torch.long, device=DEV)
                for d_ in range(latent):
                    code = code * levels + k[:, d_]
                uniq, inv, cnt = torch.unique(code, return_inverse=True, return_counts=True)
                sums = torch.zeros(uniq.numel(), D, device=DEV)
                sums.index_add_(0, inv, x)
                means = sums / cnt.unsqueeze(1)
                within = float(((x - means[inv]) ** 2).sum(1).mean() / var_tot)
                p_ = cnt.float() / N
                ent = float(-(p_ * (p_ + 1e-12).log()).sum()) / math.log(vocab)
                top1 = float(cnt.max()) / N
            res.append({"vocab": vocab, "latent": latent, "levels": levels, "dec": dec,
                        "err": err, "codes_used": int(uniq.numel()),
                        "util": float(uniq.numel()) / vocab, "within_var": within,
                        "entropy_norm": ent, "top1_share": top1, "sec": round(time.time()-t0)})
            print(f"  vocab {vocab:>6} latent={latent:>2} l={levels:>2} dec={dec:<6} "
                  f"err {err:.4f}  码字 {int(uniq.numel()):>5} ({100*uniq.numel()/vocab:>5.1f}%)  "
                  f"词内方差 {within:.4f}  归一熵 {ent:.3f}  最大码 {100*top1:.2f}%  {time.time()-t0:.0f}s")

    out = ROOT / "reports/fsq_v4_decoder.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[v4] -> {out}")

    print()
    print("  === 解码器是不是瓶颈 ===")
    for vocab, latent, levels in CONFIGS:
        a = next(r for r in res if r["vocab"] == vocab and r["dec"] == "linear")
        b = next(r for r in res if r["vocab"] == vocab and r["dec"] == "mlp")
        drop = 1 - b["err"] / max(a["err"], 1e-9)
        print(f"  vocab {vocab:>6} latent={latent} l={levels}: "
              f"linear {a['err']:.4f} -> mlp {b['err']:.4f}  ({100*drop:+.1f}%)")
    print()
    print("  === l=2 是否还退化（修公式的验证）===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
