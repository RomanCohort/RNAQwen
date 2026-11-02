"""有限词表 FSQ —— 路线 C 的第二道闸门。

v2 的问题：latent=128 × l=16 = 16^128 个名义码，**每个位置一个独一无二的码**。
那测的是高码率编码，不是 tokenizer。**喂 LLM 的词表必须有限、跨位置共享。**

这一版固定词表规模，改「怎么把 bit 预算分配在维度和级数上」：

    4096 码   的几种分法:  4×8=(l^d=8^4)  6×4=(4^6)  12×2=(2^12)
    65536 码  的几种分法:  4×16         8×4        16×2

三个判据：
  ① 重建误差（能不能留住信息）
  ② 码字利用率（塌缩 = 所有输入挤到少数几个码；RiboSphere 说稀疏但有选择性是好的，
     真正的病是**塌**）
  ③ **词内输入方差 / 总方差** —— 同一个码对应的输入是不是真的相似？
     这是"码字有没有意义"的直接度量，也是 RiboSphere "codes enriched for RNA motifs" 的代理
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
N_SEQ = 64
STEPS = 1500
ANISO = ROOT / "data/processed/ernierna_aniso_mean.pt"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"

# 固定词表规模下的几种 (latent, levels) 分法
CONFIGS = [
    (4096, [(4, 8), (6, 4), (12, 2)]),
    (65536, [(4, 16), (8, 4), (16, 2)]),
]


class FSQAE(nn.Module):
    def __init__(self, dim: int, latent: int, levels: int):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.dec = nn.Linear(latent, dim)
        self.levels = levels
        self.half = (levels - 1) / 2.0

    def forward(self, x):
        z = torch.tanh(self.enc(x))
        q = torch.round(z * self.half) / self.half
        q = q + (z - z.detach())
        idx = (torch.round(z * self.half) + self.half).long()
        return self.dec(q), q, idx


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
        h = o.hidden.float().to(DEV)
        mk = o.mask.to(DEV).squeeze(-1) > 0
    x = h[mk]
    N, D = x.shape
    var_tot = float(x.var(dim=0).sum())
    print(f"[fsq] {len(seqs)} 条序列，有效位置 {N}，维度 {D}")

    results = []
    for vocab, splits in CONFIGS:
        print(f"\n=== 固定词表 {vocab} ===")
        for latent, levels in splits:
            assert levels ** latent == vocab, (latent, levels, levels ** latent)
            torch.manual_seed(0)
            ae = FSQAE(D, latent, levels).to(DEV)
            opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
            t0 = time.time()
            for _ in range(STEPS):
                p = torch.randperm(N, device=DEV)[:512]
                xb = x[p]
                rec, _, _ = ae(xb)
                loss = F.mse_loss(rec, xb) * D
                opt.zero_grad(); loss.backward(); opt.step()
            ae.eval()
            with torch.no_grad():
                rec, q, idx = ae(x)
                err = float(((rec - x) ** 2).sum(1).mean() / var_tot)
                # 组合成单一码字 id
                code = torch.zeros(N, dtype=torch.long, device=DEV)
                for d_ in range(latent):
                    code = code * levels + idx[:, d_]
                uniq, inv, counts = torch.unique(code, return_inverse=True, return_counts=True)
                n_used = int(uniq.numel())
                # 词内输入方差：同一个码对应的输入有多像
                # sum_over_codes sum_{i in c} ||x_i - mean_c||^2  /  (N * var_tot)
                sums = torch.zeros(n_used, D, device=DEV)
                sums.index_add_(0, inv, x)
                means = sums / counts.unsqueeze(1)
                within = float(((x - means[inv]) ** 2).sum(1).mean() / var_tot)
                # 熵：码字分布有多散（塌缩时熵低）
                p = counts.float() / N
                ent = float(-(p * (p + 1e-12).log()).sum())
                ent_norm = ent / math.log(vocab)
                top1 = float(counts.max()) / N
            results.append({"vocab": vocab, "latent": latent, "levels": levels,
                            "err": err, "codes_used": n_used,
                            "util": n_used / vocab, "within_var": within,
                            "entropy_norm": ent_norm, "top1_share": top1,
                            "n_pos": N})
            print(f"  latent={latent:>2} l={levels:>2}  err {err:.4f}  "
                  f"用到码字 {n_used:>6}/{vocab} ({100*n_used/vocab:.1f}%)  "
                  f"词内方差 {within:.4f}  归一熵 {ent_norm:.3f}  "
                  f"最大码占比 {100*top1:.2f}%  {time.time()-t0:.0f}s")

    out = ROOT / "reports/fsq_finite_vocab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fsq] -> {out}")

    print()
    for vocab, _ in CONFIGS:
        rs = [r for r in results if r["vocab"] == vocab]
        b = min(rs, key=lambda r: r["err"])
        print(f"  词表 {vocab:>6}: 最优 latent={b['latent']} l={b['levels']}  "
              f"err {b['err']:.4f}  词内方差 {b['within_var']:.4f}  "
              f"利用率 {100*b['util']:.1f}%  归一熵 {b['entropy_norm']:.3f}")
    print()
    print("  读法：")
    print("    err        < 1 才说明比'输出均值'好")
    print("    词内方差    越低 = 同一个码对应的输入越相似 = 码字越有意义")
    print("    归一熵      低 = 少数码占据多数位置（塌缩）；RiboSphere 说**有选择性的稀疏是好的**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
