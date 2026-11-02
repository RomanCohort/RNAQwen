"""FSQ 重建实验（v2）—— 路线 C 的第一道闸门。

⚠️ v1 错在哪（记下来，这是"测出来的和以为测的不是一回事"的第五次）：
    v1 里 FSQ 返回的是**投影空间**的量化值，我却直接拿它和**原始 768 维**比，
    而那个投影层还是随机初始化、从没训过。
    于是那个"重建误差"量的其实是"随机线性映射 + 取整 能不能还原 x" —— 显然不能。
    报出来的 2.86 不是量化器的性质，是我的 bug。

正确做法：**编码器 → FSQ → 解码器**，训起来，然后比解码输出与原输入。

判据：误差 = MSE(重建, 原始) / Var(原始，逐位置)
     < 1  = 比"直接输出均值"好，保留了一些信息
     ≈ 1  = 什么都没保留
     > 1  = 比均值还差

同时报**码字利用率**：RiboSphere 发现稀疏但有选择性的使用是**好事**，
真正的病是"所有输入映射到同几个码字"（塌缩）。所以要看 unique 数，不是看它高不高。
"""
from __future__ import annotations

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

from rnaqwen.models.encoder import RnaEncoder  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_SEQ = 64
STEPS = 1500
ANISO = ROOT / "data/processed/ernierna_aniso_mean.pt"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"


class FSQAE(nn.Module):
    """编码器 -> FSQ -> 解码器。FSQ 无码本，所以不会塌缩。"""

    def __init__(self, dim: int, latent: int, levels: int):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.dec = nn.Linear(latent, dim)
        self.levels = levels
        self.half = (levels - 1) / 2.0

    def forward(self, x):
        z = torch.tanh(self.enc(x))                  # 落在 (-1,1)，FSQ 要求
        q = torch.round(z * self.half) / self.half   # 取到 l 个格点
        q = q + (z - z.detach())                     # straight-through
        idx = (torch.round(z * self.half) + self.half).long()   # 0..l-1
        return self.dec(q), idx


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
    mean = torch.load(ANISO, map_location="cpu")["mean"].to(DEV)
    with torch.no_grad():
        o = enc(seqs)
        h = o.hidden.float().to(DEV)
        mk = o.mask.to(DEV).squeeze(-1) > 0
    x = h[mk]                                        # (Npos, 768)
    print(f"[fsq] {len(seqs)} 条序列，有效位置 {x.shape[0]}，维度 {x.shape[1]}")

    results = []
    for tag, xin in [("原始", x), ("去公共方向", x - mean.to(x.dtype))]:
        var_ref = float(xin.var(dim=0).sum())        # 总方差（和）
        for latent in [16, 32, 64, 128]:
            for levels in [4, 8, 16]:
                torch.manual_seed(0)
                ae = FSQAE(768, latent, levels).to(DEV)
                opt = torch.optim.AdamW(ae.parameters(), lr=3e-3, weight_decay=0.0)
                t0 = time.time()
                for _ in range(STEPS):
                    perm = torch.randperm(xin.shape[0], device=DEV)[:512]
                    xb = xin[perm]
                    rec, _ = ae(xb)
                    loss = F.mse_loss(rec, xb) * xb.shape[1]      # 乘维度以匹配 var_ref 口径
                    opt.zero_grad(); loss.backward(); opt.step()
                ae.eval()
                with torch.no_grad():
                    rec, idx = ae(xin)
                    err = float(((rec - xin) ** 2).sum(1).mean() / var_ref)
                    uniq = int(torch.unique(idx, dim=0).shape[0])
                    ent = 0.0
                    for d_ in range(idx.shape[1]):
                        p = torch.bincount(idx[:, d_], minlength=levels).float()
                        p = p / p.sum()
                        ent += float(-(p * (p + 1e-12).log()).sum())
                results.append({"aniso": tag, "latent": latent, "levels": levels,
                                "err": err, "uniq_codes": uniq,
                                "bits": latent * (levels - 1).bit_length(),
                                "entropy_bits": ent, "n_pos": int(xin.shape[0]),
                                "sec": round(time.time() - t0, 1)})
                print(f"  {tag:<12} latent={latent:>3} l={levels:>2}  "
                      f"err {err:.4f}  码字组合 {uniq:>6}  "
                      f"{latent*(levels-1).bit_length():>4} bit/位置  {time.time()-t0:.0f}s")

    out = ROOT / "reports/fsq_reconstruction.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[fsq] -> {out}")

    best = min(results, key=lambda r: r["err"])
    print()
    print(f"  最优: {best['aniso']} latent={best['latent']} l={best['levels']}  "
          f"err {best['err']:.4f}  {best['bits']} bit/位置")
    print()
    print("  读法：err < 1 才说明比'输出均值'好；> 1 等于没留住东西。")
    print("        **这是表示重建，不是任务性能 —— 只是第一道闸门。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
