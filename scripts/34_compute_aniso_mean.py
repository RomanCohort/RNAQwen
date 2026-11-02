"""算 ERNIE-RNA 输出的「公共方向」（跨序列均值），存下来供训练时减掉。

为什么能这么做：**编码器是冻结的**，所以它的输出分布是个静态量。
均值方向可以离线算一次、固定住，训练时直接减 —— 不引入新参数，
不改变梯度流，完全可逆（去掉就是原样）。

实测动机：ERNIE-RNA 原始输出的模长有 **92.1%** 落在跨序列公共方向上。
RNA 特异的部分只占 40%（能量口径），再经压缩器和投影被稀释到 21%。
如果 LLM 拿到的向量九成是一个常数方向，它就只能靠少数极值位置去分辨不同 RNA
—— 这正好对应我们观察到的重尾依赖。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401
import torch  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402

N = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ROOT / "data/processed/ernierna_aniso_mean.pt"


def main() -> int:
    recs = [r.as_dict() for r in read_jsonl(ROOT / "data/processed/rnaqa_n6_merged.jsonl")]
    seqs, seen = [], set()
    for r in recs:
        s = r.get("sequence") or ""
        b = len(s) // 100                      # 按长度分层，别只抽到同一类
        if s and b not in seen:
            seen.add(b); seqs.append(s)
        if len(seqs) >= N:
            break
    seqs = seqs + [r["sequence"] for r in recs[: max(0, N - len(seqs))]]
    seqs = [s for s in seqs if s][:N]
    lens = sorted(len(s) for s in seqs)
    print(f"[aniso] {len(seqs)} 条序列，长度 {lens[0]}–{lens[-1]}（中位 {lens[len(lens)//2]}）")

    cfg = Stage1Config(encoder_path=str(ROOT / "models/ernierna"), dtype="float32")
    m = Stage1Aligner(cfg)
    del m  # 只借它的 encoder，避免多占一份 LLM 显存
    from rnaqwen.models.encoder import RnaEncoder

    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()

    # 累加 masked 均值
    total = None
    count = 0
    B = 8
    with torch.no_grad():
        for i in range(0, len(seqs), B):
            chunk = seqs[i:i + B]
            o = enc(chunk)
            h = o.hidden.float().to(DEV)
            mk = o.mask.to(DEV).unsqueeze(-1).float()
            s = (h * mk).sum(dim=(0, 1))        # (D,)
            total = s if total is None else total + s
            count += int(mk.sum())
    mean = total / max(count, 1)                # (768,)
    print(f"[aniso] 样本位置数 {count}")
    print(f"[aniso] 均值向量模长 {float(mean.norm()):.4f}")

    # 校验：拿一批序列看公共分量占比
    with torch.no_grad():
        o = enc(seqs[:32])
        h = o.hidden.float().to(DEV)
        mk = o.mask.to(DEV).unsqueeze(-1).float()
        pooled = (h * mk).sum(1) / mk.sum(1).clamp_min(1.0)         # (32, 768)
        mu = pooled.mean(0, keepdim=True)
        ratio = float(mu.norm() / pooled.norm(dim=-1).mean())
        after = float(((pooled - mu).norm(dim=-1) / pooled.norm(dim=-1)).mean())
    print(f"[aniso] 序列级：公共分量占模长 {100*ratio:.1f}%   去均值后保留 {100*after:.1f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean.cpu(), "hidden_size": int(mean.numel()),
                "n_positions": count, "n_sequences": len(seqs),
                "encoder": str(ROOT / "models/ernierna"),
                "public_ratio": ratio}, OUT)
    print(f"[aniso] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
