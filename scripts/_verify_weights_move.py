"""逐元素核对：训练到底有没有改动投影权重。

起因：`checkpoints/_losscheck`（100 步）与 `checkpoints/_timing_gc`（8 步）的
`proj.weight` 统计量完全相同（std=0.02283）。bf16 在 0.023 附近的 ULP ≈ 1.5e-4，
100 步 × lr 1e-4 的更新理应远超这个分辨率。所以必须逐元素确认。

做法：fresh 初始化 -> 抓好权重的 fp32 副本 -> 跑 N 步 -> 逐元素比差。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402

from rnaqwen.models.alignment import Stage1Aligner, Stage1Config, cosine_with_warmup  # noqa: E402

torch.manual_seed(0)
rows = [json.loads(l) for l in (ROOT / "data" / "processed" / "annotations_50.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
recs = [{"sequence": r["sequence"], "annotation": r["annotation"]} for r in rows[:4]]

dev = torch.device("cuda")
cfg = Stage1Config(grad_checkpointing=True)
model = Stage1Aligner(cfg)
model.encoder.to(dev)
model.llm.to(dev)
for m in model.trainable_modules():
    m.to(dev)
    m.train()

w0 = model.projection.proj.weight.detach().float().clone()
b0 = model.projection.proj.bias.detach().float().clone()
print(f"init   w: std={w0.std():.6f} max={w0.abs().max():.6f}")

opt = torch.optim.AdamW(model.projection.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=0.05)
print(f"优化器参数组里的张量数: {sum(len(g['params']) for g in opt.param_groups)}")
print(f"参数组里的张量: {[tuple(p.shape) for g in opt.param_groups for p in g['params']]}")

lrs = []
for step in range(5):
    rec = recs[step % len(recs)]
    model.zero_grad(set_to_none=True)
    loss = model.forward_loss([rec["sequence"]], [rec["annotation"]], dev)
    loss.backward()
    gn = sum(float(p.grad.float().norm() ** 2) for p in model.projection.parameters()
             if p.grad is not None) ** 0.5
    lr = cosine_with_warmup(step, Stage1Config(lr=1e-4, warmup_lr=1e-6, warmup_steps=10), 100)
    for g in opt.param_groups:
        g["lr"] = lr
    torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    w = model.projection.proj.weight.detach().float()
    dw = (w - w0)
    lrs.append(lr)
    print(f"step {step}: loss={float(loss):.4f} lr={lr:.3e} grad_norm={gn:.4f} "
          f"|dw|max={dw.abs().max():.3e} |dw|mean={dw.abs().mean():.3e} "
          f"w_std={w.std():.6f}")

db = (model.projection.proj.bias.detach().float() - b0)
print(f"\nbias |db|max={db.abs().max():.3e} |db|mean={db.abs().mean():.3e}")
print(f"lr 序列: {[f'{x:.2e}' for x in lrs]}")

moved = float((model.projection.proj.weight.detach().float() - w0).abs().max())
print(f"\n判定: 5 步后权重最大变化 = {moved:.3e} -> "
      f"{'OK，权重确实在动' if moved > 1e-6 else '❌ 权重没动，训练是空转'}")
