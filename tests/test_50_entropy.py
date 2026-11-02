"""熵正则必须是可微的 —— 对应"l_ent 是常数"那个 bug。

老版本：
    codes = torch.cat(codes_all)          # round().long()，梯度为零
    hist  = torch.bincount(codes, ...)    # 只吃整数
    l_ent = -entropy(hist) * coef         # 常数 * coef = 常数
=> requires_grad=False，对参数零梯度。系数从 0.01 调到 1.0 都没用，因为
   它乘在一个常数上。这才是码一直塌缩的原因。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

t0 = time.time()
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else '**FAIL**'}  {name}" + (f"   {detail}" if detail else ""))


import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("routeC", ROOT / "scripts/50_routeC_e2e.py")
routeC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routeC)

DIM, LAT, LEV, DM, POOL = 16, 4, 2, 32, 4
VOCAB = LEV ** LAT


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fsq = routeC.FSQCode(DIM, LAT, LEV, DM, pool=POOL)
        self.rna_emb = nn.Embedding(VOCAB, DM)
        nn.init.normal_(self.rna_emb.weight, std=0.02)
        self.fsq_usage = None
        self._usage_parts = None

    rna_vectors = routeC.E2EModel.rna_vectors
    reset_usage = routeC.E2EModel.reset_usage


m = Stub()
tau = LAT / 2
x = torch.randn(200, DIM) * 8.7

print("=" * 78)
print("测试 A：老做法（bincount）算出来的熵是不可微的常数")
print("=" * 78)
out, code, _, _ = Stub.rna_vectors(m, x, tau=tau)
hist = torch.bincount(code, minlength=VOCAB).float()
p = hist / hist.sum()
ent_old = -(p * (p + 1e-9).log()).sum()
check("bincount 熵 requires_grad == False", not ent_old.requires_grad,
      f"requires_grad={ent_old.requires_grad}")
# 这证明老代码里 l_ent 是个常数，乘任何系数都没有梯度
l_ent_old = -ent_old * 1.0
check("老 l_ent 是常数（乘系数也没用）", not l_ent_old.requires_grad)

print()
print("=" * 78)
print("测试 B：新做法（软使用率）必须可微且能反传到 enc")
print("=" * 78)
u = m.fsq_usage
check("fsq_usage 已记录", u is not None, f"shape={tuple(u.shape) if u is not None else None}")
check("fsq_usage 形状 == (VOCAB,)", tuple(u.shape) == (VOCAB,))
check("fsq_usage 是概率（和≈1）", abs(float(u.sum()) - 1.0) < 1e-4, f"sum={float(u.sum()):.6f}")
check("fsq_usage requires_grad", u.requires_grad)

m.zero_grad()
u2 = m.fsq_usage
un = u2 / u2.sum().clamp(min=1e-9)
l_ent = (un * (un + 1e-9).log()).sum() * 1.0
check("新 l_ent requires_grad", l_ent.requires_grad, f"l_ent={float(l_ent):.4f}")
l_ent.backward()
ge = m.fsq.enc.weight.grad
check("熵正则能反传到 fsq.enc", ge is not None and float(ge.norm()) > 1e-6,
      f"|grad|={float(ge.norm()) if ge is not None else -1:.3e}")

print()
print("=" * 78)
print("测试 C：最大化熵确实让使用率变均匀（方向对不对）")
print("=" * 78)
m2 = Stub()
m2.load_state_dict({k: v for k, v in m.state_dict().items()
                    if k != "fsq_usage"}, strict=False)
m2.fsq_usage = None
opt = torch.optim.AdamW(m2.fsq.enc.parameters(), lr=0.05)
x2 = torch.randn(400, DIM) * 8.7
Stub.rna_vectors(m2, x2, tau=tau)          # 先跑一次，fsq_usage 才有值
h0 = float(-(lambda q: (q * (q + 1e-9).log()).sum())(m2.fsq_usage / m2.fsq_usage.sum()))
for _ in range(60):
    m2.reset_usage()      # 不清空的话，累加器里会留着上次已 backward 过的图的张量
    Stub.rna_vectors(m2, x2, tau=tau)
    uu = m2.fsq_usage
    un2 = uu / uu.sum().clamp(min=1e-9)
    loss = (un2 * (un2 + 1e-9).log()).sum()      # 最小化 -H == 最大化 H
    loss.backward(); opt.step(); opt.zero_grad()
h1 = float(-(lambda q: (q * (q + 1e-9).log()).sum())(m2.fsq_usage / m2.fsq_usage.sum()))
check("优化后熵上升", h1 > h0, f"{h0:.3f} -> {h1:.3f}")
check("优化后用到的码变多",
      int(m2.fsq_usage.gt(1e-4).sum()) >= int(m.fsq_usage.gt(1e-4).sum()),
      f"{int(m.fsq_usage.gt(1e-4).sum())} -> {int(m2.fsq_usage.gt(1e-4).sum())}")

print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败   用时 {time.time()-t0:.1f}s")
print("=" * 78)
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
    raise SystemExit(1)
print("全部通过")
