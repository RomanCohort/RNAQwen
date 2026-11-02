"""再审第 4 轮：fsq_usage 的作用域。

两个真实发生过的错法，都是"看着对、实际只覆盖 1/N"：
  · 用 bincount(codes) 算熵 -> codes 是 round().long()，梯度为零，熵是常数
  · fsq_usage 在逐条循环里被赋值 -> 只剩最后一条样本，且每步换一条

正确行为：整批累加，且每步清空。
"""
from __future__ import annotations

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
        self._usage_parts = None
        self.fsq_usage = None

    rna_vectors = routeC.E2EModel.rna_vectors
    reset_usage = routeC.E2EModel.reset_usage


def usage_of(m, x, tau=LAT / 2):
    """单独跑一条，返回它的 soft.mean(0)（不污染 m 的累积器）。"""
    saved, saved_parts = m.fsq_usage, m._usage_parts
    m._usage_parts = None
    Stub.rna_vectors(m, x, tau=tau)
    u = m.fsq_usage.clone()
    m.fsq_usage, m._usage_parts = saved, saved_parts
    return u


m = Stub()
tau = LAT / 2

print("=" * 78)
print("测试 A：多条件本必须累加，不能只留最后一条")
print("=" * 78)
# 造三条差异极大的序列：第一条全用了某些码，第二条完全不同
torch.manual_seed(0)
xs = [torch.randn(60, DIM) * 8.7 for _ in range(3)]
xs[1] = xs[1] * 5.0                       # 量级不同 -> 码分布不同

m.reset_usage()
for x in xs:
    Stub.rna_vectors(m, x, tau=tau)
u_acc = m.fsq_usage.clone()

u_last = usage_of(m, xs[-1], tau)
u_first = usage_of(m, xs[0], tau)
u_mean = (u_first + usage_of(m, xs[1], tau) + u_last) / 3

check("累加结果 != 最后一条", not torch.allclose(u_acc, u_last, atol=1e-6),
      f"max|diff|={float((u_acc-u_last).abs().max()):.3e}")
check("累加结果 == 三条的均值",
      torch.allclose(u_acc, u_mean, atol=1e-5),
      f"max|diff|={float((u_acc-u_mean).abs().max()):.3e}")
check("累加结果仍是概率", abs(float(u_acc.sum()) - 1) < 1e-4, f"sum={float(u_acc.sum()):.6f}")

print()
print("=" * 78)
print("测试 B：reset_usage 必须真的清空（否则跨步累积）")
print("=" * 78)
m.reset_usage()
check("reset 后无累积", m._usage_parts is None and m.fsq_usage is None)
Stub.rna_vectors(m, xs[0], tau=tau)
n1 = len(m._usage_parts)
Stub.rna_vectors(m, xs[1], tau=tau)
n2 = len(m._usage_parts)
check("两条后累积长度 == 2", n2 == 2, f"{n1} -> {n2}")
m.reset_usage()
check("再次 reset 后清空", m._usage_parts is None and m.fsq_usage is None)
Stub.rna_vectors(m, xs[0], tau=tau)
check("reset 后重新开始累积", len(m._usage_parts) == 1)

print()
print("=" * 78)
print("测试 C：整个累加链必须仍然可微（可微性 + 作用域两个条件要同时满足）")
print("=" * 78)
m.zero_grad()
m.reset_usage()
outs = [Stub.rna_vectors(m, x, tau=tau)[0] for x in xs]
u = m.fsq_usage
check("累加后 requires_grad", u.requires_grad)
un = u / u.sum().clamp(min=1e-9)
l_ent = (un * (un + 1e-9).log()).sum() * 1.0
l_ent.backward()
ge = m.fsq.enc.weight.grad
check("整批累加的熵能反传到 fsq.enc", ge is not None and float(ge.norm()) > 1e-6,
      f"|grad|={float(ge.norm()) if ge is not None else -1:.3e}")

print()
print("=" * 78)
print("测试 D：ent 的打印值（bincount）与梯度用的量（软使用率）不该差太多")
print("=" * 78)
m.reset_usage()
cs = []
for x in xs:
    _, c, _, _ = Stub.rna_vectors(m, x, tau=tau)
    cs.append(c)
codes = torch.cat(cs)
h = torch.bincount(codes, minlength=VOCAB).float()
p = h / h.sum()
ent_hard = float(-(p * (p + 1e-9).log()).sum())
u_c = m.fsq_usage / m.fsq_usage.sum()
ent_soft = float(-(u_c * (u_c + 1e-9).log()).sum())
print(f"        硬统计熵 {ent_hard:.3f}   软使用率熵 {ent_soft:.3f}   "
      f"上限 ln{VOCAB}={torch.log(torch.tensor(float(VOCAB))):.3f}")
check("软使用率熵高于硬统计熵（软分配覆盖更多码）", ent_soft > ent_hard,
      f"{ent_soft:.3f} > {ent_hard:.3f}")

print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败   用时 {time.time()-t0:.1f}s")
print("=" * 78)
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
    raise SystemExit(1)
print("全部通过")
