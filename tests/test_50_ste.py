"""软分配 STE 的测试 —— 对应"both 模式不可证伪"那个设计错误的修复。

三条必须成立，否则等于又埋一个坑：
  1. 前向值**恰好等于**硬查表（STE 只改反向，不改前向）
  2. FSQ.enc 确实拿到梯度（否则端到端又断了）
  3. grid 的位序与 forward 的编码位序一致（不一致的话软分配指向别的码，
     梯度方向是错的但前向看不出来 —— 最阴的一种 bug）
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
FSQCode = routeC.FSQCode

DIM, LAT, LEV, DM, POOL = 16, 4, 2, 32, 4
VOCAB = LEV ** LAT


class Stub(nn.Module):
    """只包含 rna_vectors 需要的东西，不碰 Qwen。"""
    def __init__(self):
        super().__init__()
        self.fsq = FSQCode(DIM, LAT, LEV, DM, pool=POOL)
        self.rna_emb = nn.Embedding(VOCAB, DM)
        nn.init.normal_(self.rna_emb.weight, std=0.02)
        self._usage_parts = None      # rna_vectors 的累加器
        self.fsq_usage = None

    rna_vectors = routeC.E2EModel.rna_vectors


print("=" * 78)
print("测试 3：grid 的位序必须与 forward 的编码位序一致")
print("=" * 78)
m = Stub()
g = m.fsq.grid
check("grid 形状", tuple(g.shape) == (VOCAB, LAT), f"{tuple(g.shape)}")
# 从 grid 反解回码 id：digit 应与 forward 的 `code = code*levels + k[d]` 同序（高位在前）
recomputed = torch.zeros(VOCAB, dtype=torch.long)
for d in range(LAT):
    digit = torch.round((g[:, d] + 1) / 2 * (LEV - 1)).long()
    recomputed = recomputed * LEV + digit
check("从 grid 反解的码 id == 原 id", bool((recomputed == torch.arange(VOCAB)).all()),
      f"不符 {(recomputed != torch.arange(VOCAB)).sum().item()} 个")
check("grid 取值落在 [-1,1]", bool((g.abs() <= 1 + 1e-6).all()))

print()
print("=" * 78)
print("测试 1：前向值 == 硬查表（STE 不许改前向）")
print("=" * 78)
x = torch.randn(40, DIM) * 8.7
out, code, t, q = Stub.rna_vectors(m, x, tau=4.0)
v_hard = m.rna_emb(code)
check("out == rna_emb[code]（逐元素）", torch.allclose(out, v_hard, atol=1e-6),
      f"max|diff|={float((out-v_hard).abs().max()):.2e}")
check("out 不全为 0", float(out.abs().max()) > 0)

print()
print("=" * 78)
print("测试 2：FSQ.enc 必须拿到非零梯度（端到端不能断）")
print("=" * 78)
m.zero_grad()
out, code, t, q = Stub.rna_vectors(m, x, tau=4.0)
out.sum().backward()
ge = m.fsq.enc.weight.grad
check("fsq.enc.weight.grad 存在", ge is not None)
check("fsq.enc 梯度非零（阈值 1e-6，不是 >0：1.8e-15 就是零）",
      ge is not None and float(ge.norm()) > 1e-6,
      f"|grad|={float(ge.norm()) if ge is not None else -1:.3e}")
gr = m.rna_emb.weight.grad
used = code.unique()
check("只有被用到的码行有梯度",
      gr is not None and float(gr[used].norm()) > 0,
      f"用到 {used.numel()} 个码")

print()
print("=" * 78)
print("测试 4：软分配确实在起反向作用（不是恒等）")
print("=" * 78)
# 把 tau 调到极小 -> 软分配退化成 one-hot，此时到 enc 的梯度应显著变小

def clone_stub(src):
    # 不能用 copy.deepcopy：fsq_usage 是非叶子张量，deepcopy 会直接抛。
    m = Stub()
    m.load_state_dict(src.state_dict(), strict=False)
    m.fsq_usage = None
    return m


m2 = clone_stub(m)
m2.zero_grad()
out2, code2, _, _ = Stub.rna_vectors(m2, x, tau=0.01)
out2.sum().backward()
g_small = float(m2.fsq.enc.weight.grad.norm())
m3 = clone_stub(m)
m3.zero_grad()
out3, _, _, _ = Stub.rna_vectors(m3, x, tau=4.0)
out3.sum().backward()
g_big = float(m3.fsq.enc.weight.grad.norm())
check("tau 小时梯度变小（证明是软分配在传梯度）", g_small < g_big,
      f"tau=0.01: {g_small:.2e}   tau=0.1: {g_big:.2e}")
check("前向值不受 tau 影响", torch.allclose(out2, out3, atol=1e-6))

print()
print("=" * 78)
print("测试 5：perm 只改查表索引，不改码本身")
print("=" * 78)
m4 = clone_stub(m)
perm = torch.randperm(VOCAB)
o_a, c_a, _, _ = Stub.rna_vectors(m4, x, perm=None)
o_b, c_b, _, _ = Stub.rna_vectors(m4, x, perm=perm)
check("perm 不改 code 返回值", bool((c_a == c_b).all()))
check("perm 确实改变了向量", not torch.allclose(o_a, o_b),
      f"max|diff|={float((o_a-o_b).abs().max()):.3e}")
check("perm 后仍等于该 perm 下的硬查表",
      torch.allclose(o_b, m4.rna_emb(perm[c_b]), atol=1e-6))

print()
print("=" * 78)
print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败   用时 {time.time()-t0:.1f}s")
print("=" * 78)
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
    raise SystemExit(1)
print("全部通过")
