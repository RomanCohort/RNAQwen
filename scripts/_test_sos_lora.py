"""SOS-LoRA 实现的正确性测试。

测的不是"能跑",是**论文声称的每一条性质是否真的成立**：
  1. 同预算：r_tot 相同时参数量与标准 LoRA 一致（论文说"matching the parameter budget"）
  2. 零初始更新：B 零初始化 => 初始 forward == 原层 forward（论文"zero initial update"）
  3. 尺度：s_k = (α/r_tot)·γ_k，γ_k 在 [1,2.5] 上线性等距
  4. 正交初始化：不同专家的 A 方向两两正交
  5. 合并等价：ΔW = Σ s_k A_k B_k 并进 W₀ 后，输出与未合并前一致
  6. 梯度分档：每个专家的梯度幅度 ∝ s_k（这是"scale-separated dynamics"的机制本身）
  7. 正则：同专家内部不罚、跨专家才罚
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rnaqwen.models.sos_lora import (  # noqa: E402
    SOSLoRALayer, fixed_scales, replace_linear_with_sos, sos_lora_penalty,
)

torch.manual_seed(0)
fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("  [%s] %-46s %s" % ("ok  " if ok else "FAIL", name, detail))
    if not ok:
        fails.append(name)


IN, OUT, R_TOT, K, ALPHA = 64, 128, 8, 4, 16.0

print("=" * 84)
print("1) 同预算：SOS-LoRA 与标准 LoRA 参数量一致")
print("=" * 84)
base = nn.Linear(IN, OUT, bias=False)
sos = SOSLoRALayer(base, r_tot=R_TOT, k=K, alpha=ALPHA)
n_sos = sos.n_trainable()
# 标准 LoRA @ r_tot：A(r,in) + B(out,r)
n_lora = R_TOT * IN + OUT * R_TOT
check("参数量一致", n_sos == n_lora, f"SOS {n_sos:,}  LoRA {n_lora:,}")
print("  ", sos.describe())

print()
print("=" * 84)
print("2) 零初始更新：初始时 SOS 层输出 == 原层输出")
print("=" * 84)
x = torch.randn(3, IN)
with torch.no_grad():
    d = (sos(x) - base(x)).abs().max().item()
check("初始 ΔW = 0", d == 0.0, f"max|Δ| = {d:.3e}")

print()
print("=" * 84)
print("3) 尺度：s_k = (α/r_tot)·γ_k，γ_k 在 [1, γ_max] 线性等距")
print("=" * 84)
expected = [(ALPHA / R_TOT) * (1.0 + (2.5 - 1.0) / (K - 1) * i) for i in range(K)]
check("s_k 数值", all(abs(a - b) < 1e-9 for a, b in zip(sos.scales, expected)),
      f"{['%.3f' % s for s in sos.scales]} vs {['%.3f' % s for s in expected]}")
check("K=1 退化为 α/r", fixed_scales(8, 1, 16.0) == [16.0 / 8], str(fixed_scales(8, 1, 16.0)))
check("γ 最小 = 1（即 s_min = α/r_tot）", abs(sos.scales[0] - ALPHA / R_TOT) < 1e-9)
check("γ 最大 = 2.5", abs(sos.scales[-1] / (ALPHA / R_TOT) - 2.5) < 1e-9,
      f"比值 {sos.scales[-1] / (ALPHA / R_TOT):.4f}")

print()
print("=" * 84)
print("4) 正交初始化：不同专家的 A 方向两两正交")
print("=" * 84)
worst = 0.0
for i in range(K):
    for j in range(i + 1, K):
        a, b = sos.A[i].reshape(-1), sos.A[j].reshape(-1)
        cos = abs(float(a @ b) / (float(a.norm()) * float(b.norm()) + 1e-9))
        worst = max(worst, cos)
check("跨专家 |cos| ≈ 0", worst < 1e-4, f"最大 |cos| = {worst:.3e}")

print()
print("=" * 84)
print("5) 合并等价：ΔW = Σ s_k A_k B_k")
print("=" * 84)
# ⚠️ 必须先 eval()：训练模式下 self.dropout 是开的，sos(x) 用的是被 dropout 过的 x，
#    而 y_merged = x @ w_merged 用的是完整 x —— 两者本来就该不一样。
#    第一版测试漏了这一步，报了个假的 FAIL。
sos.eval()
with torch.no_grad():
    for k in range(K):                      # 给 B 一些非零值，否则合并测不出东西
        sos.B[k].normal_(0, 0.02)
w_merged = sos.merged_weight()
with torch.no_grad():
    y_sos = sos(x)
    y_merged = x @ w_merged.t()
d = (y_sos - y_merged).abs().max().item()
check("合并后输出一致", d < 1e-5, f"max|Δ| = {d:.3e}")

print()
print("=" * 84)
print("6) 梯度分档：每个专家收到的梯度幅度 ∝ s_k")
print("=" * 84)
# ⚠️ ∂L/∂A_k = s_k·Xᵀ(Δ B_kᵀ) —— 梯度**还取决于 B_k 的数值**。
#    要单独测出 s_k 这个乘子，必须让所有专家的 B_k 完全相同，
#    否则测到的是 s_k 与随机 B 的混合（第一版就踩了这个）。
with torch.no_grad():
    b_shared = torch.randn(OUT, sos.r) * 0.02
    for k in range(K):
        sos.B[k].copy_(b_shared)
for p in sos.parameters():
    p.grad = None
loss = sos(x).sum()
loss.backward()
gmags = [sos.A[k].grad.norm().item() for k in range(K)]
ratios = [g / max(gmags[0], 1e-12) for g in gmags]
scale_ratios = [s / sos.scales[0] for s in sos.scales]
check("梯度比 == 尺度比", all(abs(a - b) < 1e-6 for a, b in zip(ratios, scale_ratios)),
      f"梯度比 {['%.4f' % r for r in ratios]}  尺度比 {['%.4f' % r for r in scale_ratios]}")

print()
print("=" * 84)
print("7) 正则：跨专家才罚，同专家内部不罚")
print("=" * 84)
# 把两个专家的 A 设成同一个方向 -> 惩罚必须 > 0
with torch.no_grad():
    sos.A[1].copy_(sos.A[0])
pen_same = float(sos.ortho_penalty())
check("两个专家同方向 -> 惩罚 > 0", pen_same > 0.1, f"penalty = {pen_same:.4f}")

# 重置成正交 -> 惩罚应接近 0
sos.reset_parameters(ortho_init=True)
pen_ortho = float(sos.ortho_penalty())
check("正交时惩罚 ≈ 0", pen_ortho < 1e-6, f"penalty = {pen_ortho:.3e}")

# 单个专家的 A 内部：把同一专家的两行设成相同，惩罚不应变
sos2 = SOSLoRALayer(nn.Linear(IN, OUT, bias=False), r_tot=R_TOT, k=K, alpha=ALPHA)
with torch.no_grad():
    sos2.A[0][1].copy_(sos2.A[0][0])
pen_inner = float(sos2.ortho_penalty())
check("同专家内部重复 -> 惩罚不变（不罚内部）", abs(pen_inner - pen_ortho) < 1e-6,
      f"{pen_inner:.3e} vs {pen_ortho:.3e}")

print()
print("=" * 84)
print("8) 替换工具：确实替换到了目标层，且名字可核对")
print("=" * 84)
class Blk(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(IN, OUT, bias=False)
        self.k_proj = nn.Linear(IN, OUT, bias=False)
        self.mlp = nn.Linear(IN, OUT, bias=False)
        self.unrelated = nn.Linear(IN, OUT, bias=False)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = Blk()
        self.layer1 = Blk()

m = M()
rep = replace_linear_with_sos(m, ("q_proj", "k_proj"), R_TOT, K, ALPHA)
check("替换到 4 个层", len(rep) == 4, str(rep))
check("是 SOSLoRALayer", isinstance(m.layer0.q_proj, SOSLoRALayer))
check("mlp 未被替换", isinstance(m.layer0.mlp, nn.Linear))
check("unrelated 未被替换", isinstance(m.layer0.unrelated, nn.Linear))
check("全局正则聚合可用", isinstance(sos_lora_penalty(m), torch.Tensor),
      f"penalty = {float(sos_lora_penalty(m)):.4f}")

print()
print("=" * 84)
print("9) dtype 跟随底座（真模型跑 bfloat16，A/B 建错 dtype 会直接崩）")
print("=" * 84)
# 踩过的坑：A/B 默认 float32，Qwen 跑 bfloat16 ->
#   "expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float"
# 单测当时用的是默认 dtype，所以没测出来。这里把它测死。
for dt in (torch.float32, torch.bfloat16, torch.float16):
    b = nn.Linear(IN, OUT, bias=False).to(dt)
    s = SOSLoRALayer(b, r_tot=R_TOT, k=K, alpha=ALPHA)
    xd = torch.randn(2, IN).to(dt)
    try:
        y = s(xd)
        ok = (y.dtype == dt) and all(p.dtype == dt for p in list(s.A) + list(s.B))
        check(f"{str(dt):<16} 前向不崩且 dtype 一致", ok, f"out={y.dtype}")
    except RuntimeError as e:
        check(f"{str(dt):<16} 前向不崩且 dtype 一致", False, str(e)[:70])

print()
print("=" * 84)
if fails:
    print("VERDICT: FAIL")
    for f in fails:
        print("   ✗", f)
else:
    print("VERDICT: PASS —— 论文声称的 8 条性质全部成立")
print("=" * 84)
raise SystemExit(1 if fails else 0)
