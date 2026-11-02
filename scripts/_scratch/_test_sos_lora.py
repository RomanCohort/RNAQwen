"""SOS-LoRA 冒烟单测（CPU，秒级）。

为什么单独写这个：SOS 这条路在 50_ 里是手写的，没有 peft 兜底。它要在 35B 上跑，
而 35B 的装载要 523 秒 —— 拿真模型试错的代价太大。所以在这里用一棵**仿 35B 的假树**把
三个真正会出事的地方钉死：

  1. 匹配范围：35B 里只有注意力层是 nn.Linear，专家是量化过的非 nn.Linear。
     挂错（挂到专家上）或挂漏（一个都没挂上）都必须报出来。
  2. dtype：Qwen 跑 bfloat16，A/B 默认建 float32 会在前向炸
     （`expected mat1 and mat2 to have the same dtype`）。这个坑用默认 dtype 的单测查不出来。
  3. 合并：论文的核心卖点是 merge 后推理端不认 SOS。合并前后前向必须逐位接近。

跑法： python scripts/_scratch/_test_sos_lora.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.models.sos_lora import (  # noqa: E402
    SOSLoRALayer, replace_linear_with_sos, count_sos_layers,
    merge_sos_layers, freeze_except_sos, sos_trainable_params, sos_lora_penalty,
)

# 与 qwen36.lora_targets("C") 一致
PRESET_C = ("q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")

D = 32
FAILS: list[str] = []


def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


class FakeQuant(nn.Module):
    """模拟 bnb 量化后的专家：有权重，但**不是** nn.Linear。必须被跳过。"""

    def __init__(self, o, i):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(o, i), requires_grad=False)


class Attn(nn.Module):
    """全注意力层（10 层这种）：q/k/v/o 是 nn.Linear。"""

    def __init__(self, d, dt):
        super().__init__()
        self.q_proj = nn.Linear(d, d, dtype=dt)
        self.k_proj = nn.Linear(d, d, dtype=dt)
        self.v_proj = nn.Linear(d, d, dtype=dt)
        self.o_proj = nn.Linear(d, d, dtype=dt)
        self.q_norm = nn.LayerNorm(d)


class LinAttn(nn.Module):
    """线性注意力层（30 层这种）：5 个投影。"""

    def __init__(self, d, dt):
        super().__init__()
        for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
            setattr(self, n, nn.Linear(d, d, dtype=dt))


class MLP(nn.Module):
    """专家 + 路由 + 共享专家。这里一个都不该被 SOS 碰到。"""

    def __init__(self, d, dt):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.ModuleDict({"gate_up_proj": FakeQuant(2 * d, d),
                           "down_proj": FakeQuant(d, 2 * d)})
            for _ in range(4)])
        self.gate = nn.Linear(d, 2, dtype=dt)
        sh = nn.Module()
        sh.gate_proj = nn.Linear(d, d, dtype=dt)
        sh.up_proj = nn.Linear(d, d, dtype=dt)
        sh.down_proj = nn.Linear(d, d, dtype=dt)
        self.shared_expert = sh
        self.shared_expert_gate = nn.Linear(d, 1, dtype=dt)


class Layer(nn.Module):
    def __init__(self, d, dt, kind):
        super().__init__()
        self.self_attn = Attn(d, dt)
        self.linear_attn = LinAttn(d, dt)
        self.mlp = MLP(d, dt)
        self.kind = kind

    def forward(self, x):
        a = self.self_attn if self.kind == "full" else self.linear_attn
        return a.o_proj(a.q_proj(x)) if self.kind == "full" else a.out_proj(a.in_proj_qkv(x))


class Fake35B(nn.Module):
    def __init__(self, d=32, dt=torch.bfloat16, n_full=2, n_lin=3):
        super().__init__()
        layers = [Layer(d, dt, "full") for _ in range(n_full)]
        layers += [Layer(d, dt, "lin") for _ in range(n_lin)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


def count_plain_linears(m):
    return sum(1 for mod in m.modules() for c in mod.children()
               if isinstance(c, nn.Linear))


def main():
    dt = torch.bfloat16
    m = Fake35B(dt=dt, n_full=2, n_lin=3)
    # 每个 Layer 同时挂着 self_attn(4 个) 和 linear_attn(5 个)，
    # 正是 35B 的样子（全注意力层与线性注意力层在同一个 ModuleList 里）。
    expect_targets = 5 * (4 + 5)
    before_linear = count_plain_linears(m)

    print("== 1. 匹配范围 ==")
    print(f"  树里原本有 {before_linear} 个 nn.Linear（含 mlp.gate / shared_expert / shared_expert_gate）")
    rep = replace_linear_with_sos(m, PRESET_C, r_tot=8, k=4, alpha=16.0, dropout=0.0)
    check(len(rep) == expect_targets,
          f"替换 {len(rep)} 个，期望 {expect_targets}（只该是注意力投影）")
    check(count_sos_layers(m) == expect_targets,
          f"count_sos_layers = {count_sos_layers(m)}")

    # 没被碰到的必须还是 nn.Linear
    l0 = m.layers[0]
    check(isinstance(l0.mlp.gate, nn.Linear), "mlp.gate 未被替换")
    check(isinstance(l0.mlp.shared_expert.down_proj, nn.Linear),
          "mlp.shared_expert.down_proj 未被替换（out_proj 不该匹配 down_proj）")
    check(isinstance(l0.mlp.shared_expert_gate, nn.Linear), "shared_expert_gate 未被替换")
    check(isinstance(l0.mlp.experts[0]["gate_up_proj"], FakeQuant),
          "专家 gate_up_proj 未被替换（它不是 nn.Linear）")
    check(isinstance(l0.mlp.experts[0]["down_proj"], FakeQuant), "专家 down_proj 未被替换")
    check(isinstance(l0.self_attn.q_norm, nn.LayerNorm), "q_norm 未被替换")

    print("\n== 2. dtype ==")
    sos = l0.self_attn.q_proj
    check(isinstance(sos, SOSLoRALayer), "q_proj 已变成 SOSLoRALayer")
    check(all(p.dtype == dt for p in sos.A) and all(p.dtype == dt for p in sos.B),
          f"A/B dtype 跟随底座 bfloat16（实际 {sos.A[0].dtype}）")

    print("\n== 2b. device（35B 上第 1 步炸在这里）==")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        lin = nn.Linear(D, D, dtype=dt).to(dev)
        s2 = SOSLoRALayer(lin, r_tot=8, k=4, alpha=16.0, dropout=0.0)
        # 基准必须取底座自己的 device：torch.device("cuda") == torch.device("cuda:0") 是 False，
        # 拿 "cuda" 去比会误报（这个比法本身踩过一次）。
        ref = lin.weight.device
        check(all(p.device == ref for p in list(s2.A) + list(s2.B)),
              f"A/B 跟随底座设备（底座 {ref}，A[0] {s2.A[0].device}）")
        try:
            y2 = s2(torch.randn(2, D, dtype=dt, device=ref))
            check(torch.isfinite(y2.float()).all(), "cuda 上前向正常")
        except RuntimeError as e:
            check(False, f"cuda 上前向炸了：{e}")
    else:
        print("  [skip] 没有可用 GPU —— 这条正是 CPU 单测抓不到 device 坑的原因")

    print("\n== 3. 前向 / 反向 ==")
    m.train()
    x = torch.randn(2, 4, D, dtype=dt, requires_grad=True)
    y = m(x)
    check(y.dtype == dt and torch.isfinite(y.float()).all(), f"前向输出 {tuple(y.shape)} {y.dtype}")
    loss = (y.float() ** 2).mean()
    loss.backward()
    n_grad = sum(1 for p in sos.parameters() if p.requires_grad and p.grad is not None
                 and p.grad.abs().sum() > 0)
    # 标准 LoRA 初始化：A 随机、B 置零，所以首步 grad_A = Bᵀ·上游 = 0，
    # 只有 B 那 k 个拿到梯度。这不是 bug，是 LoRA 的定义。等 B 非零后 A 才动。
    check(n_grad == sos.k, f"首步有梯度的参数 {n_grad} 个（LoRA 初始化下应恰为 k={sos.k}）")
    check(sos.base.weight.grad is None, "底座 weight 无梯度（真的冻住了）")

    # 把 B 推离零，再走一遍：这次 A 必须也拿到梯度
    with torch.no_grad():
        for b in sos.B:
            b.copy_(torch.randn_like(b) * 0.1)
    m.zero_grad(set_to_none=True)
    (m(x.detach()) ** 2).float().mean().backward()
    n_grad2 = sum(1 for p in sos.parameters() if p.requires_grad and p.grad is not None
                  and p.grad.abs().sum() > 0)
    check(n_grad2 == 2 * sos.k,
          f"B 非零后有梯度的参数 {n_grad2} 个，期望 {2 * sos.k}（A 和 B 都动）")

    print("\n== 4. 参数量与 LoRA 同预算 ==")
    got = sum(p.numel() for p in sos.parameters() if p.requires_grad)
    want = (sos.in_features + sos.out_features) * sos.r_tot
    check(got == want, f"SOS 可训 {got}，等价 LoRA(rank={sos.r_tot}) {want}")

    m.zero_grad(set_to_none=True)
    pen = sos_lora_penalty(m)
    check(torch.isfinite(pen), f"正交惩罚有限 {float(pen.detach()):.6f}")
    n_frz = freeze_except_sos(m)
    check(n_frz > 0, f"freeze_except_sos 冻了 {n_frz} 个张量")
    keep = {id(p) for mod in m.modules() if isinstance(mod, SOSLoRALayer)
            for p in list(mod.A) + list(mod.B)}
    check(all(p.requires_grad for p in m.parameters() if id(p) in keep),
          "SOS 的 A/B 仍然可训")
    check(not any(p.requires_grad for p in m.parameters() if id(p) not in keep),
          "除 A/B 外全部冻结")
    check(sos_trainable_params(m) == got * expect_targets,
          f"sos_trainable_params = {sos_trainable_params(m)}")

    print("\n== 5. 合并（论文核心卖点）==")
    m.eval()
    with torch.no_grad():
        y_before = m(x.detach()).float().clone()
        n_merged = merge_sos_layers(m)
    check(n_merged == expect_targets, f"合并 {n_merged} 层")
    check(count_sos_layers(m) == 0, "合并后不再有 SOSLoRALayer")
    check(isinstance(m.layers[0].self_attn.q_proj, nn.Linear),
          "合并后 q_proj 还原成 nn.Linear")
    with torch.no_grad():
        y_after = m(x.detach()).float()
    dmax = (y_before - y_after).abs().max().item()
    check(dmax < 1e-2, f"合并前后前向最大差 {dmax:.3e}（bf16 容差 1e-2）")

    print()
    if FAILS:
        print(f"===== {len(FAILS)} 项失败 =====")
        for f in FAILS:
            print("  - " + f)
        return 1
    print("===== 全部通过 =====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
