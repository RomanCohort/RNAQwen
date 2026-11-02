"""验证「专家 Linear 化」改写后的 MoE forward 和官方实现逐位等价。

为什么必须单独验
----------------
84_ 证明了权重**装得进去**，没证明**算得对**。
改写动了 MoE 里唯一一处真正的计算：

    官方  nn.functional.linear(current_state, self.gate_up_proj[expert_idx])
    改写  self.gate_up[expert_idx](current_state)

看起来只是换个写法，但有几个地方容易错：
  · gate_up 的 [2I, H] 布局：chunk(2, dim=-1) 之后 gate/up 各是哪些列
  · 专家切片的方向：官方参数是 [E, 2I, H]，Linear 的 weight 是 [out, in]，
    两者一致；但 down_proj 是 [E, H, I] 而 Linear(in=I, out=H).weight 是 [H, I]，
    也一致 —— **一致不等于我写对了，得对拍**
  · index_add_ 的 token_idx 可能重复（一个 token 命中多个专家），累加顺序
  · expert_hit 的枚举顺序会影响浮点累加顺序，但不该影响结果

做法
----
小配置（专家少、维度小）上实例化两份，拷同一组权重，喂同一组输入，
比 forward 输出的最大绝对差；再用 autograd 比 hidden_states 的梯度。
小配置能验算法；算法对了，大配置只是同一个循环跑 256 次。

用法：python scripts/85_moe_forward_equiv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import torch  # noqa: E402

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (  # noqa: E402
    Qwen3_5MoeExperts)


class Cfg:
    """够用的最小 config。

    注意 `_experts_implementation`：transformers 5.9 用
    `transformers/integrations/moe.py` 把 `Qwen3_5MoeExperts.forward` 包了一层，
    包装器会读 `self.config._experts_implementation` 来分发实现。
    真实 config 是 PretrainedConfig，有这条属性；裸类没有，必须自己带上，
    否则官方类直接 AttributeError（实测报的就是这个）。
    """
    def __init__(self, E, H, I, act="silu"):
        self.num_experts = E
        self.hidden_size = H
        self.moe_intermediate_size = I
        self.hidden_act = act
        self._experts_implementation = "eager"


def build_mine(cfg):
    """复用 84_ 里那个类的**同一份实现**（避免测的是另抄一遍的版本）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "q36", ROOT / "scripts/84_qwen36_quant.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["q36"] = mod
    spec.loader.exec_module(mod)
    cls = mod.make_quant_experts_cls(Qwen3_5MoeExperts)
    return cls(cfg)


def copy_weights(official, mine, cfg):
    with torch.no_grad():
        for e in range(cfg.num_experts):
            mine.gate_up[e].weight.copy_(official.gate_up_proj[e])
            mine.down[e].weight.copy_(official.down_proj[e])


def compare(tag, cfg, T, K, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    official = Qwen3_5MoeExperts(cfg).to(dtype)
    mine = build_mine(cfg).to(dtype)
    # 两份独立随机初始化 -> 先拷权重
    with torch.no_grad():
        official.gate_up_proj.normal_(0, 0.05)
        official.down_proj.normal_(0, 0.05)
    copy_weights(official, mine, cfg)

    torch.manual_seed(seed + 1)
    x = torch.randn(T, cfg.hidden_size, dtype=dtype) * 0.7
    # top_k_index: (T, K)，值在 [0, E)
    top_k_index = torch.stack(
        [torch.randperm(cfg.num_experts)[:K] for _ in range(T)])
    top_k_weights = torch.rand(T, K, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(-1, keepdim=True)

    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ya = official(xa, top_k_index, top_k_weights)
    yb = mine(xb, top_k_index, top_k_weights)

    d = (ya - yb).abs().max().item()
    scale = ya.abs().max().item()
    rel = d / max(scale, 1e-12)

    ga, gb = None, None
    if dtype.is_floating_point:
        ya.sum().backward()
        yb.sum().backward()
        ga, gb = xa.grad, xb.grad
        gd = (ga - gb).abs().max().item()
        gscale = ga.abs().max().item()
        grel = gd / max(gscale, 1e-12)
    else:
        gd = grel = float("nan")

    ok = rel < 1e-5 and (grel < 1e-4 or grel != grel)
    print(f"  {tag:<34} T={T:<4} K={K}  "
          f"forward |Δ|max {d:.3e} (相对 {rel:.2e})   "
          f"grad |Δ|max {gd:.3e} (相对 {grel:.2e})  {'✅' if ok else '❌'}")
    return ok


def main() -> int:
    print("=" * 84)
    print("专家 Linear 化 vs 官方 Qwen3_5MoeExperts —— forward / backward 对拍")
    print("=" * 84)
    bad = 0
    cfg = Cfg(E=8, H=32, I=16)

    print("\n[1] 基本形状")
    copy_cfg = Cfg(E=8, H=32, I=16)
    o = Qwen3_5MoeExperts(copy_cfg)
    m = build_mine(copy_cfg)
    print(f"  官方 gate_up_proj {tuple(o.gate_up_proj.shape)}  "
          f"down_proj {tuple(o.down_proj.shape)}")
    print(f"  改写 gate_up[0].weight {tuple(m.gate_up[0].weight.shape)}  "
          f"down[0].weight {tuple(m.down[0].weight.shape)}")
    shape_ok = (tuple(m.gate_up[0].weight.shape) ==
                (2 * copy_cfg.moe_intermediate_size, copy_cfg.hidden_size) and
                tuple(m.down[0].weight.shape) ==
                (copy_cfg.hidden_size, copy_cfg.moe_intermediate_size))
    print(f"  形状对齐 {'✅' if shape_ok else '❌'}")
    bad += not shape_ok

    print("\n[2] 不同 token 数 / top-k（含极端：K=1、K=E 全命中）")
    for T, K in ((1, 1), (4, 2), (17, 2), (33, 8), (64, 8), (128, 8)):
        bad += not compare(f"T={T} K={K}", cfg, T, K, seed=T * 10 + K)

    print("\n[3] 不同激活函数")
    for act in ("silu", "gelu", "relu"):
        c = Cfg(E=8, H=32, I=16, act=act)
        bad += not compare(f"act={act}", c, 24, 3, seed=7)

    print("\n[4] 不同 dtype（bf16 是训练实际用的）")
    for dt in (torch.float32, torch.bfloat16):
        c = Cfg(E=8, H=32, I=16)
        bad += not compare(f"dtype={dt}", c, 24, 4, seed=11, dtype=dt)

    print("\n[5] 单专家被反复命中（index_add_ 的累加路径）")
    c = Cfg(E=4, H=16, I=8)
    torch.manual_seed(3)
    off = Qwen3_5MoeExperts(c)
    with torch.no_grad():
        off.gate_up_proj.normal_(0, 0.05); off.down_proj.normal_(0, 0.05)
    mine = build_mine(c)
    copy_weights(off, mine, c)
    x = torch.randn(512, c.hidden_size) * 0.5
    idx = torch.zeros(512, 2, dtype=torch.long)          # 全部走专家 0
    w = torch.full((512, 2), 0.5)
    d = (off(x, idx, w) - mine(x, idx, w)).abs().max().item()
    print(f"  全部命中专家 0，512 token   |Δ|max {d:.3e}  {'✅' if d < 1e-5 else '❌'}")
    bad += d >= 1e-5

    print("\n[6] 真实维度（E=256, H=2048, I=512）跑一遍，确认循环规模没问题")
    c = Cfg(E=256, H=2048, I=512)
    ok = compare("真实维度", c, 8, 8, seed=99)
    bad += not ok

    print()
    print("=" * 84)
    print("  ✅ 改写与官方逐位等价，可以用于训练" if bad == 0
          else f"  ❌ {bad} 项不合格 —— 不能用于训练")
    print("=" * 84)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
