"""验证 LoRA target 在 Qwen3.6-35B 上真的挂得上（meta device，不耗内存）。

为什么必须实测
--------------
PLAN §14.4 里那四套 target 是**按模块名推出来的**，没验过 peft 认不认。
这个模型的注意力有两套命名：
    full_attention（10 层）  self_attn.q_proj / k_proj / v_proj / o_proj
    linear_attention（30 层）linear_attn.in_proj_qkv / in_proj_z / in_proj_a /
                             in_proj_b / out_proj
而且 `attn_output_gate = true` 让 q_proj 的输出从 4096 变成 8192。

peft 的 target_modules 是**按名字后缀匹配**的：
    "o_proj"  不会匹配 "out_proj"
    "q_proj"  也不会匹配 "in_proj_qkv"
所以原来 8B 那套 ["q_proj","k_proj","v_proj","o_proj"] 只会命中 10/40 层。
这一条要实测确认，不能靠读名字。

做法：在 meta 上建 VLM，套 LoraConfig，数一下被替换的模块数和 LoRA 参数量。
meta 上建只要 3 秒、不占内存，可以随便试几套配置。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import torch  # noqa: E402

MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")

# 四套候选（PLAN §14.4）
CANDIDATES = {
    "A 照搬 8B（只 full 层的 q/k/v/o）":
        ["q_proj", "k_proj", "v_proj", "o_proj"],
    "B A + 线性层 qkv/z/out":
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "in_proj_qkv", "in_proj_z", "out_proj"],
    "C B + 线性层 a/b":
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"],
    "D C + 共享专家 gate/up/down":
        ["q_proj", "k_proj", "v_proj", "o_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
         "gate_proj", "up_proj", "down_proj"],
}


def main() -> int:
    from transformers import AutoConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration as VLM)

    cfg = AutoConfig.from_pretrained(str(MDL))
    tcfg = cfg.text_config
    print("=" * 80)
    print("Qwen3.6-35B-A3B 的 LoRA target 实测（meta device）")
    print("=" * 80)

    # 先看真实的 Linear 模块名分布
    with torch.device("meta"):
        raw = VLM(cfg)
    lay = raw.model.language_model.layers
    from collections import Counter
    per = {}
    for i, L in enumerate(lay):
        for n, m in L.named_modules():
            if isinstance(m, torch.nn.Linear):
                per.setdefault(n, 0)
                per[n] += 1
    print(f"\n  text 层数 {len(lay)}")
    print("  每层里的 Linear 模块名（出现次数 = 有多少层有它）:")
    for n, c in sorted(per.items(), key=lambda x: -x[1]):
        shp = None
        for _, m in lay[0].named_modules():
            pass
        print(f"    {c:>4}/40  {n}")
    del raw
    import gc; gc.collect()

    print()
    print("=" * 80)
    print("  peft 挂载结果")
    print("=" * 80)
    from peft import LoraConfig, get_peft_model
    for tag, targets in CANDIDATES.items():
        with torch.device("meta"):
            m = VLM(cfg)
        lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=targets, task_type="CAUSAL_LM")
        try:
            pm = get_peft_model(m, lc)
        except Exception as e:
            print(f"  {tag:<36} ❌ {type(e).__name__}: {str(e)[:120]}")
            del m
            gc.collect()
            continue
        # 数被替换的模块
        hit = Counter()
        for n, mod in pm.named_modules():
            if hasattr(mod, "lora_A"):
                base = n.split(".lora_A")[0]
                short = base.split(".layers.")[-1].split(".", 1)[-1]
                hit[short] += 1
        tot = sum(hit.values())
        n_lora = sum(p.numel() for n, p in pm.named_parameters() if "lora_" in n)
        print(f"  {tag:<36} 命中 {tot:>6} 个模块  LoRA {n_lora/1e6:>7.2f} M")
        for k, v in sorted(hit.items(), key=lambda x: -x[1])[:12]:
            print(f"        {v:>4}  {k}")
        del pm, m
        gc.collect()

    print()
    print("=" * 80)
    print("  对照：8B 那套配置在 8B 上的样子（确认方法本身没写错）")
    print("=" * 80)
    p8 = ROOT / "models/Qwen3-8B"
    if p8.exists():
        from transformers import AutoModelForCausalLM
        c8 = AutoConfig.from_pretrained(str(p8))
        with torch.device("meta"):
            m8 = AutoModelForCausalLM.from_config(c8)
        lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        task_type="CAUSAL_LM")
        pm8 = get_peft_model(m8, lc)
        tot8 = sum(1 for n, mod in pm8.named_modules() if hasattr(mod, "lora_A"))
        n8 = sum(p.numel() for n, p in pm8.named_parameters() if "lora_" in n)
        print(f"  Qwen3-8B  {c8.num_hidden_layers} 层  命中 {tot8} 个模块  "
              f"LoRA {n8/1e6:.2f} M")
        print(f"  期望 {c8.num_hidden_layers * 4} 个（每层 q/k/v/o）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
