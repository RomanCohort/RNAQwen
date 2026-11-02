"""极小规模排练 50_ 的 35B 路径 —— 最后一个没验过的环节。

为什么还差这一块
----------------
88_ 验通了 peft x bnb x inputs_embeds，但用的是普通 `Qwen3ForCausalLM`。
35B 真实用的是 **VLM 类** `Qwen3_5MoeForConditionalGeneration`（只有它认识
`model.language_model.*` 这个 key 前缀），peft 包 VLM 走的是另一条路。

这里把 config 的维度缩到极小（hidden 64、2 层、8 专家），
**用同一个类、同一套专家 Linear 化改写、同一套 LoRA target**，
在 CPU/小显存上把 50_ 的路径整条走一遍：

    建 VLM -> 专家 Linear 化 -> 前向 inputs_embeds -> 反向 -> 收梯度

维度缩小只影响规模，不影响接线。接线错了，这里就会错。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import torch  # noqa: E402


def tiny_vlm_cfg(n_layers=4, n_experts=8, hidden=64):
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    lt = ["linear_attention", "full_attention"] * (n_layers // 2)
    tc = Qwen3_5MoeTextConfig(
        vocab_size=512, hidden_size=hidden, intermediate_size=128,
        num_hidden_layers=n_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128,
        num_experts=n_experts, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        layer_types=lt[:n_layers], full_attention_interval=2,
        linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_value_head_dim=16, attn_output_gate=True,
        tie_word_embeddings=False,
    )
    cfg = Qwen3_5MoeConfig(text_config=tc)
    cfg.vision_config.depth = 1
    cfg.vision_config.hidden_size = 32
    cfg.vision_config.intermediate_size = 32
    cfg.vision_config.num_heads = 4
    cfg.vision_config.out_hidden_size = hidden
    return cfg


def main() -> int:
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M
    from peft import LoraConfig, get_peft_model
    import torch.nn.functional as F

    print("=" * 88)
    print("极小 Qwen3.5-MoE VLM 上排练 50_ 的 35B 路径")
    print("=" * 88)

    # ---- 0. 先验 conv1d 替换的数值等价性（在 CPU 上，因为 CUDA 上原版会崩）----
    from rnaqwen.qwen36 import patch_conv1d
    Cv = patch_conv1d(M)
    torch.manual_seed(0)
    C, K, L = 6, 4, 11
    ref = torch.nn.Conv1d(C, C, kernel_size=K, groups=C, padding=K - 1, bias=False)
    mine = Cv(ref.weight, None)
    x = torch.randn(1, C, L)
    with torch.no_grad():
        a = ref(x)[:, :, :L]
        b = mine(x)
    d = float((a - b).abs().max())
    print(f"[conv] 替换 vs nn.Conv1d  最大差 {d:.3e}  "
          f"{'✅ 数值等价' if d < 1e-6 else '❌ 不等价'}")
    if d >= 1e-6:
        print("      ** conv 替换不对，先修这个再往下 **")
        return 1

    cfg = tiny_vlm_cfg()
    tc = cfg.text_config
    print(f"[cfg] 层 {tc.num_hidden_layers}  层型 {tc.layer_types}  专家 {tc.num_experts}")

    # ---- 1. 专家 Linear 化（和 qwen36.py 用的是同一个改写函数）----
    from rnaqwen.qwen36 import patch_moe_experts, lora_targets, swap_linears
    patch_moe_experts(M)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 故意在**真实设备**上建（不是 meta）—— 这正是能暴露 _init_weights 那条 AttributeError 的路径
    vlm = M.Qwen3_5MoeForConditionalGeneration(cfg).to(dev)
    text = vlm.model.language_model
    print(f"[build] {type(vlm).__name__}  "
          f"{sum(p.numel() for p in vlm.parameters())/1e6:.2f} M 参数")

    ne = sum(1 for m in vlm.modules() if hasattr(m, "gate_up"))
    print(f"[experts] 换成 per-expert Linear 的层数 {ne}")

    # ---- 2. bnb int4 量化专家（真跑时的配置）----
    nq = 0
    for i in range(len(text.layers)):
        nq += swap_linears(text.layers[i], "int4",
                           f"model.language_model.layers.{i}")
        text.layers[i].to(dev)
    print(f"[quant] 量化 {nq} 个专家 Linear")

    # ---- 3. peft，用 35B 那套 target ----
    for p in vlm.parameters():
        p.requires_grad_(False)
    targets = lora_targets("C")
    pm = get_peft_model(vlm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                        target_modules=targets,
                                        task_type="CAUSAL_LM"))
    n_lora_mod = sum(1 for _, x in pm.named_modules() if hasattr(x, "lora_A"))
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"[lora] target {len(targets)} 项  挂载 {n_lora_mod} 个  "
          f"可训 {n_train/1e3:.1f} K")

    # ---- 4. 前向走 inputs_embeds（route C 的注入方式），反向 ----
    emb_layer = pm.get_input_embeddings()
    T = 8
    ids = torch.randint(0, tc.vocab_size, (1, T), device=dev)
    base = emb_layer(ids).detach()
    v = torch.randn(3, base.shape[-1], device=dev, dtype=base.dtype,
                    requires_grad=True)
    inp = base.clone()
    inp[0, 2:5] = v.to(inp.dtype)
    inp.requires_grad_(True)
    out = pm(inputs_embeds=inp)
    logits = out.logits.float()
    print(f"[fwd ] logits {tuple(logits.shape)}  有限 "
          f"{bool(torch.isfinite(logits).all())}")
    logits.pow(2).mean().backward()
    g_v = v.grad
    g_lora = [p for p in pm.parameters() if p.requires_grad and p.grad is not None]
    lora_keys = [k for k in pm.state_dict() if "lora_" in k]
    ok_v = g_v is not None and bool(torch.isfinite(g_v).all()) and \
        float(g_v.abs().sum()) > 0
    ok_l = len(g_lora) > 0 and all(bool(torch.isfinite(p.grad).all()) for p in g_lora)

    print(f"[bwd ] 输入梯度 {'✓' if ok_v else '✗'} "
          f"|g_v|max {float(g_v.abs().max()) if g_v is not None else float('nan'):.3e}"
          f"   LoRA 梯度 {'✓' if ok_l else '✗'}（{len(g_lora)} 个张量）")
    print(f"[save] state_dict 里含 'lora_' 的 key {len(lora_keys)} 个  "
          f"save_ckpt 过滤{'✓' if lora_keys else '✗'}")
    print(f"       {lora_keys[:3]}")

    bad = 0
    for tag, ok in (("inputs_embeds 前向", bool(torch.isfinite(logits).all())),
                    ("输入梯度回传", ok_v), ("LoRA 梯度", ok_l),
                    ("save 过滤", bool(lora_keys))):
        if not ok:
            print(f"  ** {tag} 失败")
            bad += 1
    print()
    print("=" * 88)
    print("  ✅ 35B 的接线在极小模型上整条走通" if bad == 0
          else f"  ❌ {bad} 项失败 —— 别启动 35B")
    print("=" * 88)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
