"""peft x bitsandbytes x inputs_embeds 的集成验证 —— 换 35B 前最该先验的一环。

为什么单独测这个
----------------
86_ 验的是「LoRA target 挂不挂得上」，但它是在 **meta 上的普通 nn.Linear** 上测的。
真跑 35B 时那些 Linear 是 **bnb 量化过的**（Linear8bitLt / Linear4bit），
peft 对它们走的是另一条路径（peft.tuners.lora.bnb），行为不经相同。

而且 route C 还有个额外的硬要求：**梯度必须能从输出回到 `inputs_embeds`**，
因为我们不在 tokenizer 词表里，而是直接把码向量写进 embedding 的位置。
如果这条路径断了，LoRA 再对也没用 —— 码表和 FSQ 收不到任何梯度，
训练会「跑得很正常」但什么都不学。

四件事
------
1. 小 Qwen3 模型 -> bnb 量化 -> get_peft_model，看挂载数和可训参数
2. 看 state_dict 的 key 名，确认 50_ 里 save_ckpt 的 `"lora_" in k` 过滤能命中
3. 前向走 `inputs_embeds`，反向，确认**梯度回到了输入 embedding**
4. 不调 prepare_model_for_kbit_training 会怎样（peft 是否需要它）

全程 CPU/小模型，几十 MB。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

import torch  # noqa: E402


def tiny_causal_lm(quant: str):
    """建一个小 Qwen3 因果 LM，按需把 Linear 换成 bnb 版本。"""
    from transformers import Qwen3Config, AutoModelForCausalLM
    cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    m = AutoModelForCausalLM.from_config(cfg).to("cuda" if torch.cuda.is_available()
                                                 else "cpu")
    if quant == "none":
        return m, cfg
    dev = next(m.parameters()).device
    n = 0
    for name, mod in list(m.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                w = child.weight.data.detach().to("cpu").contiguous()
                if quant == "int8":
                    from bitsandbytes.nn import Int8Params, Linear8bitLt
                    new = Linear8bitLt(child.in_features, child.out_features,
                                       bias=child.bias is not None,
                                       has_fp16_weights=False, threshold=6.0)
                    new.weight = Int8Params(w, has_fp16_weights=False,
                                            requires_grad=False)
                else:
                    from bitsandbytes.nn import Linear4bit, Params4bit
                    new = Linear4bit(child.in_features, child.out_features,
                                     bias=child.bias is not None,
                                     compute_dtype=torch.bfloat16,
                                     compress_statistics=True, quant_type="nf4")
                    new.weight = Params4bit(w, requires_grad=False,
                                            compress_statistics=True,
                                            quant_type="nf4")
                setattr(mod, cname, new.to(dev))
                n += 1
    return m, cfg, n


def run(quant: str, targets, use_prep: bool):
    from peft import LoraConfig, get_peft_model
    tag = f"quant={quant:<4} prep={int(use_prep)}"
    try:
        out = tiny_causal_lm(quant)
        if quant == "none":
            m, cfg = out
            nq = 0
        else:
            m, cfg, nq = out
        if use_prep:
            from peft import prepare_model_for_kbit_training
            m = prepare_model_for_kbit_training(m)
        for p in m.parameters():
            p.requires_grad_(False)
        lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=targets, task_type="CAUSAL_LM")
        pm = get_peft_model(m, lc)
    except Exception as e:
        print(f"  {tag}  ❌ {type(e).__name__}: {str(e)[:140]}")
        return False

    n_att = sum(1 for x, _ in pm.named_modules() if hasattr(x if False else _, "lora_A"))
    lora_keys = [k for k in pm.state_dict() if "lora_" in k]
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    flag_filter = len(lora_keys) > 0

    # ---- 前向 + 反向：梯度必须回到 inputs_embeds ----
    dev = next(pm.parameters()).device
    emb_layer = pm.get_input_embeddings()
    T = 6
    ids = torch.randint(0, cfg.vocab_size, (1, T), device=dev)
    emb = emb_layer(ids).detach().float()
    # 模拟 route C：把中间两个位置换成"码向量"，并让它 requires_grad
    v = torch.randn(2, emb.shape[-1], device=dev, dtype=emb.dtype,
                    requires_grad=True)
    inp = emb.clone()
    inp[0, 2:4] = v.to(inp.dtype)
    inp.requires_grad_(True)
    out = pm(inputs_embeds=inp.to(next(pm.parameters()).dtype))
    loss = out.logits.float().pow(2).mean()
    loss.backward()
    g_v = v.grad
    g_lora = [p.grad for p in pm.parameters() if p.requires_grad and p.grad is not None]
    ok_v = g_v is not None and torch.isfinite(g_v).all() and g_v.abs().sum() > 0
    ok_l = len(g_lora) > 0 and all(torch.isfinite(g).all() for g in g_lora)

    print(f"  {tag}  量化 {nq:>2} 个 Linear  挂载 {n_att:>2} 个 LoRA  "
          f"可训 {n_train/1e3:>6.1f}K  lora_key {len(lora_keys):>3}  "
          f"save过滤{'✓' if flag_filter else '✗'}  "
          f"输入梯度{'✓' if ok_v else '✗'}  LoRA梯度{'✓' if ok_l else '✗'}  "
          f"|g_v|max {float(g_v.abs().max()) if g_v is not None else float('nan'):.3e}")
    del pm, m
    import gc; gc.collect()
    return bool(flag_filter and ok_v and ok_l)


def main() -> int:
    import peft, bitsandbytes
    print("=" * 92)
    print(f"peft {peft.__version__}   bitsandbytes {bitsandbytes.__version__}   "
          f"torch {torch.__version__}   cuda {torch.cuda.is_available()}")
    print("=" * 92)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    bad = 0
    print("\n[1] 普通 Linear（86_ 测的就是这种）")
    bad += not run("none", targets, False)
    print("\n[2] bnb int8 量化 Linear，不调 prepare_model_for_kbit_training")
    bad += not run("int8", targets, False)
    print("\n[3] bnb int8 量化 Linear，调 prepare_model_for_kbit_training")
    bad += not run("int8", targets, True)
    print("\n[4] bnb int4 (nf4) 量化 Linear（35B 实际用的就是这个）")
    bad += not run("int4", targets, False)
    print("\n[5] bnb int4 + prepare")
    bad += not run("int4", targets, True)

    print()
    print("=" * 92)
    print("  判据：save 过滤能命中（save_ckpt 靠 'lora_' in k 找 LoRA）")
    print("        输入梯度不为零（route C 的码表/FSQ 全靠这条回传）")
    print("        LoRA 梯度不为零（LoRA 真的在训）")
    print("=" * 92)
    print("  ✅ 全部通过" if bad == 0 else f"  ❌ {bad} 项不合格")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
