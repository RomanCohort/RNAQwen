"""换底座前的可行性探查 —— 全部 CPU，不加载权重，不动 GPU。

要回答四件事
------------
1. 类解析：transformers 认不认 qwen3_5_moe / qwen3_5_moe_text，
   从 VLM 目录能不能直接拿到 **文本** 因果 LM（我们不要视觉塔）。
2. 模块名与形状：LoRA 该打在哪些模块上。40 层里 30 层是 linear_attention
   （Mamba 式 SSM，in_proj_qkv / in_proj_z / out_proj），10 层是 full_attention
   （q/k/v/o_proj）。现在 50_ 里写死的 ["q_proj","k_proj","v_proj","o_proj"]
   只命中 10 层 —— 必须量化重选。
3. 参数账：三套候选 target 的 LoRA 参数量、显存估算（bf16/8bit/4bit）。
4. tokenizer：think 块、eos/pad、chat 模板是否和 Qwen3-8B 一致。
   `Tok.no_think` 是硬编码拼出来的，对不上就会重演 §「THINK THE FIRST OF」那个坑。

只读 safetensors 头部（128 字节 JSON），不载权重。
用法：python scripts/81_probe_qwen36.py
"""
from __future__ import annotations

import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402  必须先于 transformers —— 否则 torch.distributed 报
                # ModuleNotFoundError: torch._C._distributed_c10d

MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
HID = 2048
LORA_R = 8


def st_header(p: Path) -> dict:
    """读 safetensors 头（前 8 字节 = 头长度 little-endian u64）。"""
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main() -> int:
    print("=" * 78)
    print("1  类解析")
    print("=" * 78)
    import transformers
    print(f"  transformers {transformers.__version__}")
    import transformers.models as TM
    for name in ("Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration",
                 "Qwen3_5MoeTextConfig", "Qwen3_5MoeConfig"):
        try:
            ok = hasattr(transformers, name)
        except Exception as e:
            ok = f"导入失败 {type(e).__name__}"
        print(f"  {name:<42} {ok}")
    cfg_path = MDL / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tc = cfg["text_config"]
    print(f"  顶层 model_type  {cfg['model_type']}   arch {cfg['architectures']}")
    print(f"  text  model_type {tc['model_type']}")

    from transformers import AutoConfig
    ac = AutoConfig.from_pretrained(str(MDL))
    print(f"  AutoConfig -> {type(ac).__name__}  "
          f"has text_config={hasattr(ac, 'text_config')}")
    try:
        tcfg = ac.text_config
        print(f"  ac.text_config -> {type(tcfg).__name__}  "
              f"hidden={getattr(tcfg,'hidden_size',None)}  "
              f"layers={getattr(tcfg,'num_hidden_layers',None)}  "
              f"vocab={getattr(tcfg,'vocab_size',None)}")
    except Exception as e:
        print(f"  ac.text_config 取不到: {type(e).__name__}: {e}")

    # 用 from_config 在 meta 上建结构 —— 只解析类/模块树，不分配真权重。
    # （from_pretrained + meta 会被 transformers 5.x 的防呆拦住）
    print()
    print("  用 from_config 在 meta device 上建结构")
    import torch
    from transformers import AutoModelForCausalLM
    try:
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(ac)
        n = sum(p.numel() for p in m.parameters())
        print(f"    AutoModelForCausalLM(顶层 config) -> {type(m).__name__}")
        print(f"      参数 {n/1e9:.2f} B")
        print(f"      顶层子模块 {[k for k, _ in list(m.named_children())]}")
        del m
    except Exception as e:
        print(f"    ❌ AutoModelForCausalLM: {type(e).__name__}: {str(e)[:200]}")

    try:
        from transformers import Qwen3_5MoeForConditionalGeneration as VLM
        with torch.device("meta"):
            v = VLM.from_config(ac)
        print(f"    Qwen3_5MoeForConditionalGeneration -> {type(v).__name__}")
        print(f"      顶层子模块 {[k for k, _ in list(v.named_children())]}")
        for cn, ch in v.named_children():
            print(f"        {cn}: {type(ch).__name__}  "
                  f"子模块 {[k for k, _ in list(ch.named_children())][:8]}")
        del v
    except Exception as e:
        print(f"    ❌ VLM: {type(e).__name__}: {str(e)[:200]}")

    try:
        from transformers import Qwen3_5MoeForCausalLM as TLM
        with torch.device("meta"):
            t = TLM.from_config(ac.text_config)
        n = sum(p.numel() for p in t.parameters())
        print(f"    Qwen3_5MoeForCausalLM(text_config) -> {type(t).__name__}  "
              f"参数 {n/1e9:.2f} B")
        print(f"      顶层子模块 {[k for k, _ in list(t.named_children())]}")
        print(f"      model 的子模块 {[k for k, _ in list(t.model.named_children())]}")
        del t
    except Exception as e:
        print(f"    ❌ 文本-only: {type(e).__name__}: {str(e)[:200]}")

    print()
    print("=" * 78)
    print("2  模块名与形状")
    print("=" * 78)
    idx = json.loads((MDL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wmap = idx["weight_map"]
    # 形状要真读头部
    shapes: dict[str, list[int]] = {}
    for shard in sorted({v for v in wmap.values()}):
        h = st_header(MDL / shard)
        for k, v in h.items():
            if k == "__metadata__":
                continue
            shapes[k] = v["shape"]
    print(f"  张量 {len(shapes):,} 个（从 {len(set(wmap.values()))} 个分片头部读出）")

    layer_t = Counter()
    for k in shapes:
        if ".layers." in k:
            layer_t[k.split(".layers.")[1].split(".", 1)[1]] += 1
    for name, c in sorted(layer_t.items(), key=lambda x: -x[1]):
        print(f"    {c:>4} x  {name}")

    def show(pat):
        for k in sorted(shapes):
            if pat in k:
                print(f"      {k:<72} {shapes[k]}")
    print("\n  full_attention 层（layer 3）:")
    show("layers.3.self_attn.")
    print("\n  linear_attention 层（layer 0）:")
    show("layers.0.linear_attn.")
    print("\n  MoE（layer 0）:")
    show("layers.0.mlp.")
    print("\n  嵌入 / 输出 / MTP:")
    for k in sorted(shapes):
        if "embed" in k or "lm_head" in k or "mtp" in k.lower():
            print(f"      {k:<72} {shapes[k]}")

    print()
    print("=" * 78)
    print(f"3  LoRA 参数账（r={LORA_R}, lora_alpha={LORA_R*2}）")
    print("=" * 78)
    # 每层每种模块只算一次
    def per_layer(names: list[str]) -> dict[str, int]:
        out = {}
        for n in names:
            for k, s in shapes.items():
                if k.endswith("." + n + ".weight") and ".layers.0." in k:
                    i, o = s if len(s) == 2 else (s[0], s[1])
                    out[n] = LORA_R * (i + o)
                    break
        return out

    full_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lin_names = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
    mlp_names = ["gate_proj", "up_proj", "down_proj"]

    n_full = sum(1 for t in tc["layer_types"] if t == "full_attention")
    n_lin = sum(1 for t in tc["layer_types"] if t == "linear_attention")
    pl_full = per_layer(full_names)
    pl_lin = per_layer(lin_names)
    pl_mlp = per_layer(mlp_names)
    print(f"  层数  full {n_full}   linear {n_lin}")
    print(f"  每层全注意力 LoRA  {pl_full}  合计 {sum(pl_full.values()):,}")
    print(f"  每层线性注意力 LoRA {pl_lin}  合计 {sum(pl_lin.values()):,}")
    print(f"  每层共享专家 LoRA   {pl_mlp}  合计 {sum(pl_mlp.values()):,}")
    print()

    options = {
        "A 照搬 8B（只在 full 层的 q/k/v/o）":
            n_full * sum(pl_full.values()),
        "B A + 线性层的 in_proj_qkv/in_proj_z/out_proj":
            n_full * sum(pl_full.values())
            + n_lin * sum(pl_lin[k] for k in ("in_proj_qkv", "in_proj_z", "out_proj") if k in pl_lin),
        "C B + 线性层的 in_proj_a/in_proj_b":
            n_full * sum(pl_full.values())
            + n_lin * sum(pl_lin.values()),
        "D C + 共享专家的 gate/up/down":
            n_full * sum(pl_full.values())
            + n_lin * sum(pl_lin.values())
            + (n_full + n_lin) * sum(pl_mlp.values()),
    }
    for name, n in options.items():
        print(f"  {name:<48} {n/1e6:>8.2f} M  ({100*n/35.9e9:.3f}% of 35.9B)")
    print()
    print("  注：MoE 的 mlp.experts.gate_up_proj / down_proj 无 .weight 后缀，")
    print("      是堆叠 3D nn.Parameter，不是 Linear —— 不列入候选。")

    print()
    print("=" * 78)
    print("4  显存估算（权重 + LoRA + 优化器 + 激活粗估）")
    print("=" * 78)
    for tag, bpp in (("bf16", 2), ("8-bit", 1), ("4-bit (NF4)", 0.5)):
        wgb = 35.9e9 * bpp / 1024 ** 3
        print(f"  {tag:<12} 权重 {wgb:>5.1f} GiB   +LoRA/优化器/激活 约 6~10 GiB"
              f"   = {wgb+6:.1f}~{wgb+10:.1f} GiB")
    print("  物理内存 63.6 GB。bf16 装不下（§12 已记）。")

    print()
    print("=" * 78)
    print("5  tokenizer")
    print("=" * 78)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(str(MDL))
    print(f"  class {type(tk).__name__}   vocab {tk.vocab_size}  "
          f"len {len(tk)}")
    print(f"  eos {tk.eos_token!r} {tk.eos_token_id}   "
          f"pad {tk.pad_token!r} {tk.pad_token_id}   "
          f"bos {tk.bos_token!r} {tk.bos_token_id}")
    probe = ["<|im_start|>", "<|im_end|>", "system", "user", "assistant",
             " thinking\n\n<｜end▁of▁thinking｜>\n\n", "A:", "</RNA>", "<RNA>"]
    for s in probe:
        try:
            ids = tk(s, add_special_tokens=False).input_ids
            print(f"    {s!r:<44} -> {ids}")
        except Exception as e:
            print(f"    {s!r:<44} -> ERR {type(e).__name__}")
    print()
    ct = getattr(tk, "chat_template", None)
    print(f"  chat_template: {'有，%d 字符' % len(ct) if ct else '**没有**'}")
    if ct:
        print("    前 400 字符：")
        print("    " + ct[:400].replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
