"""算清楚 Qwen3-8B 上不同 LoRA 配法的可训参数，用来定 Stage 2 的方案。

Qwen3-8B 结构（已核对本地 config.json）：
    hidden_size = 4096      num_hidden_layers = 36
    intermediate_size = 12288
    num_attention_heads = 32, num_key_value_heads = 8, head_dim = 128
    → q_proj: 4096 x 4096 ; k_proj/v_proj: 4096 x 1024 ; o_proj: 4096 x 4096
    → gate/up: 4096 x 12288 ; down: 12288 x 4096
"""

from __future__ import annotations

H = 4096
L = 36
I = 12288
KV = 8 * 128   # 1024

# (name, in_features, out_features)
SHAPES = {
    "q_proj": (H, H),
    "k_proj": (H, KV),
    "v_proj": (H, KV),
    "o_proj": (H, H),
    "gate_proj": (H, I),
    "up_proj": (H, I),
    "down_proj": (I, H),
}

GROUPS = {
    "attn_only": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn+mlp（全量 target）": list(SHAPES),
}


def lora_params(targets: list[str], r: int, layers: int = L) -> int:
    per_layer = sum(r * (SHAPES[t][0] + SHAPES[t][1]) for t in targets)
    return per_layer * layers


def frozen_qwen3_8b_params() -> tuple[int, int]:
    """返回 (线性层参数, 含 embedding 的总参数，近似)。"""
    lin = 0
    for name, (a, b) in SHAPES.items():
        lin += a * b * L
    emb = 151936 * H * 2          # token embedding + lm_head（tie=False）
    return lin, lin + emb


def main() -> None:
    lin, total = frozen_qwen3_8b_params()
    print(f"Qwen3-8B 线性层参数 ≈ {lin/1e9:.2f} B；含 embedding ≈ {total/1e9:.2f} B")
    print(f"单层线性参数 ≈ {lin/L/1e6:.1f} M\n")

    print(f"{'配置':38s} {'层数':>5s} {'r':>4s} {'可训参数':>12s} {'占比':>8s}")
    print("-" * 74)
    rows = [
        ("attention 4 投影", "attn_only", 8, L),
        ("attention 4 投影", "attn_only", 16, L),
        ("attention 4 投影", "attn_only", 32, L),
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 8, L),
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 16, L),
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 32, L),
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 16, 12),   # 只调顶部 12 层
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 16, 6),    # 只调顶部 6 层
        ("attention+MLP 7 投影", "attn+mlp（全量 target）", 64, L),
    ]
    for label, gkey, r, layers in rows:
        n = lora_params(GROUPS[gkey], r, layers)
        print(f"{label:38s} {layers:5d} {r:4d} {n/1e6:10.1f} M {100*n/total:7.3f}%")

    print()
    print("参考：Stage 1 只训的线性投影 = 640*4096 + 4096 = "
          f"{(640*4096+4096)/1e6:.2f} M（占比 {100*(640*4096+4096)/total:.4f}%）")


if __name__ == "__main__":
    main()
