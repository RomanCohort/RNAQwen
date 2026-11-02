"""确认 35B-A3B 能不能真的加载。

AutoConfig 过了只说明 config 认得，不代表 AutoModelForCausalLM 有对应的类。
真正要确认三件事：
  1. 26 个分片按 index.json 全部存在、safetensors 头可解析
  2. AutoModelForCausalLM 能解析出模型类
  3. 4-bit 能不能装下（只估内存，不真加载 67 GB）
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rnaqwen  # noqa: F401,E402  必须先于 transformers

P = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")

print("=" * 70)
print("1. 分片完整性")
print("=" * 70)
idx = json.loads((P / "model.safetensors.index.json").read_text(encoding="utf-8"))
wmap = idx.get("weight_map", {})
files = sorted(set(wmap.values()))
print(f"  index 引用 {len(wmap):,} 个权重，分布在 {len(files)} 个分片")
tot = 0
bad = []
for f in files:
    fp = P / f
    if not fp.exists():
        bad.append(f"{f} 缺失")
        continue
    sz = fp.stat().st_size
    tot += sz
    try:
        with open(fp, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        # safetensors 头里声明的数据长度必须和文件实际大小对得上
        need = 8 + n + max((v["data_offsets"][1] for v in hdr.values()
                            if isinstance(v, dict) and "data_offsets" in v), default=0)
        if need > sz:
            bad.append(f"{f} 头声明 {need} > 实际 {sz}")
    except Exception as e:
        bad.append(f"{f} 头解析失败 {type(e).__name__}")
print(f"  合计 {tot/1024**3:.2f} GB")
print("  " + ("**全部正常**" if not bad else "问题: " + "; ".join(bad[:5])))

print()
print("=" * 70)
print("2. AutoModelForCausalLM 能不能解析出模型类")
print("=" * 70)
from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(str(P))
print(f"  model_type = {cfg.model_type}")
for k in ("num_hidden_layers", "hidden_size", "vocab_size", "num_experts",
          "num_experts_per_tok", "n_routed_experts", "moe_intermediate_size",
          "num_attention_heads", "num_key_value_heads"):
    if hasattr(cfg, k):
        print(f"    {k} = {getattr(cfg, k)}")
try:
    cls = AutoModelForCausalLM._model_mapping.get(type(cfg))
    print(f"  **模型类 = {cls}**" if cls else "  **没有对应的模型类 —— 这个架构 transformers 还没实现**")
except Exception as e:
    print(f"  解析失败: {type(e).__name__}: {str(e)[:200]}")

print()
print("=" * 70)
print("3. 内存估算（不真加载）")
print("=" * 70)
n_param = tot  # safetensors 里就是参数，bf16 每参数 2 字节
p_est = tot / 2
print(f"  参数量约 {p_est/1e9:.1f} B")
for name, bpp in (("bf16", 2), ("8-bit", 1), ("4-bit", 0.5)):
    print(f"    {name:<6} 权重约 {p_est*bpp/1e9:.1f} GB")
print("  物理 RAM 63.6 GB（实际可用约 50 GB）")
