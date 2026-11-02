"""换底座的关键一问：文本-only 类能不能吃下 VLM 目录里的权重。

背景
----
config.json 的顶层 arch 是 Qwen3_5MoeForConditionalGeneration（VLM），
checkpoint 里的权重名是 `model.language_model.layers.*`。
而 AutoModelForCausalLM.from_config(顶层 config) 建出来的是 Qwen3_5MoeForCausalLM，
它的 `model` 子模块下面**没有** language_model 这一层。

前缀差一层 —— 要么 transformers 自己会映射，要么得手工改 key。
这个不确认，换底座就是撞墙。

做法：在 meta 上建文本模型，把它的 state_dict key 和 checkpoint 的 index 对一遍。
不加载任何真实权重，纯 key 比对。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402

MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")


def main() -> int:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    ac = AutoConfig.from_pretrained(str(MDL))
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(ac)
    exp = set(m.state_dict().keys())
    print(f"[class] {type(m).__name__}   参数 {sum(p.numel() for p in m.parameters())/1e9:.2f} B")
    print(f"[class] state_dict keys {len(exp):,}")
    print(f"[class] model 子模块 {[k for k, _ in m.model.named_children()][:12]}")

    idx = json.loads((MDL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    ck = set(idx["weight_map"].keys())
    print(f"[ckpt]  keys {len(ck):,}")

    print()
    print("=== 直接比对（不做任何重映射）===")
    inter = exp & ck
    only_ck = ck - exp
    only_ex = exp - ck
    print(f"  交集 {len(inter):,}   只在 ckpt {len(only_ck):,}   只在模型 {len(only_ex):,}")

    def top(s: set[str], n: int = 12, label=""):
        from collections import Counter
        c = Counter(".".join(k.split(".")[:3]) for k in s)
        print(f"  {label}")
        for k, v in c.most_common(n):
            print(f"      {v:>5}  {k}")

    if only_ck:
        top(only_ck, 10, "只在 ckpt 里的前缀（前 10）:")
    if only_ex:
        top(only_ex, 10, "只在模型里的前缀（前 10）:")

    # 试试去掉 model.language_model. 这一层
    print()
    print("=== 假设 ckpt key 去掉 'model.language_model.' 前缀 ===")
    remap = {}
    for k in ck:
        nk = k.replace("model.language_model.", "model.", 1)
        remap[k] = nk
    newkeys = set(remap.values())
    inter2 = exp & newkeys
    print(f"  交集 {len(inter2):,}   只在重映射后 ckpt {len(newkeys-exp):,}   "
          f"只在模型 {len(exp-newkeys):,}")
    if newkeys - exp:
        top(newkeys - exp, 8, "  仍对不上的:")
    if exp - newkeys:
        top(exp - newkeys, 8, "  模型里缺的:")

    print()
    print("=== 关键张量 ===")
    for k in sorted(exp):
        if "embed_tokens" in k or "lm_head" in k:
            s = m.state_dict()[k]
            print(f"  模型  {k:<48} {tuple(s.shape)}")
    for k in sorted(ck):
        if "embed_tokens" in k or "lm_head" in k:
            print(f"  ckpt  {k}")

    print()
    print("=== 视觉塔张量（要丢掉多少个）===")
    vis = [k for k in ck if "visual" in k or "vision" in k]
    print(f"  {len(vis)} 个")
    for k in vis[:6]:
        print(f"      {k}")

    print()
    print("=== mtp（多 token 预测）层 ===")
    mtp = [k for k in ck if k.startswith("mtp.")]
    print(f"  {len(mtp)} 个   模型里有没有 mtp: "
          f"{any(k.startswith('mtp.') for k in exp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
