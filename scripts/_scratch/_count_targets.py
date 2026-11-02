"""不加载模型，直接数各预设会挂上多少个模块、多少参数。

为什么需要：`replace_linear_with_sos` 是按 `name.endswith(suffix)` 匹配的，
而权重名和**模块名不一定一样** —— `patch_moe_experts` 把专家命名成 `gate_up` / `down`
（不是权重里的 `gate_up_proj` / `down_proj`），所以预设 D 的 `up_proj` 后缀
**不会**误伤那 20,480 个专家矩阵。这个区别光看权重表是看不出来的，必须对着代码核。

跑法： python scripts/_scratch/_count_targets.py
"""
import collections
import glob
import json
import os
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.qwen36 import LORA_TARGETS  # noqa: E402

MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")


def load_shapes():
    out = {}
    for f in sorted(glob.glob(os.path.join(MDL, "*.safetensors"))):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                out[k] = tuple(v.get("shape", ()))
    return out


def main() -> int:
    shapes = load_shapes()
    weights = {k: s for k, s in shapes.items() if k.endswith(".weight")}
    print(f"  权重表 {len(weights)} 个 .weight 张量\n")

    # patch_moe_experts 之后这些专家矩阵的**模块名**是 gate_up / down，
    # 不是权重名里的 gate_up_proj / down_proj。ModuleList 的子项名是下标。
    expert_like = [k for k in weights if "experts." in k]
    print("  专家张量（模块名会变成 gate_up / down + 下标）:")
    for k in expert_like[:2]:
        print(f"    {k}  {weights[k]}")
    print()

    for name in ("C", "D"):
        sufs = LORA_TARGETS[name]
        hits = [k[:-7] for k in weights if any(k[:-7].endswith(s) for s in sufs)]
        by_suf = collections.Counter()
        total = 0
        for b in hits:
            suf = next(s for s in sufs if b.endswith(s))
            sh = weights[b + ".weight"]
            by_suf[suf] += 1
            if len(sh) == 2:
                total += (sh[1] + sh[0]) * 8          # SOS r_tot=8
        print(f"  === 预设 {name}：{len(hits)} 个模块，SOS(r_tot=8) {total:,} 参数 "
              f"({total/1e6:.2f} M) ===")
        for suf in sufs:
            if by_suf[suf]:
                print(f"      {suf:<14} x{by_suf[suf]}")
        lay = sorted({int(re.search(r"layers\.(\d+)\.", b).group(1))
                      for b in hits if "layers." in b})
        print(f"      层号 {lay[0]}..{lay[-1]}（{len(lay)} 层）")
        print()

    print("  === 对照：专家矩阵会不会被 D 误伤 ===")
    for k in expert_like[:2]:
        base = k[:-7]
        print(f"    权重名 {base}")
        print(f"      被 D 后缀命中? {[s for s in LORA_TARGETS['D'] if base.endswith(s)] or '否'}")
    print("    模块名会是 gate_up / down + 下标 '0'..'255'")
    for m in ("gate_up", "down", "0"):
        print(f"      '{m}' 被 D 后缀命中? "
              f"{[s for s in LORA_TARGETS['D'] if m.endswith(s)] or '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
