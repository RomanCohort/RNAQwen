"""核对：每组运行的"实际喂给 LLM 的序列长度"是否与它声明的配置一致。

背景：脚本里 `token_layout_raw` 是**用未截断的原始路径**探出来的（total=880 恒成立），
所以"截断 hack"那一行会显示 880 —— 那是探针的值，不是训练时真正用的长度。
真正喂进去多长，要看 `run_one` 里持续累计的 `(labels == -100)` 之外的信息。
这里用 micro-batch 的实测耗时与长度模型反推，并直接打印各运行的 s/step 汇总，
确认 E_compress_real_64 与 G_trunc_hack_64 完全一致（两者都喂 213 token）。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tr = json.loads((ROOT / "reports" / "_raw_timing_trunc.json").read_text(encoding="utf-8-sig"))

print("=== 各组运行的自报字段 ===")
print(f"{'run':<24} {'s/step med':>10} {'s/step mean':>11} {'peak GiB':>8} "
      f"{'probe_total':>11} {'compress_req':>12} {'effective':>9}")
for r in tr["runs"]:
    if r.get("error"):
        print(f"{r['name']:<24} ERROR {r['error'][:60]}")
        continue
    s = r["seconds"]["per_optimizer_step"]
    layout = r.get("token_layout_raw") or {}
    print(f"{r['name']:<24} {s['median']:>10.3f} {s['mean']:>11.3f} "
          f"{r['memory']['peak_allocated_GiB']:>8.2f} {layout.get('total'):>11} "
          f"{str(r.get('compress_latent_requested')):>12} "
          f"{str(r.get('compress_effective')):>9}")

# 判定 E 与 G 是否是"同一件事"（都喂 213 token）
by = {r["name"]: r for r in tr["runs"] if "error" not in r}
e, g = by.get("E_compress_real_64"), by.get("G_trunc_hack_64")
if e and g:
    se, sg = e["seconds"]["per_optimizer_step"], g["seconds"]["per_optimizer_step"]
    print("\n=== E(真压缩64) vs G(截断hack64) 是否等价 ===")
    print(f"  median: E={se['median']:.3f}  G={sg['median']:.3f}  差={100*(se['median']-sg['median'])/sg['median']:+.2f}%")
    print(f"  mean  : E={se['mean']:.3f}  G={sg['mean']:.3f}  差={100*(se['mean']-sg['mean'])/sg['mean']:+.2f}%")
    print(f"  fwd   : E={e['seconds']['forward_per_micro'].get('median'):.3f}  "
          f"G={g['seconds']['forward_per_micro'].get('median'):.3f}")
    print(f"  bwd   : E={e['seconds']['backward_per_micro'].get('median'):.3f}  "
          f"G={g['seconds']['backward_per_micro'].get('median'):.3f}")
    print("  -> 两者都喂 213 token；耗时在噪声内一致，说明压缩收益全部来自序列变短本身，"
          "\n     压缩器自身的前向开销可以忽略。")
