"""交叉验证：我自建训练回路的 loss 序列 vs `20_train_stage1.py` 官方入口的 loss 序列。

这一点很重要 —— 如果我自建的回路（用于计时）与官方回路行为不一致，
那所有 s/step 数字都不可信。两边都是：50 条 × 2 epoch × bs=1 × accum=1 × grad-ckpt on。
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ls = json.loads((ROOT / "reports" / "_raw_timing_loss.json").read_text(encoding="utf-8-sig"))
    mine = ls["runs"][0]["losses"]                      # 100 步

    pub = json.loads((ROOT / "checkpoints" / "_losscheck" / "run_summary.json")
                     .read_text(encoding="utf-8"))
    hist = pub["result"]["history"]                     # 每 10 步一条
    theirs = {h["step"]: h["loss"] for h in hist}

    print("=== 官方入口 vs 自建回路 ===")
    print(f"{'step':>5} {'官方 20_train':>14} {'我的回路':>10} {'差':>10}")
    diffs = []
    for step in sorted(theirs):
        a = theirs[step]
        b = mine[step - 1]          # mine 是 0-based 列表
        diffs.append(b - a)
        print(f"{step:>5} {a:>14.6f} {b:>10.6f} {b-a:>+10.2e}")

    print(f"\n最大绝对差 = {max(abs(d) for d in diffs):.3e}")
    print(f"平均绝对差 = {st.mean(abs(d) for d in diffs):.3e}")
    print(f"最后 10 步: 官方 {st.mean([theirs[s] for s in sorted(theirs)[-1:]]) if False else theirs[100]:.6f}  "
          f"我的 {st.mean(mine[-10:]):.6f}")

    lr = [h["lr"] for h in hist]
    print(f"\nlr 序列（官方）: {lr[0]:.3e} ... {lr[-1]:.3e}  (warmup=10 后进入 cosine)")

    # 官方 run 的总耗时
    print(f"\n官方 run_summary steps={pub['result']['steps']}  "
          f"config.epochs={pub['config']['epochs']}  n_samples={pub['n_samples']}")
    print(f"官方 elapsed_s 序列: {[h['elapsed_s'] for h in hist]}")
    tail = hist[-1]["elapsed_s"] - hist[-2]["elapsed_s"]
    print(f"最后 10 步耗时 {tail:.1f}s -> {tail/10:.3f} s/step（含我这次实测的冷启动开销）")

    verdict = "一致（自建回路可信）" if max(abs(d) for d in diffs) < 1e-3 else "不一致，需排查"
    print(f"\n结论: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
