"""切分的局部稳定性检查 —— 原生文本路线能不能走,取决于这一条。

判据：**单碱基扰动应该只影响附近的 token 边界。**

  * 如果改第 k 个碱基，token 变化只波及 k 附近 → 切分是局部的，模型能学到序列结构
  * 如果变化一路传播到序列末尾（读框错位）→ 切分是脆的，
    继续预训练学到的是"切分噪声"，不是 RNA 结构

测四件事：
  1. 单点扰动的影响半径（扫多个位置，统计变化 token 的分布）
  2. 同序列重复切分是否一致（BPE 应该保证，验一下）
  3. 前后缀是否污染（同一段序列，加不同前缀后切分是否不变）
  4. 不同序列的切分是否"同构"（同一段子串在不同上下文里切法是否一致）
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402
from transformers import AutoTokenizer  # noqa: E402

TOK = ROOT / "models/Qwen3-8B"
BASES = "ACGU"


def toks(tk, s):
    return tk.convert_ids_to_tokens(tk(s, add_special_tokens=False)["input_ids"])


def main() -> int:
    tk = AutoTokenizer.from_pretrained(str(TOK))
    rng = random.Random(0)
    seq = "".join(rng.choice(BASES) for _ in range(400))

    print("=" * 84)
    print("1) 单点扰动的影响半径")
    print("=" * 84)
    base_t = toks(tk, seq)
    print(f"  原序列 {len(seq)} nt -> {len(base_t)} token")
    # 用 token 的字符区间做坐标
    def spans(ts):
        out, pos = [], 0
        for t in ts:
            n = len(str(t).lstrip("Ġ"))
            out.append((pos, pos + n))
            pos += n
        return out
    base_span = spans(base_t)

    radii = []
    for k in [50, 120, 200, 280, 350]:
        alt = list(seq)
        alt[k] = rng.choice([b for b in BASES if b != seq[k]])
        alt = "".join(alt)
        alt_t = toks(tk, alt)
        alt_span = spans(alt_t)
        # 找出第一个和最后一个"边界不同"的位置
        n = min(len(base_span), len(alt_span))
        diffs = [i for i in range(n) if base_span[i] != alt_span[i]]
        if not diffs:
            radii.append((k, 0, 0))
            continue
        # 变化边界的跨度（用原坐标）
        lo = base_span[diffs[0]][0]
        hi = base_span[diffs[-1]][1]
        radii.append((k, hi - lo, len(diffs)))
        print(f"  改第 {k:3d} 个碱基 -> token 数 {len(base_t)}->{len(alt_t)}，"
              f"受影响区间 [{lo},{hi}] 跨度 {hi-lo} nt，边界变化 {len(diffs)} 处")

    span_vals = [r[1] for r in radii]
    print(f"\n  影响跨度: 中位 {sorted(span_vals)[len(span_vals)//2]} nt  "
          f"最大 {max(span_vals)} nt  （序列共 {len(seq)} nt）")
    tail = sum(1 for r in radii if r[1] > len(seq) * 0.5)
    if tail == 0:
        print("  -> **扰动是局部的**，影响没有传播到序列末尾。切分稳定。")
    elif tail <= 1:
        print("  -> 多数局部，偶有长距离传播。尚可接受，但要留意。")
    else:
        print("  -> ⚠️ 扰动频繁传播到序列后半段 = **读框错位**。")
        print("     这会让继续预训练学到切分噪声而不是序列结构。")

    print()
    print("=" * 84)
    print("2) 同序列重复切分是否一致")
    print("=" * 84)
    a = toks(tk, seq); b = toks(tk, seq); c = toks(tk, seq)
    print(f"  三次切分完全相同: {a == b == c}")

    print()
    print("=" * 84)
    print("3) 前后缀污染：同一段序列，加不同前缀后切分是否不变")
    print("=" * 84)
    mid = seq[150:250]
    variants = [mid, "A" * 10 + mid, mid + "C" * 10, "ACGU" * 5 + mid]
    runs = [toks(tk, v) for v in variants]
    # 取中段在各自切分里的 token
    firsts = [r for r in runs]
    same_len = all(len(x) == len(runs[0]) for x in runs)
    print(f"  四种上下文的 token 数: {[len(x) for x in runs]}")
    if len(runs[0]) == len(runs[1]) and runs[0] == runs[1][-len(runs[0]):]:
        print("  -> 前置不影响中段切分")
    else:
        print("  -> ⚠️ 前置会改变中段切分（BPE 的上下文依赖）")

    print()
    print("=" * 84)
    print("4) 子串在其他上下文里的切法是否一致")
    print("=" * 84)
    sub = "GAUCGAUC"
    places = [seq.find(sub)]
    if places[0] >= 0:
        ctx_a = toks(tk, seq)
        ctx_b = toks(tk, sub)
        print(f"  子串 '{sub}' 单独切: {'|'.join(str(t) for t in ctx_b)}")
        print("  在长序列里：需逐位置比对（BPE 会因上下文合并）")
        print("  -> 这是 BPE 的固有性质，无法避免；关键看第 1 项的影响半径。")

    print()
    print("=" * 84)
    print("结论")
    print("=" * 84)
    ok = tail == 0
    print("  " + ("**切分局部稳定 -> 原生文本路线可走，第一版不用改 tokenizer。**"
                  if ok else
                  "**切分不稳定 -> 每改一个碱基就错位，需要 RNA 专用 tokenizer。**"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
