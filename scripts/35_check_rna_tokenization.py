"""Qwen3 的 tokenizer 怎么切 RNA 序列？这决定"原生文本路线"是不是几乎零代价。

要看三件事：
  1. 压缩比：一条 1022 nt 的序列变成多少 token
  2. 切分方式：是逐碱基、还是 BPE 把常见 k-mer 合成一个 token
  3. 有没有"奇怪"的切分：碱基被吸进非核酸 token，或前后缀被污染

如果压缩比接近 1:1（一碱基一 token），1022 nt 就是 1022 token —— 仍然可接受
（Qwen3 上下文 32K）。如果 BPE 能合成 k-mer，那更好。
真正要警惕的是第 3 类：切得"很奇怪"，那会破坏序列的局部结构。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
from transformers import AutoTokenizer  # noqa: E402

TOK = ROOT / "models/Qwen3-8B"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"


def main() -> int:
    tk = AutoTokenizer.from_pretrained(str(TOK))
    print(f"  Qwen3 vocab size = {len(tk)}")

    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    # 按长度分层取 8 条
    recs.sort(key=lambda r: len(r.get("sequence") or ""))
    idx = [0, len(recs)//7, len(recs)*2//7, len(recs)*3//7, len(recs)*4//7,
           len(recs)*5//7, len(recs)*6//7, len(recs)-1]
    pick = [recs[i] for i in idx if recs[i].get("sequence")]

    print()
    print("  %8s %8s %8s %9s" % ("nt", "tokens", "ratio", "示例切分"))
    ratios = []
    for r in pick:
        s = r["sequence"]
        ids = tk(s, add_special_tokens=False)["input_ids"]
        toks = tk.convert_ids_to_tokens(ids)
        ratios.append(len(ids) / max(len(s), 1))
        head = "|".join(str(t).replace("Ġ", "_") for t in toks[:8])
        print("  %8d %8d %8.3f  %s" % (len(s), len(ids), len(ids)/len(s), head[:50]))

    print()
    print(f"  平均压缩比 tokens/nt = {sum(ratios)/len(ratios):.3f}")
    print("  （1.0 = 逐碱基；<1 = BPE 把 k-mer 合成了）")

    # 逐碱基切分的占比：单字符 token 有多少
    s = pick[len(pick)//2]["sequence"]
    ids = tk(s, add_special_tokens=False)["input_ids"]
    toks = tk.convert_ids_to_tokens(ids)
    lens = Counter(len(str(t).lstrip("Ġ")) for t in toks)
    print()
    print("  token 长度分布（字符数）:", dict(sorted(lens.items())))
    single = sum(v for k, v in lens.items() if k == 1) / max(len(toks), 1)
    print(f"  单字符 token 占比 {100*single:.1f}%")

    # 关键检查：把序列大写/加空格，token 数会不会变（说明切分不稳定）
    s2 = " ".join(s)
    n2 = len(tk(s2, add_special_tokens=False)["input_ids"])
    print(f"  同一序列加空格后 token 数: {n2}（原 {len(ids)}）"
          f"  {'→ 切分对分隔符敏感' if abs(n2-len(ids))>len(ids)*0.2 else '→ 基本不变'}")

    print()
    if sum(ratios)/len(ratios) > 0.95:
        print("  -> **逐碱基**。1022 nt ≈ 1022 token，上下文装得下，路线可行。")
    else:
        print("  -> BPE 有合并，比逐碱基更省。")
    print("  下一步：看切分是否**稳定且局部**（相邻碱基的 token 边界要一致），")
    print("  否则继续预训练学到的是切分噪声而不是序列结构。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
