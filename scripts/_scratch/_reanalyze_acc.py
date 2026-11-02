"""Re-read the 300 stored generations instead of re-running them.

The stored exact-match table is depressed by a decoder that does not stop at the
answer: a typical A-arm miss is "CUGG\nuser\nUGCC" for gold "CUGG" (char-F1 0.50,
exact-match 0).  This separates three things the single number merges:

  exact     the whole normalized generation equals the answer
  prefix    the answer is a prefix of it            <- "did it know the answer"
  turn      the generation ran past the answer turn (contains user/assistant)

Also recomputes normalization from the raw string, and asserts it matches the value
the scoring pass stored, so this cannot quietly use a different normalization.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

RAW = Path(r"C:\qwen-api\RNAQwen\reports\acc_by_source_D_qwen36_sos_final.json.raw.json")
ARMS = ("A", "B", "C")
TURN = re.compile(r"(user|assistant|<\|im_(start|end)\|>)", re.I)


def norm(s: str) -> str:
    return "".join(c for c in s if c.isalpha()).upper()


def main() -> int:
    d = json.loads(RAW.read_text(encoding="utf-8"))
    per = {a: dict(n=0, exact=0, prefix=0, turn=0, f1=0.0) for a in ARMS}
    table = []
    mismatch = 0
    for tpl, arms in d.items():
        row = {"template": tpl}
        for arm in ARMS:
            st = dict(n=0, exact=0, prefix=0, turn=0, f1=0.0)
            for stored_norm, gold, f1, raw, q in arms[arm]:
                g = norm(raw)
                if g != stored_norm:
                    mismatch += 1
                gold = norm(gold)
                st["n"] += 1
                st["exact"] += (g == gold)
                st["prefix"] += bool(gold) and g.startswith(gold)
                st["turn"] += bool(TURN.search(raw))
                st["f1"] += float(f1)
            for k in st:
                per[arm][k] += st[k] if k != "f1" else st[k]
            st["f1"] = st["f1"] / max(st["n"], 1)
            row[arm] = st
        table.append(row)

    if mismatch:
        print(f"  ** {mismatch} 条的归一化与落盘值不一致 —— 口径不同，结论作废")
        return 3

    print(f"归一化与落盘逐条一致（{sum(r['A']['n'] for r in table)} 条/臂 x 3 臂）\n")
    hdr = f"{'arm':>4} {'n':>4} {'exact':>8} {'prefix':>8} {'turn':>8} {'charF1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for arm in ARMS:
        p = per[arm]
        n = max(p["n"], 1)
        print(f"{arm:>4} {p['n']:>4} {p['exact']/n:>7.1%} {p['prefix']/n:>8.1%} "
              f"{p['turn']/n:>8.1%} {p['f1']/n:>7.3f}")

    print("\n按模板（prefix = 答案是否作为前缀出现；n=30）")
    h = f"{'template':<52} {'A ex':>6} {'A px':>6} {'B px':>6} {'C px':>6}"
    print(h)
    print("-" * len(h))
    for row in table:
        t = row["template"]
        cells = []
        for arm in ARMS:
            st = row[arm]
            cells.append((st["exact"] / st["n"], st["prefix"] / st["n"]))
        print(f"{t[:50]:<52} {cells[0][0]:>6.1%} {cells[0][1]:>6.1%} "
              f"{cells[1][1]:>6.1%} {cells[2][1]:>6.1%}")

    print("\nA 臂前缀对的样例（看它答对之后怎么跑飞）")
    shown = 0
    for row in table:
        for stored_norm, gold, f1, raw, q in d[row["template"]]["A"]:
            g = norm(raw)
            if g.startswith(norm(gold)) and g != norm(gold) and shown < 6:
                print(f"  [{row['template'][:44]}] gold {gold} -> {raw!r}")
                shown += 1

    # 每模板 30 条里，「答案槽」（生成的前 len(gold) 个字符）有多少种不同的串？
    # C 臂（码置零）的退化长相就在这里：同一模板不管喂哪条序列，答案槽基本是常数。
    # 只取答案槽，不取整段生成 —— 否则量到的是后半段絮叨的多样性。
    print("\n答案槽多样性（每模板 30 条；distinct = 不同答案槽串数，mode = 最高频串占比）")
    h = f"{'arm':>4} {'distinct/30':>12} {'mode 串占比':>12} {'命中前 1 串':>12}"
    print(h)
    print("-" * len(h))
    for arm in ARMS:
        ds, ms, ls = [], [], []
        for row in table:
            outs = [o for o in (norm(x[3])[:len(norm(x[1]))]
                                for x in d[row["template"]][arm]) if o]
            if not outs:
                continue
            c = Counter(outs)
            ds.append(len(c))
            ms.append(c.most_common(1)[0][1] / len(outs))
            ls.append(c.most_common(1)[0][1] / len(outs))
        n = len(ds)
        print(f"{arm:>4} {sum(ds)/n:>11.1f} {sum(ms)/n:>11.1%} {sum(ls)/n:>11.1%}")
    print("\n  A 应接近 30（每条序列一个答案）；C 应接近 1 —— 那就是「问题先验」的长相。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
