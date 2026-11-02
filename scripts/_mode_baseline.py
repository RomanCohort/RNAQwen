"""先验预测器的地板：按问题取训练集里最常见的答案。

为什么必须先有这一行
--------------------
A_v3 的准确率日志里，同一个模板下的**三条不同 RNA 给出了完全相同的答案**
（first-6 -> 'GGGGAG'，G 计数 -> '273'，A -> '200'，C -> '250'...）。
这不是「答案错」，是「模型没读 RNA，在按问题的先验答」。

那就得把这条先验本身量出来：一个**不看 RNA、只按问题给众数答案**的预测器，
在这批数据上能拿多少分。任何模型的准确率如果不显著高于它，
就不能声称自己「用了 RNA」—— acc(C)（不给 RNA）高于 acc(A)（给真 RNA），
正是这一行在作祟。

跑法:
    python scripts/_mode_baseline.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_seq30_split.jsonl"
OUT = ROOT / "reports/mode_baseline.json"


def norm(s: str) -> str:
    return "".join(ch for ch in s.strip().upper() if ch.isalnum() or ch in ".%").rstrip(".")


def main() -> int:
    train, test = [], []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (train if r.get("split") == "train" else test).append(r)
    print(f"[data] train {len(train):,} / test {len(test):,}")

    cnt: dict[str, Counter] = {}
    for r in train:
        cnt.setdefault(r["question"], Counter())[norm(r["answer"])] += 1
    mode = {q: c.most_common(1)[0][0] for q, c in cnt.items()}

    hit = sum(1 for r in test if mode.get(r["question"]) == norm(r["answer"]))
    overall = hit / max(len(test), 1)
    print(f"\n[地板] 全测试集 众数答案准确率 {100*overall:.2f}%  ({hit}/{len(test)})")

    per, rows = {}, []
    byq: dict[str, list] = {}
    for r in test:
        byq.setdefault(r["question"], []).append(r)
    acc_of = {q: sum(1 for r in rs if mode.get(q) == norm(r["answer"])) / len(rs)
              for q, rs in byq.items()}
    print(f"\n{'准确率':>8}  {'n':>5}  {'众数答案':<22}  问题")
    for q, rs in sorted(byq.items(), key=lambda kv: -acc_of[kv[0]]):
        a = acc_of[q]
        per[q] = a
        rows.append(dict(question=q, acc=a, n=len(rs), mode=mode.get(q, "")))
        print(f"{100*a:>7.1f}%  {len(rs):>5}  {str(mode.get(q,''))[:22]:<22}  {q[:56]}")

    # 63_accuracy_3arm.py 用的那 90 条（按模板轮流取）上的地板 —— 与实测 acc 同口径
    picked, i = [], 0
    while len(picked) < 90 and i < 10 ** 6:
        got = False
        for q, lst in byq.items():
            if i < len(lst) and len(picked) < 90:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1
    h90 = sum(1 for r in picked if mode.get(r["question"]) == norm(r["answer"]))
    print(f"\n[同口径] 63_ 那 90 条上的众数地板 {100*h90/max(len(picked),1):.1f}%  ({h90}/{len(picked)})")
    per90 = {}
    for r in picked:
        q = r["question"]
        per90.setdefault(q, []).append(mode.get(q) == norm(r["answer"]))
    print("  按模板（每条 3 个样本）：")
    for q, v in sorted(per90.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"    {100*sum(v)/len(v):>5.1f}%  {q[:56]}")

    top = max(per.values()) if per else 0.0
    print(f"\n[判据] 最高的单模板地板 {100*top:.1f}%；A 臂实测 acc(A) 16.7% / acc(C) 25.6%（n=90）")
    print("       => 实测值若落在 [地板, 地板+噪声] 区间内，就是「先验预测器」，不是「读了 RNA」。")
    OUT.write_text(json.dumps(dict(overall=overall, per_template=per, rows=rows,
                                   n_test=len(test)), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[save] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
