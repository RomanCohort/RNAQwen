"""模板筛选器：瞎猜基线

模型有一条退路 —— 完全无视 RNA，直接输出答案的边缘分布。
这条退路能拿多少分，决定了 RNA 还有多少空间。

三个可算的数（全部零成本、不用训练）：
  · 瞎猜命中率   = 测试集上"永远答最常见答案"的正确率   -> 越低越好
  · 归一化熵     = H(A)/log(答案种类)                   -> 越高越好
  · 无视RNA收益  = 1 - 瞎猜命中率（粗略下界）           -> 越高越好

判据：命中率 > 40% 的模板基本没用（模型不用读 RNA 也能对一大半）。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"


def gc(seq):
    s = seq.upper()
    return 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1)


def first_nt(r):
    return r["sequence"][0].upper()


# ---- 候选模板。删掉了 Which organism（序列里没有物种信息）----
TEMPLATES = [
    ("How many nucleotides long is this RNA?",
     lambda r: f"{len(r['sequence'])} nt"),
    ("What is the GC content of this RNA?",
     lambda r: f"{gc(r['sequence']):.1f}%"),
    ("What is the first nucleotide of this RNA?", first_nt),
    ("What is the last nucleotide of this RNA?",
     lambda r: r["sequence"][-1].upper()),
    ("Is this RNA longer than 100 nucleotides?",
     lambda r: "Yes" if len(r["sequence"]) > 100 else "No"),
    ("Does this RNA contain the motif GGAA?",
     lambda r: "Yes" if "GGAA" in r["sequence"].upper() else "No"),
    ("Does this RNA contain the motif CUGU?",
     lambda r: "Yes" if "CUGU" in r["sequence"].upper() else "No"),
    ("Does this RNA start with a purine (A or G)?",
     lambda r: "Yes" if r["sequence"][0].upper() in "AG" else "No"),
    ("What type of RNA is this?",
     lambda r: r.get("rna_type") or ""),
    ("Is this RNA longer than 500 nucleotides?",
     lambda r: "Yes" if len(r["sequence"]) > 500 else "No"),
    ("What is the most frequent nucleotide in this RNA?",
     lambda r: Counter(r["sequence"].upper()).most_common(1)[0][0]),
    ("What percentage of this RNA is A? (to 1 decimal)",
     lambda r: f"{100.0*r['sequence'].upper().count('A')/max(len(r['sequence']),1):.1f}%"),
    ("Which organism does this RNA come from?",
     lambda r: (r.get("organism") or "").replace("taxid:", "taxid ")),
]


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("sequence"):
                    recs.append(r)
    N = len(recs)
    print(f"RNA 总数 {N}")
    print()

    # 固定划分：前 80% 当"训练集"估边缘分布，后 20% 当"测试集"评瞎猜
    cut = int(N * 0.8)
    tr, te = recs[:cut], recs[cut:]

    print("=" * 96)
    print(f"{'模板':<50} {'种类':>6} {'归一熵':>8} {'瞎猜命中':>9} {'判定':>10}")
    print("=" * 96)
    rows = []
    for q, fn in TEMPLATES:
        cnt_tr = Counter()
        for r in tr:
            try:
                a = fn(r)
            except Exception:
                continue
            if a:
                cnt_tr[a] += 1
        if not cnt_tr:
            continue
        top = cnt_tr.most_common(1)[0][0]

        hit = 0
        n = 0
        for r in te:
            try:
                a = fn(r)
            except Exception:
                continue
            if not a:
                continue
            n += 1
            if a == top:
                hit += 1
        hit_rate = hit / max(n, 1)

        p = [c / sum(cnt_tr.values()) for c in cnt_tr.values()]
        H = -sum(x * math.log(x) for x in p if x > 0)
        Hmax = math.log(len(cnt_tr))
        norm = H / Hmax if Hmax > 0 else 0.0

        if hit_rate > 0.40:
            verdict = "**扔掉**"
        elif hit_rate > 0.20:
            verdict = "弱"
        else:
            verdict = "好"
        rows.append((q, len(cnt_tr), norm, hit_rate, verdict))
        print(f"  {q[:48]:<48} {len(cnt_tr):>6} {norm:>8.3f} "
              f"{100*hit_rate:>8.1f}% {verdict:>10}")

    print()
    good = [r for r in rows if r[3] <= 0.20]
    weak = [r for r in rows if 0.20 < r[3] <= 0.40]
    bad = [r for r in rows if r[3] > 0.40]
    print("=" * 96)
    print(f"好 {len(good)}   弱 {len(weak)}   扔掉 {len(bad)}")
    print("=" * 96)
    for q, _, _, h, _ in bad:
        print(f"  扔掉: {q}   瞎猜命中 {100*h:.1f}%")
    for q, _, _, h, _ in weak:
        print(f"  弱  : {q}   瞎猜命中 {100*h:.1f}%")

    # ---- 在 10 条试点上看具体长什么样 ----
    print()
    print("=" * 96)
    print("10 条试点上的实际取值（看瞎猜会不会蒙对）")
    print("=" * 96)
    by_type = {}
    for r in recs:
        by_type.setdefault(r.get("rna_type", "?"), []).append(r)
    picked, i = [], 0
    while len(picked) < 10:
        got = False
        for t, lst in by_type.items():
            if i < len(lst) and len(picked) < 10:
                picked.append(lst[i]); got = True
        if not got:
            break
        i += 1

    keep = [q for q, _, _, h, _ in rows if h <= 0.40]
    print(f"{'RNA':<6} " + " ".join(f"{q[:11]:>12}" for q in keep[:6]))
    for k, r in enumerate(picked, 1):
        vals = []
        for q, fn in TEMPLATES:
            if q in keep[:6]:
                try:
                    vals.append(str(fn(r))[:11])
                except Exception:
                    vals.append("?")
        print(f"  {k:<4} " + " ".join(f"{v:>12}" for v in vals))

    out = ROOT / "reports/template_screen.json"
    out.write_text(json.dumps(
        [dict(q=q, kinds=k, norm_entropy=round(ne, 4), blind_hit=round(h, 4), verdict=v)
         for q, k, ne, h, v in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
