"""数据重做的试点：先做 10 条，看形状对不对。

只做**零成本、零幻觉、100% 可验证**的那部分 —— 答案直接取自字段或从序列算，
不经过 LLM。自由文本模板（功能描述等）等这一步看过再做。

关键检查：问题必须**不透露 RNA 身份**，答案必须**随 RNA 变化**。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"

# ---------------- 模板：全部不透露 RNA 身份 ----------------
# (问题, 答案取值函数)  —— 问题里绝不能出现 RNA 的名字/类型/物种

def gc(seq):
    s = seq.upper()
    return 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1)

def has_motif(seq, m):
    return m in seq.upper()

TEMPLATES = [
    ("How many nucleotides long is this RNA?",
     lambda r: f"{len(r['sequence'])} nt"),
    ("What type of RNA is this?",
     lambda r: r.get("rna_type") or ""),
    ("Which organism does this RNA come from?",
     lambda r: (r.get("organism") or "").replace("taxid:", "taxid ")),
    ("What is the GC content of this RNA?",
     lambda r: f"{gc(r['sequence']):.1f}%"),
    ("Does this RNA contain the motif GGAA?",
     lambda r: "Yes" if has_motif(r["sequence"], "GGAA") else "No"),
    ("What is the first nucleotide of this RNA?",
     lambda r: r["sequence"][0].upper()),
    ("Does this RNA contain the motif CUGU?",
     lambda r: "Yes" if has_motif(r["sequence"], "CUGU") else "No"),
    ("Is this RNA longer than 100 nucleotides?",
     lambda r: "Yes" if len(r["sequence"]) > 100 else "No"),
]


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("sequence"):
                    recs.append(r)

    # 取 10 条**多样**的：按 rna_type 轮流取，避免全是同一类
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

    print(f"总 RNA {len(recs)}   类型 {len(by_type)}   试点取 {len(picked)} 条")
    print()
    print("=" * 90)
    print("生成的 QA 样例")
    print("=" * 90)
    n_qa = 0
    for k, r in enumerate(picked, 1):
        print(f"\n--- RNA {k}  {r['uid'][:34]}  长度 {len(r['sequence'])}  "
              f"类型 {r.get('rna_type')} ---")
        print(f"    序列 {r['sequence'][:56]}{'...' if len(r['sequence'])>56 else ''}")
        for q, fn in TEMPLATES:
            a = fn(r)
            n_qa += 1
            print(f"    Q: {q}")
            print(f"    A: {a}")
    print()
    print("=" * 90)
    print(f"共 {n_qa} 条 QA  ({len(picked)} RNA x {len(TEMPLATES)} 模板)")
    print("=" * 90)

    # ---- 关键检查 1：问答比 ----
    print()
    print("检查 1：问题是否与答案形成 1:1（旧数据的病）")
    qs = [q for q, _ in TEMPLATES] * len(picked)
    print(f"  QA {n_qa} 条   不同问题 {len(set(qs))} 个   "
          f"问答比 {len(set(qs))/n_qa:.4f}")
    print(f"  （旧数据是 85440/92680 = 0.922，几乎 1:1）")

    # ---- 关键检查 2：每个模板的答案熵 ----
    print()
    print("检查 2：每个模板的答案熵 —— 太低说明模型乱猜也能对")
    print()
    print(f"{'问题(截断)':<48} {'答案种类':>8} {'熵':>8} {'上限':>8} {'占比':>7}")
    for q, fn in TEMPLATES:
        ans = []
        for r in recs[:1000]:                       # 用 1000 条估熵
            try:
                a = fn(r)
            except Exception:
                continue
            if a:
                ans.append(a)
        if not ans:
            continue
        c = Counter(ans)
        H = -sum((v / len(ans)) * math.log(v / len(ans)) for v in c.values())
        Hmax = math.log(len(c))
        print(f"  {q[:46]:<46} {len(c):>8} {H:>8.3f} {Hmax:>8.3f} "
              f"{H/Hmax if Hmax>0 else 0:>7.3f}")

    # ---- 关键检查 3：问题不透露身份 ----
    print()
    print("检查 3：问题里有没有泄漏 RNA 身份的词？")
    leak = 0
    for q, _ in TEMPLATES:
        low = q.lower()
        for w in ("mir", "rrna", "trna", "snrna", "s. pombe", "homo", "mitochondrial"):
            if w in low:
                print(f"  !! 泄漏: {q}  (含 {w})"); leak += 1
    if not leak:
        print("  PASS 全部模板都不含身份词")

    out = ROOT / "data/processed/pilot_rnaqa.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            for q, fn in TEMPLATES:
                a = fn(r)
                if a:
                    f.write(json.dumps({
                        "uid": r["uid"], "sequence": r["sequence"],
                        "question": q, "answer": a, "template": q,
                        "source": "structured", "kind": "closed",
                    }, ensure_ascii=False) + "\n")
    print(f"\n[save] {out}  {n_qa} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
