"""扩到 30 个模板并生成全量 QA（4,481 × 30 = 134,430 条）

流程：候选 -> 瞎猜基线筛选（<20% 且 归一熵 >0.5）-> 取前 30 -> 生成。
纯 Python，不用 API，不用 GPU。
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
OUT = ROOT / "data/processed/rnaqa_seq30.jsonl"
RC = str.maketrans("ACGU", "UGCA")


def S(r):
    return (r.get("sequence") or r.get("seq") or "").upper()


def gc(s):
    return 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1)


def pct(s, ch):
    return f"{100.0*s.count(ch)/max(len(s),1):.1f}%"


CAND = [
    # 子序列 / 反向互补
    ("What are the first 4 nucleotides of this RNA?", lambda r: S(r)[:4], 4),
    ("What are the last 4 nucleotides of this RNA?", lambda r: S(r)[-4:], 4),
    ("What are the first 6 nucleotides of this RNA?", lambda r: S(r)[:6], 6),
    ("What are the last 6 nucleotides of this RNA?", lambda r: S(r)[-6:], 6),
    ("What are the first 8 nucleotides of this RNA?", lambda r: S(r)[:8], 8),
    ("What are the last 8 nucleotides of this RNA?", lambda r: S(r)[-8:], 8),
    ("What is the reverse complement of the first 4 nucleotides?",
     lambda r: S(r)[:4].translate(RC)[::-1], 4),
    ("What is the reverse complement of the first 6 nucleotides?",
     lambda r: S(r)[:6].translate(RC)[::-1], 6),
    ("What are the nucleotides at positions 1 through 6 of this RNA?",
     lambda r: S(r)[:6], 6),
    ("What is the dinucleotide at positions 1 and 2 of this RNA?",
     lambda r: S(r)[:2], 2),
    ("What is the dinucleotide at positions 5 and 6 of this RNA?",
     lambda r: S(r)[4:6], 2),
    # 位置特异
    ("What is the 5th nucleotide of this RNA?", lambda r: S(r)[4], 5),
    ("What is the 10th nucleotide of this RNA?", lambda r: S(r)[9], 10),
    ("What is the 15th nucleotide of this RNA?", lambda r: S(r)[14], 15),
    ("What is the 20th nucleotide of this RNA?", lambda r: S(r)[19], 20),
    ("What is the 25th nucleotide of this RNA?", lambda r: S(r)[24], 25),
    ("What is the 40th nucleotide of this RNA?", lambda r: S(r)[39], 40),
    # 计数
    ("How many G nucleotides are in this RNA?", lambda r: str(S(r).count("G")), 1),
    ("How many A nucleotides are in this RNA?", lambda r: str(S(r).count("A")), 1),
    ("How many C nucleotides are in this RNA?", lambda r: str(S(r).count("C")), 1),
    ("How many U nucleotides are in this RNA?", lambda r: str(S(r).count("U")), 1),
    ("How many times does the dinucleotide UG occur in this RNA?",
     lambda r: str(S(r).count("UG")), 2),
    ("How many times does the dinucleotide AA occur in this RNA?",
     lambda r: str(S(r).count("AA")), 2),
    ("How many times does the dinucleotide GC occur in this RNA?",
     lambda r: str(S(r).count("GC")), 2),
    ("How many times does the dinucleotide CG occur in this RNA?",
     lambda r: str(S(r).count("CG")), 2),
    ("How many times does the dinucleotide CU occur in this RNA?",
     lambda r: str(S(r).count("CU")), 2),
    ("How many G nucleotides are in the first 50 nucleotides?",
     lambda r: str(S(r)[:50].count("G")), 50),
    ("How many distinct dinucleotides occur in this RNA?",
     lambda r: str(len({S(r)[i:i+2] for i in range(len(S(r))-1)})), 2),
    # 比例
    ("What is the GC content of this RNA?", lambda r: f"{gc(S(r)):.1f}%", 1),
    ("What percentage of this RNA is A? (to 1 decimal)",
     lambda r: pct(S(r), "A"), 1),
    ("What percentage of this RNA is U? (to 1 decimal)",
     lambda r: pct(S(r), "U"), 1),
    ("What percentage of this RNA is C? (to 1 decimal)",
     lambda r: pct(S(r), "C"), 1),
    ("What percentage of this RNA is G? (to 1 decimal)",
     lambda r: pct(S(r), "G"), 1),
    ("What percentage of this RNA is purines (A or G)? (to 1 decimal)",
     lambda r: f"{100.0*(S(r).count('A')+S(r).count('G'))/max(len(S(r)),1):.1f}%", 1),
    ("What percentage of this RNA is pyrimidines (C or U)? (to 1 decimal)",
     lambda r: f"{100.0*(S(r).count('C')+S(r).count('U'))/max(len(S(r)),1):.1f}%", 1),
    # 区域 GC
    ("What is the GC content of the first 50 nucleotides? (to 1 decimal)",
     lambda r: f"{gc(S(r)[:50]):.1f}%", 50),
    ("What is the GC content of the last 50 nucleotides? (to 1 decimal)",
     lambda r: f"{gc(S(r)[-50:]):.1f}%", 50),
    ("What is the GC content of the middle 50 nucleotides? (to 1 decimal)",
     lambda r: f"{gc(S(r)[len(S(r))//2-25:len(S(r))//2+25]):.1f}%", 60),
    # 结构性
    ("What is the length of the longest run of one repeated nucleotide?",
     lambda r: str(max((len(list(g)) for _, g in __import__('itertools').groupby(S(r))), default=0)), 1),
    ("What type of RNA is this?", lambda r: r.get("rna_type") or "", 1),
]


def screen(recs, tr, te, minlen, fn):
    cnt = Counter()
    for r in tr:
        if len(S(r)) < minlen:
            continue
        try:
            a = str(fn(r))
        except Exception:
            continue
        if a:
            cnt[a] += 1
    if not cnt:
        return None
    top = cnt.most_common(1)[0][0]
    hit = n = 0
    for r in te:
        if len(S(r)) < minlen:
            continue
        try:
            a = str(fn(r))
        except Exception:
            continue
        if not a:
            continue
        n += 1
        hit += (a == top)
    rate = hit / max(n, 1)
    p = [c / sum(cnt.values()) for c in cnt.values()]
    H = -sum(x * math.log(x) for x in p if x > 0)
    Hm = math.log(len(cnt))
    return dict(hit=rate, kinds=len(cnt), norm=H / Hm if Hm > 0 else 0.0)


def main() -> int:
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("sequence"):
                    recs.append(r)
    N = len(recs)
    cut = int(N * 0.8)
    tr, te = recs[:cut], recs[cut:]
    print(f"RNA {N}")

    rows = []
    for q, fn, ml in CAND:
        s = screen(recs, tr, te, ml, fn)
        if s is None:
            continue
        ok = s["hit"] <= 0.20 and s["norm"] > 0.5
        rows.append((q, fn, ml, s, ok))
        print(f"  {'收' if ok else '扔'}  {q[:58]:<58} 种类 {s['kinds']:>5} "
              f"归一熵 {s['norm']:.3f} 瞎猜 {100*s['hit']:>5.1f}%")

    keep = [r for r in rows if r[4]]
    keep.sort(key=lambda x: x[3]["hit"])          # 瞎猜命中最低的优先
    keep = keep[:30]
    print(f"\n候选 {len(rows)}  通过 {sum(1 for r in rows if r[4])}  取 {len(keep)}")

    n_qa = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for r in recs:
            for q, fn, ml, s, _ in keep:
                if len(S(r)) < ml:
                    continue
                try:
                    a = str(fn(r))
                except Exception:
                    continue
                if not a:
                    continue
                f.write(json.dumps({
                    "uid": r["uid"], "sequence": r["sequence"],
                    "question": q, "answer": a, "template": q,
                    "source": "seq_template", "kind": "closed",
                }, ensure_ascii=False) + "\n")
                n_qa += 1

    print(f"[save] {OUT}   {n_qa:,} 条   {OUT.stat().st_size/1024/1024:.1f} MB")
    print(f"       {N} RNA x {len(keep)} 模板 = {N*len(keep):,}（受最小长度限制，实际 {n_qa:,}）")
    (ROOT / "reports/template_final30.json").write_text(json.dumps(
        [dict(q=q, minlen=ml, **{k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in s.items()}) for q, _, ml, s, _ in keep],
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
