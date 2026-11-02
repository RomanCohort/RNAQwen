"""扩模板：开放式高基数候选，全部过一遍瞎猜基线筛。

上一轮结论：是/否 模板几乎全废（RNA 统计偏斜，押多数类就能拿 50-100%）。
所以要的是**答案空间大、分布接近均匀、没有多数类可押**的模板。

判据：瞎猜命中 < 20%  且  归一化熵 > 0.5
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
RC = str.maketrans("ACGU", "UGCA")


def S(r):
    return r["sequence"].upper()


def gc(s):
    return 100.0 * (s.count("G") + s.count("C")) / max(len(s), 1)


def longest_run(s):
    if not s:
        return 0
    best = cur = 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        best = max(best, cur)
    return best


# (问题, 取值, 最小序列长度要求)
CAND = [
    # --- 子序列：答案空间巨大 ---
    ("What are the first 6 nucleotides of this RNA?", lambda r: S(r)[:6], 6),
    ("What are the last 6 nucleotides of this RNA?", lambda r: S(r)[-6:], 6),
    ("What are the first 4 nucleotides of this RNA?", lambda r: S(r)[:4], 4),
    ("What is the reverse complement of the first 4 nucleotides?",
     lambda r: S(r)[:4].translate(RC)[::-1], 4),
    # --- 位置特异：单碱基但分布均匀 ---
    ("What is the 10th nucleotide of this RNA?", lambda r: S(r)[9], 10),
    ("What is the 20th nucleotide of this RNA?", lambda r: S(r)[19], 20),
    ("What is the 30th nucleotide of this RNA?", lambda r: S(r)[29], 30),
    ("What is the nucleotide at the middle of this RNA?",
     lambda r: S(r)[len(S(r)) // 2], 1),
    # --- 计数型：答案空间中等但均匀 ---
    ("How many G nucleotides are in this RNA?", lambda r: str(S(r).count("G")), 1),
    ("How many A nucleotides are in this RNA?", lambda r: str(S(r).count("A")), 1),
    ("How many times does the dinucleotide UG occur in this RNA?",
     lambda r: str(S(r).count("UG")), 2),
    ("How many times does the dinucleotide AA occur in this RNA?",
     lambda r: str(S(r).count("AA")), 2),
    ("What is the length of the longest run of one repeated nucleotide?",
     lambda r: str(longest_run(S(r))), 1),
    # --- 区域 GC：比全长 GC 更细 ---
    ("What is the GC content of the first 50 nucleotides? (to 1 decimal)",
     lambda r: f"{gc(S(r)[:50]):.1f}%", 50),
    ("What is the GC content of the last 50 nucleotides? (to 1 decimal)",
     lambda r: f"{gc(S(r)[-50:]):.1f}%", 50),
    # --- 比例型 ---
    ("What percentage of this RNA is U? (to 1 decimal)",
     lambda r: f"{100.0*S(r).count('U')/max(len(S(r)),1):.1f}%", 1),
    ("What percentage of this RNA is purines (A or G)? (to 1 decimal)",
     lambda r: f"{100.0*(S(r).count('A')+S(r).count('G'))/max(len(S(r)),1):.1f}%", 1),
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
    cut = int(N * 0.8)
    tr, te = recs[:cut], recs[cut:]
    print(f"RNA {N}   训练段 {len(tr)}   测试段 {len(te)}")
    print()
    print("=" * 100)
    print(f"{'模板':<56} {'覆盖':>6} {'种类':>6} {'归一熵':>7} {'瞎猜':>7} {'判定':>8}")
    print("=" * 100)

    rows = []
    for q, fn, minlen in CAND:
        cover = sum(1 for r in recs if len(r["sequence"]) >= minlen)
        cnt_tr = Counter()
        for r in tr:
            if len(r["sequence"]) < minlen:
                continue
            try:
                a = str(fn(r))
            except Exception:
                continue
            if a:
                cnt_tr[a] += 1
        if not cnt_tr:
            continue
        top = cnt_tr.most_common(1)[0][0]
        hit = n = 0
        for r in te:
            if len(r["sequence"]) < minlen:
                continue
            try:
                a = str(fn(r))
            except Exception:
                continue
            if not a:
                continue
            n += 1
            hit += (a == top)
        hit_rate = hit / max(n, 1)
        p = [c / sum(cnt_tr.values()) for c in cnt_tr.values()]
        H = -sum(x * math.log(x) for x in p if x > 0)
        Hmax = math.log(len(cnt_tr))
        norm = H / Hmax if Hmax > 0 else 0.0

        if hit_rate <= 0.20 and norm > 0.5:
            v = "收"
        elif hit_rate <= 0.30:
            v = "勉强"
        else:
            v = "扔"
        rows.append((q, cover / N, len(cnt_tr), norm, hit_rate, v))
        print(f"  {q[:54]:<54} {100*cover/N:>5.0f}% {len(cnt_tr):>6} "
              f"{norm:>7.3f} {100*hit_rate:>6.1f}% {v:>8}")

    print()
    keep = [r for r in rows if r[5] == "收"]
    print("=" * 100)
    print(f"收 {len(keep)} / {len(rows)}")
    print("=" * 100)
    for q, cov, k, ne, h, _ in keep:
        print(f"  收: {q[:60]:<62} 瞎猜 {100*h:.1f}%  覆盖 {100*cov:.0f}%")

    (ROOT / "reports/template_screen_v2.json").write_text(json.dumps(
        [dict(q=q, coverage=round(c, 3), kinds=k, norm_entropy=round(ne, 4),
              blind_hit=round(h, 4), verdict=v) for q, c, k, ne, h, v in rows],
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] reports/template_screen_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
