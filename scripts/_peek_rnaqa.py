"""看 RNAcentral 路线的 QA 质量：按 variant 分开抽样。"""

import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\qwen-api\RNAQwen\data\processed\rnaqa_rnacentral.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
print(f"记录 {len(rows)} 条")

by = collections.Counter()
for r in rows:
    for q in r.get("qa") or []:
        by[(q.get("variant"), q.get("kind"))] += 1
print("(variant, kind) 分布:", dict(by))

for variant in ("dc", "ais"):
    print(f"\n{'='*72}\n{variant.upper()} 变体抽样（每条记录最多 2 组）\n{'='*72}")
    shown = 0
    for r in rows:
        qs = [q for q in (r.get("qa") or []) if q.get("variant") == variant]
        if not qs:
            continue
        print(f"\n[{r.get('rna_type')}] {r.get('uid')}  {len(r.get('sequence',''))} nt")
        print(f"  定义: {r.get('definition','')[:90]}")
        print(f"  官方注释: {(r.get('annotation') or '')[:180]}")
        for q in qs[:2]:
            print(f"  ({q.get('kind')}) Q: {q.get('question','')[:130]}")
            print(f"            A: {q.get('answer','')[:130]}")
        shown += 1
        if shown >= 3:
            break

# 抽检：问题里出现"研究/作者/方法"这类元信息的比例（AIS 变体的已知风险）
import re
META = re.compile(r"\b(study|studies|authors?|paper|this (?:research|work|article)|participants?|patients? were|sample size)\b", re.I)
for variant in ("dc", "ais"):
    qs = [q for r in rows for q in (r.get("qa") or []) if q.get("variant") == variant]
    if not qs:
        continue
    n_meta = sum(1 for q in qs if META.search(q.get("question", "")))
    print(f"\n[{variant}] 问题里带「研究/作者/样本」等元信息的比例: {n_meta}/{len(qs)} = {100*n_meta/len(qs):.1f}%")
