"""看一眼生成的 QA：类型分布 + 样例（调试用）。"""

import collections
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\qwen-api\RNAQwen\data\processed\_qa_deepseek_test.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
kinds, variants = collections.Counter(), collections.Counter()
for r in rows:
    for q in r["qa"]:
        kinds[q.get("kind")] += 1
        variants[q.get("variant")] += 1
print("kind 分布   :", dict(kinds))
print("variant 分布:", dict(variants))
print("记录数      :", len(rows), " QA 总数:", sum(kinds.values()))
print()
shown = 0
for r in rows:
    for q in r["qa"]:
        if q.get("kind") != "open_ended" and shown < 6:
            print(f"[{q.get('kind')}] Q: {q['question'][:140]}")
            print(f"           A: {q['answer'][:140]}")
            shown += 1
