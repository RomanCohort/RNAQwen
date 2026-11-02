"""把 993 条新数据并进现有训练集。

合并前必须核对的（少一条都会静默出错）：
  1. 字段结构一致（uid / sequence / annotation / qa / rna_type）
  2. qa 的元素结构一致（question / answer / kind / variant）
  3. 没有 uid 或**序列**重复（序列重复会让 train/test 泄漏）
  4. 逐条统计，别只看总数
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
OLD = ROOT / "data/processed/rnaqa_n6.jsonl"
NEW = ROOT / "data/processed/rnaqa_1k_new.jsonl"
OUT = ROOT / "data/processed/rnaqa_n6_merged.jsonl"


def load(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


old = load(OLD)
new = load(NEW)
print("现有 %d 条 / 新 %d 条" % (len(old), len(new)))

ko, kn = set(old[0].keys()), set(new[0].keys())
print("\n字段对比:")
print("  共有:", sorted(ko & kn))
print("  仅现有:", sorted(ko - kn))
print("  仅新  :", sorted(kn - ko))

qo = old[0]["qa"][0]
qn = new[0]["qa"][0]
print("\nqa 元素字段:")
print("  现有:", sorted(qo.keys()))
print("  新  :", sorted(qn.keys()))

# 去重
old_uids = {r.get("uid") for r in old}
old_seqs = {(r.get("sequence") or "")[:60] for r in old}
dup_uid = sum(1 for r in new if r.get("uid") in old_uids)
dup_seq = sum(1 for r in new if (r.get("sequence") or "")[:60] in old_seqs)
print("\n新数据里与现有重复: uid %d, 序列 %d" % (dup_uid, dup_seq))
keep = [r for r in new if (r.get("sequence") or "")[:60] not in old_seqs]
print("实际并入 %d 条" % len(keep))

merged = old + keep
print("\n合并后:")
n_qa = sum(len(r.get("qa") or []) for r in merged)
print("  RNA %d   QA %d   QA/RNA %.2f" % (len(merged), n_qa, n_qa / len(merged)))
vc, kc = Counter(), Counter()
for r in merged:
    for q in r.get("qa") or []:
        vc[q.get("variant", "?")] += 1
        kc[q.get("kind", "?")] += 1
print("  variant", dict(vc))
print("  kind   ", dict(kc))

# 空值检查
bad = sum(1 for r in merged if not (r.get("annotation") or "").strip())
print("  空 annotation: %d" % bad)
lens = sorted(len(r.get("sequence") or "") for r in merged)
print("  序列长度 min %d median %d max %d" % (lens[0], lens[len(lens)//2], lens[-1]))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in merged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\n[ok] -> %s" % OUT)
print("论文目标: 5K RNA / 121K QA   ->  现在 %.0f%% / %.0f%%"
      % (100.0 * len(merged) / 5000, 100.0 * n_qa / 121000))
