"""问题本身能决定答案吗？—— 「模型不读 RNA」到底是数据问题还是路由问题

若同一个问题在大量不同的 RNA 上出现，却对应大量不同的答案，那么
**不读 RNA 就不可能答对** —— 模型答不利落，是路由问题。
若同一个问题几乎总是同一个答案，那 RNA 本来就没用，不是模型的错。

这个判据不需要训练，几秒钟能出。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"

by_q = defaultdict(list)
n_qa = 0
n_rna = 0
with open(DATA, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("sequence"):
            continue
        n_rna += 1
        for qa in (r.get("qa") or []):
            q, a = qa.get("question"), qa.get("answer")
            if not q or not a:
                continue
            n_qa += 1
            by_q[q].append((r["uid"], a, qa.get("kind")))

print(f"RNA {n_rna}   QA {n_qa}   不同问题 {len(by_q)}")
print()

# 每个问题：涉及多少条 RNA、多少个不同答案
rows = []
for q, lst in by_q.items():
    uids = {u for u, _, _ in lst}
    ans = {a for _, a, _ in lst}
    rows.append((q, len(lst), len(uids), len(ans)))
rows.sort(key=lambda x: -x[1])

multi_rna = [r for r in rows if r[2] >= 2]
multi_rna_ans = [r for r in multi_rna if r[3] >= 2]

print("=" * 78)
print("同一个问题在多少条不同 RNA 上出现 / 对应多少种不同答案")
print("=" * 78)
print(f"  问题总数                {len(rows)}")
print(f"  出现在 ≥2 条 RNA 上的   {len(multi_rna)}  ({100*len(multi_rna)/len(rows):.1f}%)")
print(f"  其中答案也 ≥2 种的      {len(multi_rna_ans)}  "
      f"({100*len(multi_rna_ans)/max(len(multi_rna),1):.1f}% of 多变体)")
print()

n_qa_multi = sum(r[1] for r in multi_rna_ans)
print(f"  **这些「同一问题、多条 RNA、多个答案」的 QA 占 {n_qa_multi}/{n_qa} "
      f"= {100*n_qa_multi/n_qa:.1f}%**")
print()

# 答案的「唯一化比例」：同一问题下不同答案数 / 样本数
print("=" * 78)
print("问题对答案的决定力")
print("=" * 78)
print(f"{'问题(截断)':<52} {'样本':>5} {'RNA':>5} {'答案':>5} {'答案/样本':>10}")
for q, n, nu, na in rows[:15]:
    tag = "  <- 答案唯一" if na == 1 else ""
    print(f"  {q[:50]:<50} {n:>5} {nu:>5} {na:>5} {na/n:>10.3f}{tag}")

print()
# 全局：问题条件下的答案熵 vs 问题本身的熵
def entropy(counts):
    t = sum(counts)
    if t == 0:
        return 0.0
    return -sum((c / t) * math.log(c / t + 1e-12) for c in counts if c > 0)

h_ans_given_q = 0.0
n_ans_per_q = [len({a for _, a, _ in lst}) for lst in by_q.values()]
for q, lst in by_q.items():
    ac = defaultdict(int)
    for _, a, _ in lst:
        ac[a] += 1
    h_ans_given_q += (len(lst) / n_qa) * entropy(list(ac.values()))

all_ans = defaultdict(int)
for lst in by_q.values():
    for _, a, _ in lst:
        all_ans[a] += 1
h_ans = entropy(list(all_ans.values()))
h_q = entropy([len(lst) for lst in by_q.values()])

print("=" * 78)
print("信息量（nat）")
print("=" * 78)
print(f"  答案的总熵            H(A)      = {h_ans:.3f}")
print(f"  已知问题后答案的熵    H(A|Q)    = {h_ans_given_q:.3f}")
print(f"  **问题解释掉的答案方差 1-H(A|Q)/H(A) = "
      f"{1 - h_ans_given_q/max(h_ans,1e-9):.3f}**")
print()
print(f"  不同问题数 {len(by_q)}   不同答案数 {len(all_ans)}")
print()
print("=" * 78)
print("判定")
print("=" * 78)
frac = 1 - h_ans_given_q / max(h_ans, 1e-9)
if frac > 0.9:
    print(f"  ❌ 问题几乎决定了答案（解释了 {100*frac:.1f}% 的熵）。")
    print("     模型只读问题就能拿到大部分分数 —— 不读 RNA 是**理性的**，不是 bug。")
    print("     要让模型必须读 RNA，得改数据/任务，不是改模型。")
elif frac > 0.6:
    print(f"  ⚠️ 问题解释了 {100*frac:.1f}%，还有 {100*(1-frac):.1f}% 必须靠 RNA。")
    print("     模型现在吃掉了问题那部分、丢掉了 RNA 那部分。**是路由问题。**")
else:
    print(f"  ✅ 问题只解释了 {100*frac:.1f}%，答案主要靠 RNA 决定。")
    print("     那模型不读 RNA 就答不好 —— 纯路由问题，值得修。")
