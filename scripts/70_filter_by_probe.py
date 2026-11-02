"""第 1 步：按码信道的探针准确率筛模板。

依据：§10 的 code_answer_probe.json —— 每个模板在「只看码」的线性探针上的准确率。
30 个模板里只有 8 个的答案信息确实存在于码里（>= 0.40），其余 22 个是
0.000 - 0.149。

为什么这一步必须先做：那 22 个模板要求模型做码里没有信息的事。它们贡献的不是
「学不会」的信号，而是「别用 RNA」的信号 —— 而后者会泛化到那 8 个能用的模板上。
表现出来就是「模型不用 RNA」，看起来像 LLM 侧瓶颈。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/rnaqa_seq30_split.jsonl"
PROBE = ROOT / "reports/code_answer_probe.json"
OUT = ROOT / "data/processed/rnaqa_seq8.jsonl"
THRESH = 0.40

probe = json.loads(PROBE.read_text(encoding="utf-8"))
acc = {t["question"]: t["probe_acc"] for t in probe["per_template"]}
print(f"探针文件  {PROBE.name}")
print(f"           ckpt={probe['ckpt']}  pool={probe['pool']}  "
      f"n_test={probe['n_test']}  overall={probe['probe_overall']:.4f}")

keep = sorted([q for q, a in acc.items() if a >= THRESH], key=lambda q: -acc[q])
drop = sorted([q for q, a in acc.items() if a < THRESH], key=lambda q: -acc[q])
print()
print(f"=== 保留 {len(keep)} 个（probe_acc >= {THRESH}）===")
for q in keep:
    print(f"  {acc[q]:.3f}  {q}")
print()
print(f"=== 丢掉 {len(drop)} 个 ===")
for q in drop:
    print(f"  {acc[q]:.3f}  {q}")

n_in = n_out = 0
sp = Counter()
uids = set()
byq = Counter()
with open(OUT, "w", encoding="utf-8") as f:
    for line in open(SRC, encoding="utf-8"):
        r = json.loads(line)
        n_in += 1
        if r["question"] not in keep:
            continue
        f.write(line if line.endswith("\n") else line + "\n")
        n_out += 1
        sp[r.get("split", "?")] += 1
        byq[r["question"]] += 1
        uids.add(r["uid"])

print()
print("=" * 70)
print(f"  {SRC.name}   {n_in:,} 条 / 30 模板")
print(f"  -> {OUT.name}   **{n_out:,} 条 / {len(keep)} 模板**   ({100*n_out/n_in:.1f}%)")
print(f"  RNA {len(uids):,}")
print(f"  split {dict(sp)}")
print()
print("  每个模板的条数:")
for q, c in byq.most_common():
    print(f"    {c:>6,}  {q}")
print("=" * 70)
