#!/usr/bin/env python
"""按家族平衡采样后再生成 Rfam QA。

上一轮被否的原因：语料里 rRNA 占 51%，导致 RF00177 一个家族就占 43.4%，
永远答它就有 43.4% 的分数。

平衡采样：每个家族最多取 PER_FAM 条。这样答案分布接近均匀，
瞎猜命中率降到 ~1/家族数，模型必须真读序列。

同时**均衡必须对训练和测试都做**，否则评测口径不一致。
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "data/external/rfam/hits32.tbl"
OUT = ROOT / "data/processed/rnaqa_rfam.jsonl"
PER_FAM = 40
SEED = 20261010

best = {}
with open(TBL, encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        c = line.rstrip("\n").split(None, 18)
        if len(c) < 18:
            continue
        try:
            ev = float(c[15])
        except ValueError:
            continue
        uid = c[2].split(":", 1)[1] if ":" in c[2] else c[2]
        cur = best.get(uid)
        if cur is None or ev < cur[0]:
            best[uid] = (ev, c[1], c[0], c[18] if len(c) > 18 else "")

print(f"唯一序列 {len(best):,}   原始家族数 {len({v[1] for v in best.values()}):,}")

byfam = defaultdict(list)
for uid, rec in best.items():
    byfam[rec[1]].append(uid)

rng = random.Random(SEED)
picked = []
for fam, uids in byfam.items():
    rng.shuffle(uids)
    picked.extend(uids[:PER_FAM])
print(f"平衡采样（每家族 <= {PER_FAM}）: {len(picked):,} 条序列   "
      f"家族 {len(byfam):,}")

n = len(picked)
fams = Counter(best[u][1] for u in picked)
top = fams.most_common(5)
blind = top[0][1] / n
H = -sum((c / n) * math.log(c / n) for c in fams.values())
Hm = math.log(len(fams)) if len(fams) > 1 else 1.0
print()
print(f"  最常见家族: " + ", ".join(f"{k} {100*c/n:.2f}%" for k, c in top))
print(f"  **瞎猜命中率 = {100*blind:.3f}%**")
print(f"  熵 {H:.2f} / 上限 {Hm:.2f}  (归一 {H/Hm:.3f})")
print()
ok = blind <= 0.02
print("  ✓ 合格" if ok else f"  ✗ 仍不合格（{100*blind:.2f}%）")

if not ok:
    raise SystemExit(1)

# ---- 划分（按家族，保证测试集有没见过的家族吗？不 —— 家族要共享，序列才不共享）
# 这里的泄漏风险是"同一序列"或"近亲"。按 uid 随机分即可，因为每条序列只出现一次；
# 但近亲序列可能落在不同 split。用 57_ 的聚类结果过滤。
rng2 = random.Random(SEED + 1)
order = list(picked)
rng2.shuffle(order)
ntr = int(0.8 * len(order))
split = {}
for u in order[:ntr]:
    split[u] = "train"
for u in order[ntr:]:
    split[u] = "test"

n_qa = 0
with open(OUT, "w", encoding="utf-8") as f:
    for u in picked:
        ev, acc, nm, desc = best[u]
        for q, a in (("Which Rfam family does this RNA belong to?", acc),
                     ("What type of RNA family does this RNA belong to?", nm)):
            if not a:
                continue
            f.write(json.dumps({
                "uid": u, "question": q, "answer": a, "template": q,
                "source": "rfam", "kind": "closed", "split": split[u],
                "evalue": ev, "family_desc": desc,
            }, ensure_ascii=False) + "\n")
            n_qa += 1

sc = Counter(split[u] for u in picked)
print(f"\n[save] {OUT}")
print(f"  {n_qa:,} 条   train {sc['train']:,} / test {sc['test']:,}")
