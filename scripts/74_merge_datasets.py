"""把 seq8 + freetext + rfam 合成一份，划分口径统一。

三个来源的 uid 体系不同：
    rnaqa_seq8        urs:URS...|taxid:...     （4,481 条带注释 RNA，有 split）
    rnaqa_freetext    同上                     （无 split）
    rnaqa_rfam        裸 md5                    （19,726 条，有 split）

**划分必须全部按 `urs:...` 的身份来，不能各划各的** ——
freetext 和 seq8 覆盖的是同一批 4,481 条 RNA，如果 freetext 用另一套划分，
同一条 RNA 会同时出现在训练和测试里，分数直接虚高。

rfam 的 RNA 是另一批（145,201 里抽的），它自己带 split，直接沿用。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed"
OUT = D / "rnaqa_merged.jsonl"

# 1) 以 seq8 的 split 为准，建立 urs -> split
urs_split = {}
with open(D / "rnaqa_seq8.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        urs_split[r["uid"]] = r.get("split", "train")
print(f"[ref] seq8 提供 {len(urs_split):,} 个 urs 的划分 "
      f"{dict(Counter(urs_split.values()))}")

n_out = 0
src_count = Counter()
sp_count = Counter()
tpl_count = Counter()
dupe = Counter()
seen = set()

with open(OUT, "w", encoding="utf-8") as fo:

    def emit(path, src, sp_of, tpl_of):
        global n_out
        n_in = n_skip = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                n_in += 1
                sp = sp_of(r)
                if sp is None:
                    n_skip += 1
                    continue
                key = (r["uid"], r["question"])
                if key in seen:
                    n_skip += 1
                    continue
                seen.add(key)
                r["split"] = sp
                r["source"] = src
                tpl = tpl_of(r)
                r["template"] = tpl
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1
                src_count[src] += 1
                sp_count[sp] += 1
                tpl_count[tpl] += 1
        print(f"  {src:<10} 读 {n_in:,}  跳过 {n_skip:,}  写入 {n_in - n_skip:,}")

    # seq8：自带 split
    emit(D / "rnaqa_seq8.jsonl", "seq8",
         lambda r: r.get("split", "train"), lambda r: r["question"])

    # freetext：**用 seq8 的划分**（同一批 RNA）
    emit(D / "rnaqa_freetext.jsonl", "freetext",
         lambda r: urs_split.get(r["uid"]), lambda r: r["question"])

    # rfam：自带 split（另一批 RNA）
    emit(D / "rnaqa_rfam.jsonl", "rfam",
         lambda r: r.get("split", "train"), lambda r: r["question"])

print()
print("=" * 72)
print(f"  合并 {OUT.name}    **{n_out:,} 条**")
print(f"  来源 {dict(src_count)}")
print(f"  划分 {dict(sp_count)}")
print(f"  模板 {len(tpl_count)} 个")
for q, c in tpl_count.most_common():
    print(f"    {c:>7,}  {q[:62]}")
print("=" * 72)
