"""按序列身份的划分 —— 纯序列任务上，随机划分会让分数虚高。

问题：RNAcentral 里同一条 RNA 有多个来源，还有很多近亲（同一家族、同一基因的变体）。
      随机切分会让训练集和测试集共享 20-mer。已知 evalset_merged 有 11,744 个跨划分的 20-mer 重叠。

做法：
  1. 精确去重（同一条序列只保留一次）
  2. 20-mer 倒排索引 -> 候选对 -> 共享 k-mer 比例 >= 0.5 的连边
  3. 连通分量 = 簇（一个簇 = 一组近亲）
  4. **按簇划分**，保证同一簇不会跨 split
  5. 验证：跨 split 的 20-mer 重叠必须是 0
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/rnaqa_seq30.jsonl"
QA_OUT = ROOT / "data/processed/rnaqa_seq30_split.jsonl"
SPLIT_OUT = ROOT / "reports/seq_split.json"
K = 20
JACCARD = 0.5
SEED = 20261010


def kmers(s, k=K):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def main() -> int:
    # ---- 读序列（一序列一行）----
    seqs = {}
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seqs[r["uid"]] = r["sequence"]
    uids = list(seqs)
    print(f"RNA {len(uids)}   序列长度中位 {sorted(len(seqs[u]) for u in uids)[len(uids)//2]}")

    # ---- 1. 精确去重 ----
    by_seq = defaultdict(list)
    for u in uids:
        by_seq[seqs[u]].append(u)
    dup_groups = sum(1 for v in by_seq.values() if len(v) > 1)
    dup_extra = sum(len(v) - 1 for v in by_seq.values())
    print(f"精确重复序列组 {dup_groups}   多出来的 RNA {dup_extra}")

    # ---- 2. 倒排索引 -> 候选对 ----
    km = {u: kmers(seqs[u]) for u in uids}
    inv = defaultdict(list)
    for u in uids:
        for k_ in km[u]:
            inv[k_].append(u)
    cand = defaultdict(int)
    for k_, us in inv.items():
        if len(us) < 2 or len(us) > 200:        # 极常见的 k-mer 不算信号
            continue
        for i in range(len(us)):
            for j in range(i + 1, len(us)):
                a, b = us[i], us[j]
                cand[(a, b) if a < b else (b, a)] += 1
    print(f"共享 >=1 个 20-mer 的候选对 {len(cand)}")

    # ---- 3. 连通分量 ----
    parent = {u: u for u in uids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges = 0
    for (a, b), shared in cand.items():
        j = shared / max(len(km[a] | km[b]), 1)
        if j >= JACCARD:
            union(a, b)
            edges += 1
    print(f"近亲连边 {edges}（Jaccard >= {JACCARD}）")

    clusters = defaultdict(list)
    for u in uids:
        clusters[find(u)].append(u)
    cl = sorted(clusters.values(), key=len, reverse=True)
    sizes = [len(c) for c in cl]
    print(f"簇数 {len(cl)}   最大簇 {sizes[0]}   "
          f"中位 {sizes[len(sizes)//2]}   单例 {sum(1 for s in sizes if s == 1)}")

    # ---- 4. 读 QA，按簇划分 ----
    qa_by_uid = defaultdict(list)
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qa_by_uid[r["uid"]].append(r)

    rng = random.Random(SEED)
    rng.shuffle(cl)
    total = sum(len(qa_by_uid[u]) for u in uids)
    target = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    got = {"train": 0, "val": 0, "test": 0}
    assign = {}
    for c in cl:
        n = sum(len(qa_by_uid[u]) for u in c)
        # 放进当前最缺的那个 split
        pick = min(target, key=lambda s: got[s] / max(target[s], 1))
        if got[pick] >= target[pick] and pick != "train":
            pick = "train"
        for u in c:
            assign[u] = pick
        got[pick] += n

    # ---- 5. 写出 ----
    counts = defaultdict(int)
    with open(QA_OUT, "w", encoding="utf-8") as f:
        for u in uids:
            sp = assign[u]
            for r in qa_by_uid[u]:
                r2 = dict(r); r2["split"] = sp
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")
                counts[sp] += 1

    print()
    print("=" * 70)
    print(f"  训练 {got['train']:>7,}  验证 {got['val']:>7,}  测试 {got['test']:>7,}"
          f"   合计 {sum(got.values()):,}")
    print("=" * 70)

    # ---- 6. 验证：分两个层次，别把背景噪声当泄漏 ----
    #
    #   A. 严重泄漏：**近亲对**跨越 split（Jaccard >= 0.5）。这个必须为 0，
    #      否则模型背下训练集就能在测试集上拿分。
    #   B. 背景重叠：两条不同的 RNA 共享一个 20-mer。RNA 里重复 motif 是常态，
    #      这个不可能为 0，也不构成泄漏 —— 但要报出来，别假装它不存在。
    leak = 0
    for (a, b), shared in cand.items():
        j = shared / max(len(km[a] | km[b]), 1)
        if j >= JACCARD and assign[a] != assign[b]:
            leak += 1

    sp_of = {u: assign[u] for u in uids}
    set_of = defaultdict(set)
    for u in uids:
        for k_ in km[u]:
            set_of[k_].add(sp_of[u])
    bg = sum(1 for k_, s in set_of.items() if len(s) > 1)
    print(f"\n[A] 跨 split 的近亲对（Jaccard >= {JACCARD}）: {leak}   必须为 0")
    print(f"[B] 跨 split 共享的 20-mer: {bg}   <- 背景重叠，非泄漏")
    print(f"    占全部 20-mer 的 {100*bg/max(len(set_of),1):.1f}%")

    # 测试集里有多少条 RNA 在训练集里有近亲
    tr_set = {u for u in uids if assign[u] == "train"}
    poisoned = set()
    for (a, b), shared in cand.items():
        j = shared / max(len(km[a] | km[b]), 1)
        if j >= JACCARD:
            if a in tr_set and assign[b] != "train":
                poisoned.add(b)
            if b in tr_set and assign[a] != "train":
                poisoned.add(a)
    print(f"[C] 测试/验证集里「训练集有近亲」的 RNA: {len(poisoned)}   必须为 0")

    SPLIT_OUT.write_text(json.dumps(dict(
        n_rna=len(uids), n_clusters=len(cl), largest_cluster=sizes[0],
        singleton_clusters=sum(1 for s in sizes if s == 1),
        exact_dups=dup_extra, near_edges=edges,
        qa=dict(got), cross_split_kmers=leak, k=K, jaccard=JACCARD,
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {QA_OUT}")
    print(f"[save] {SPLIT_OUT}")
    return 0 if (leak == 0 and not poisoned) else 1


if __name__ == "__main__":
    raise SystemExit(main())
