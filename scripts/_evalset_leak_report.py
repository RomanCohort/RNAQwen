"""评测集的**跨 split 泄漏报告**：真实值 + 随机零模型基线 + 同源簇。

为什么需要它
------------
`14_build_evalset.py` 自己报的泄漏数有两个问题：
  1. `if len(leaks) < 20` 把计数**截断**成 20 —— 打印出来的从来不是真实数量；
  2. 只给数字不给**基线**。而 k 取多少直接决定这个数字有没有意义：
     12-mer 空间 4^12=1.7e7，本语料就有 ~1e6 个 12-mer，随机重排都能撞出几百个。
     **没有基线的泄漏数字是不可解释的。**

所以这里一次给出三样东西（k 可调）：
  * `real`  ：真实序列上的跨 split 泄漏 k-mer 数 / 涉及记录对 / 涉及记录数
  * `null`  ：把每条序列随机重排（**长度和 split 保持不变**）后重测 N 次
  * `homolog`：按 k-mer 集合 Jaccard >= 阈值做并查集聚类，报同源簇规模
    （MIR9-1/MIR9-2 这种真同源，Jaccard=0.4，是**真的**会让分数虚高，必须单独量）

只读序列和 split，不改任何划分协议。
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.guard import normalize_sequence  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402


def load_evalset_module():
    spec = importlib.util.spec_from_file_location("ev14", ROOT / "scripts" / "14_build_evalset.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_rows(path: Path, ev):
    rows, seen = [], set()
    for r in read_jsonl(path):
        s = normalize_sequence(r.sequence)
        if s in seen:
            continue
        seen.add(s)
        rows.append({"uid": r.uid, "ent_id": (r.provenance or {}).get("ent_id", ""),
                     "type": r.rna_type, "seq": s,
                     "split": ev.split_of(r.nuc_accession or r.uid)})
    return rows


def kmer_sets(rows, k):
    return [{s["seq"][j:j + k] for j in range(max(0, len(s["seq"]) - k + 1))} for s in rows]


def leak_stats(rows, ksets, k):
    owner: dict[str, set[str]] = collections.defaultdict(set)
    kmer_recs: dict[str, list[int]] = collections.defaultdict(list)
    for i, ks in enumerate(ksets):
        for km in ks:
            owner[km].add(rows[i]["split"])
            kmer_recs[km].append(i)
    leaking = {km: sp for km, sp in owner.items() if len(sp) > 1}
    pairs: set[tuple[int, int]] = set()
    for km in leaking:
        idx = kmer_recs[km]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                if rows[ia]["split"] != rows[ib]["split"]:
                    pairs.add((min(ia, ib), max(ia, ib)))
    involved = {i for p in pairs for i in p}
    return {
        "k": k,
        "total_kmers": sum(len(s) for s in ksets),
        "leaking_kmers": len(leaking),
        "leaking_pairs": len(pairs),
        "records_involved": len(involved),
        "pairs": pairs,
    }


def clusters_at(rows, ksets, min_jaccard, min_shared):
    """按 k-mer 集合的 Jaccard 做并查集，找同源簇。"""
    inv: dict[str, list[int]] = collections.defaultdict(list)
    for i, ks in enumerate(ksets):
        for km in ks:
            inv[km].append(i)
    shared: collections.Counter = collections.Counter()
    for km, idx in inv.items():
        if len(idx) < 2:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                shared[(idx[a], idx[b])] += 1

    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges = []
    for (a, b), n in shared.items():
        if n < min_shared:
            continue
        j = n / max(len(ksets[a]) + len(ksets[b]) - n, 1)
        if j >= min_jaccard:
            union(a, b)
            edges.append({"a": a, "b": b, "shared": n, "jaccard": round(j, 3),
                          "same_split": rows[a]["split"] == rows[b]["split"]})

    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    cl = [g for g in groups.values() if len(g) > 1]
    cl.sort(key=len, reverse=True)
    return {
        "min_jaccard": min_jaccard, "min_shared_kmers": min_shared,
        "n_clusters": len(cl),
        "n_records_in_clusters": sum(len(g) for g in cl),
        "largest_cluster": len(cl[0]) if cl else 0,
        "cross_split_clusters": sum(
            1 for g in cl if len({rows[i]["split"] for i in g}) > 1),
        "cross_split_detail": [
            {"size": len(g), "splits": sorted({rows[i]["split"] for i in g}),
             "members": [{"uid": rows[i]["uid"], "ent_id": rows[i]["ent_id"],
                          "type": rows[i]["type"], "n_nt": len(rows[i]["seq"]),
                          "split": rows[i]["split"]} for i in g]}
            for g in cl if len({rows[i]["split"] for i in g}) > 1
        ][:20],
        "examples": [
            {"size": len(g),
             "splits": sorted({rows[i]["split"] for i in g}),
             "members": [{"uid": rows[i]["uid"], "ent_id": rows[i]["ent_id"],
                          "type": rows[i]["type"], "n_nt": len(rows[i]["seq"]),
                          "split": rows[i]["split"]} for i in g[:8]]}
            for g in cl[:5]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--ks", default="12,20")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "evalset_leak_report.json")
    args = ap.parse_args()

    ev = load_evalset_module()
    rows = load_rows(args.inp, ev)
    print(f"[leak] {len(rows)} 条唯一序列；split 分布 "
          f"{dict(collections.Counter(r['split'] for r in rows))}")

    report = {"records": len(rows),
              "split_counts": dict(collections.Counter(r["split"] for r in rows)),
              "input": str(args.inp), "by_k": {}}

    for k in [int(x) for x in args.ks.split(",")]:
        ksets = kmer_sets(rows, k)
        real = leak_stats(rows, ksets, k)
        rng = random.Random(args.seed)
        nulls = []
        for _ in range(args.trials):
            rnd_rows = [{"split": r["split"],
                         "seq": "".join(rng.choice("ACGU") for _ in range(len(r["seq"])))}
                        for r in rows]
            nulls.append(leak_stats(rnd_rows, kmer_sets(rnd_rows, k), k)["leaking_kmers"])
        # 也报"结构打乱但保留碱基组成"的版本，避免随机串 GC 含量不同带来的偏差
        # （随机 25% 各碱基 vs 真实序列的碱基组成不同）。这里用逐序列洗牌。
        rng2 = random.Random(args.seed + 1)
        shuffles = []
        for _ in range(args.trials):
            sh_rows = []
            for r in rows:
                chars = list(r["seq"])
                rng2.shuffle(chars)
                sh_rows.append({"split": r["split"], "seq": "".join(chars)})
            shuffles.append(leak_stats(sh_rows, kmer_sets(sh_rows, k), k)["leaking_kmers"])

        entry = {
            "real": {kk: vv for kk, vv in real.items() if kk != "pairs"},
            "null_random_uniform": nulls,
            "null_null_avg": round(sum(nulls) / len(nulls), 1),
            "null_shuffled_same_composition": shuffles,
            "null_shuffled_avg": round(sum(shuffles) / len(shuffles), 1),
            "alphabet_space": 4 ** k,
        }
        if k == 20:
            entry["homolog_clusters"] = clusters_at(rows, ksets, args.jaccard, 5)
            # 阈值敏感性：同源簇怎么随 Jaccard 阈值变，决定要不要在报告里声明"已知跨 split 同源"
            entry["homolog_threshold_sweep"] = {
                str(th): {kk: vv for kk, vv in clusters_at(rows, ksets, th, 5).items()
                          if kk in ("n_clusters", "n_records_in_clusters",
                                    "largest_cluster", "cross_split_clusters")}
                for th in (0.2, 0.3, 0.5, 0.7)
            }
        report["by_k"][str(k)] = entry

        print(f"\n[k={k}] 空间 4^{k}={4**k:.3e}")
        print(f"  真实泄漏 k-mer      : {real['leaking_kmers']}")
        print(f"  涉及记录对 / 记录数 : {real['leaking_pairs']} / {real['records_involved']}")
        print(f"  随机重排基线(同长度): {nulls}  均值 {sum(nulls)/len(nulls):.1f}")
        print(f"  同组成打乱基线      : {shuffles}  均值 {sum(shuffles)/len(shuffles):.1f}")
        if k == 20:
            hc = entry["homolog_clusters"]
            print(f"  同源簇 Jaccard>={args.jaccard}: {hc['n_clusters']} 簇 / "
                  f"{hc['n_records_in_clusters']} 条 / 最大 {hc['largest_cluster']} / "
                  f"跨 split {hc['cross_split_clusters']} 簇")
            for e in hc["examples"][:3]:
                print(f"    size={e['size']} splits={e['splits']} "
                      f"{[m['ent_id'] for m in e['members']]}")
            print("  阈值敏感性:", json.dumps(entry["homolog_threshold_sweep"],
                                             ensure_ascii=False))
            print("  --- 跨 split 的同源簇（这些才是会灌水的）---")
            for c in hc["cross_split_detail"]:
                print(f"    size={c['size']} splits={c['splits']} "
                      f"{[(m['ent_id'], m['split'], m['n_nt']) for m in c['members']]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[leak] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
