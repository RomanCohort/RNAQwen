"""诊断 `14_build_evalset.py` 报的跨 split 12-mer 泄漏：**真有多少、是谁和谁**。

为什么要另写一个
----------------
`14_build_evalset.py` 里 `leaks` 列表被 `if len(leaks) < 20` 截断，
所以它打印的"20 处"是**上限**，不是真实数量。要判断严重程度必须自己数一遍。
本脚本**不改 14 的任何口径**：直接 import 它的 `split_of()`，保证划分完全一致。

要回答的问题
------------
1. 真实泄漏 k-mer 有多少个？（不是截断后的 20）
2. 涉及多少条记录、多少对记录？
3. 根因是什么 —— 是"同一条序列被分到两边"（那 impossible，14 已按序列去重），
   还是"同源 RNA（同一 miRNA 在不同物种 / 同一家族）"？
   判定方法：看泄漏记录对之间的**序列相似度**和 **ent_id**。
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.guard import normalize_sequence  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402

K = 12


def load_evalset_module():
    spec = importlib.util.spec_from_file_location("ev14", ROOT / "scripts" / "14_build_evalset.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "evalset_leak_diagnosis.json")
    ap.add_argument("--pairs", type=int, default=8)
    args = ap.parse_args()

    ev = load_evalset_module()
    recs = list(read_jsonl(args.inp))
    rows = []
    seen = set()
    for r in recs:
        seq = normalize_sequence(r.sequence)
        if seq in seen:
            continue
        seen.add(seq)
        rows.append({
            "uid": r.uid,
            "ent_id": (r.provenance or {}).get("ent_id", ""),
            "type": r.rna_type,
            "seq": seq,
            "split": ev.split_of(r.nuc_accession or r.uid),
        })

    # kmer -> {split: [record idx]}
    owner: dict[str, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, row in enumerate(rows):
        for j in range(max(0, len(row["seq"]) - args.k + 1)):
            owner[row["seq"][j:j + args.k]][row["split"]].append(i)

    leaking_kmers = {km: d for km, d in owner.items() if len(d) > 1}
    pair_count: collections.Counter = collections.Counter()
    kmer_per_pair: collections.Counter = collections.Counter()
    for km, d in leaking_kmers.items():
        splits = sorted(d)
        for a_i in range(len(splits)):
            for b_i in range(a_i + 1, len(splits)):
                for ia in set(d[splits[a_i]]):
                    for ib in set(d[splits[b_i]]):
                        key = tuple(sorted((ia, ib)))
                        pair_count[key] += 1

    details = []
    for (ia, ib), nk in pair_count.most_common(args.pairs):
        ra, rb = rows[ia], rows[ib]
        # 最长公共前缀式的粗略相似度：共享 k-mer 比例
        ka = {ra["seq"][j:j + args.k] for j in range(max(0, len(ra["seq"]) - args.k + 1))}
        kb = {rb["seq"][j:j + args.k] for j in range(max(0, len(rb["seq"]) - args.k + 1))}
        jac = len(ka & kb) / max(len(ka | kb), 1)
        details.append({
            "shared_kmers": nk, "jaccard_12mer": round(jac, 3),
            "a": {"uid": ra["uid"], "ent_id": ra["ent_id"], "type": ra["type"],
                  "split": ra["split"], "n_nt": len(ra["seq"])},
            "b": {"uid": rb["uid"], "ent_id": rb["ent_id"], "type": rb["type"],
                  "split": rb["split"], "n_nt": len(rb["seq"])},
        })

    report = {
        "records": len(rows),
        "split_counts": dict(collections.Counter(r["split"] for r in rows)),
        "k": args.k,
        "leaking_kmers_true": len(leaking_kmers),
        "leaking_record_pairs": len(pair_count),
        "records_involved": len({i for pair in pair_count for i in pair}),
        "note": "14_build_evalset.py 打印的数字被 `len(leaks) < 20` 截断，只是上限",
        "top_pairs": details,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "top_pairs"},
                     indent=2, ensure_ascii=False))
    print("\n--- 泄漏最多的记录对 ---")
    for d in details:
        print(f"  {d['shared_kmers']:5d} 个共享 12-mer  jaccard={d['jaccard_12mer']}")
        print(f"     A {d['a']['ent_id']:20s} {d['a']['type']:10s} {d['a']['split']:5s} {d['a']['n_nt']:5d}nt {d['a']['uid']}")
        print(f"     B {d['b']['ent_id']:20s} {d['b']['type']:10s} {d['b']['split']:5s} {d['b']['n_nt']:5d}nt {d['b']['uid']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[leak] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
