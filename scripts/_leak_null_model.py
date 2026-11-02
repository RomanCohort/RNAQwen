"""零模型：把序列换成**随机 ACGU 串**（长度和 split 保持不变），再跑同一个 12-mer 泄漏检查。

要回答的问题
------------
`14_build_evalset.py` 报跨 split 12-mer 泄漏。这到底是
  (a) 数据里真有同源/近重复序列（生物学问题），还是
  (b) **12-mer 这个尺子太细**，在几千条序列的规模上光靠概率就会大量撞车（度量问题）？

做法
----
保持每条记录的长度和 split 划分完全不变，只把碱基随机重排，重跑同一套检测。
如果随机串也能撞出同量级的泄漏，那 (b) 成立 —— 这不是数据脏，是判据在
4^12 = 1.6e7 的空间里对 10^6 量级的 k-mer 数不够判别力。

注意：**不改任何口径**，`14` 照原样跑；这里只是给它的报警数字配一个基线。
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


def count_leaks(seqs: list[tuple[str, str]], k: int) -> tuple[int, int]:
    """seqs = [(split, sequence)]；返回 (泄漏 k-mer 数, 涉及的记录对数)。"""
    owner: dict[str, set[str]] = collections.defaultdict(set)
    for split, s in seqs:
        for j in range(max(0, len(s) - k + 1)):
            owner[s[j:j + k]].add(split)
    leaking = {km: sp for km, sp in owner.items() if len(sp) > 1}
    return len(leaking), len(owner)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--trials", type=int, default=3, help="随机重排跑几遍取平均")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "evalset_leak_null_model.json")
    args = ap.parse_args()

    ev = load_evalset_module()
    recs = list(read_jsonl(args.inp))
    real: list[tuple[str, str]] = []
    seen = set()
    for r in recs:
        s = normalize_sequence(r.sequence)
        if s in seen:
            continue
        seen.add(s)
        real.append((ev.split_of(r.nuc_accession or r.uid), s))

    n_real, n_kmers_real = count_leaks(real, args.k)

    rng = random.Random(args.seed)
    nulls = []
    for _ in range(args.trials):
        perm = [((sp), "".join(rng.choice("ACGU") for _ in range(len(s)))) for sp, s in real]
        n_null, n_kmers_null = count_leaks(perm, args.k)
        nulls.append((n_null, n_kmers_null))
    n_null_avg = sum(x[0] for x in nulls) / len(nulls)

    report = {
        "records": len(real),
        "k": args.k,
        "alphabet_space": 4 ** args.k,
        "total_kmers_real": n_kmers_real,
        "total_kmers_null_avg": sum(x[1] for x in nulls) / len(nulls),
        "leaking_kmers_real": n_real,
        "leaking_kmers_null_random": [x[0] for x in nulls],
        "leaking_kmers_null_avg": round(n_null_avg, 1),
        "excess_over_random_x": round(n_real / max(n_null_avg, 1e-9), 2),
        "verdict": ("判据在随机序列上同样大量报警 → 12-mer 对本规模语料判别力不足"
                    if n_null_avg > 0.2 * n_real else
                    "随机基线远低于实测 → 泄漏更可能来自真实的序列相似性"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[null] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
