"""人工抽查：从最终 RNA-QA 里随机抽 N 条，连序列长度 / summary / 两组 QA 一起打印。

同时给出全量统计（kind 分布、variant 分布、每条 QA 数分布），
因为这些数字要写进验收报告。

跑法：
    python scripts/_spotcheck_rnacentral.py --in data/processed/rnaqa_rnacentral.jsonl --n 10
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.records import read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "spotcheck_rnacentral.json")
    args = ap.parse_args()

    recs = list(read_jsonl(args.inp))
    print(f"[spot] 读入 {len(recs)} 条 <- {args.inp}\n")

    # ---- 全量统计 ----
    kinds = collections.Counter()
    variants = collections.Counter()
    sources = collections.Counter()
    per_rec = []
    q_lens, a_lens = [], []
    empty_recs = 0
    for r in recs:
        per_rec.append(len(r.qa))
        if not r.qa:
            empty_recs += 1
        for p in r.qa:
            kinds[p.get("kind", "?")] += 1
            variants[p.get("variant", "?")] += 1
            sources[p.get("source", "?")] += 1
            q_lens.append(len(p.get("question", "")))
            a_lens.append(len(p.get("answer", "")))
    per_rec.sort()
    stats = {
        "records": len(recs),
        "records_without_qa": empty_recs,
        "qa_total": sum(per_rec),
        "kinds": dict(kinds),
        "variants": dict(variants),
        "sources": dict(sources),
        "qa_per_record": {
            "min": per_rec[0] if per_rec else 0,
            "median": per_rec[len(per_rec) // 2] if per_rec else 0,
            "max": per_rec[-1] if per_rec else 0,
            "mean": round(sum(per_rec) / max(len(per_rec), 1), 2),
        },
        "q_len_median": q_lens[len(q_lens) // 2] if q_lens else 0,
        "a_len_median": a_lens[len(a_lens) // 2] if a_lens else 0,
    }
    print("[stats]", json.dumps(stats, ensure_ascii=False, indent=2))

    # ---- 抽查 ----
    rng = random.Random(args.seed)
    elig = [r for r in recs if r.qa]
    pick = rng.sample(elig, min(args.n, len(elig)))
    samples = []
    for i, r in enumerate(pick, 1):
        open_qa = next((p for p in r.qa if p.get("kind") == "open_ended"), None) or \
                  next((p for p in r.qa if p.get("variant") == "dc"), None)
        closed_qa = next((p for p in r.qa if p.get("kind") in ("yes_no", "multiple_choice")), None)
        s = {
            "idx": i,
            "uid": r.uid,
            "ent_id": r.provenance.get("ent_id", ""),
            "n_nt": len(r.sequence),
            "sequence": r.sequence,
            "rna_type": r.rna_type,
            "definition": r.definition,
            "summary_150": r.annotation[:150],
            "summary_full": r.annotation,
            "literature_300": r.literature[:300],
            "n_qa": len(r.qa),
            "open": open_qa,
            "closed": closed_qa,
        }
        samples.append(s)
        print("=" * 100)
        print(f"[{i}] uid={r.uid}")
        print(f"    ent_id={s['ent_id']}  type={r.rna_type!r}  len={s['n_nt']} nt  n_qa={s['n_qa']}")
        print(f"    definition: {r.definition[:120]}")
        print(f"    seq[:60]  : {r.sequence[:60]}")
        print(f"    summary   : {r.annotation[:150]}")
        if open_qa:
            print(f"    Q(open)   : {open_qa['question'][:170]}")
            print(f"    A(open)   : {open_qa['answer'][:220]}")
        else:
            print("    Q(open)   : <无>")
        if closed_qa:
            print(f"    Q(closed) : {closed_qa['question'][:170]}   [{closed_qa.get('kind')}]")
            print(f"    A(closed) : {closed_qa['answer'][:180]}")
        else:
            print("    Q(closed) : <无>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"stats": stats, "samples": samples},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[spot] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
