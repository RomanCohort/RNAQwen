"""Mix the new class task with the old character task, 1:1 per split.

Why mix
-------
The D_qwen36_sos checkpoint already answers the ten local-character templates at
0.69 (reports/B3_score_audit.md).  Fine-tuning on class questions alone would
most likely overwrite that: the two tasks share the same prompt shape and the
same answer-only loss, so nothing protects the old behaviour.

Mixing per split (not globally) keeps the 70/15/15 proportions intact, so the
class rows' val/test sides stay clean.

Sampling is deterministic and proportional to the source split sizes, and it is
reservoir-free: the seqtask file is grouped in memory by split (400k rows,
~300 MB).  That is small enough on this box and keeps the code obvious.

Usage:
    python scripts/B6_mix_class_seqtask.py --ratio 1.0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

D = ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-file", default=str(D / "rnaqa_class.jsonl"))
    ap.add_argument("--old-file", default=str(D / "rnaqa_seqtask_full.jsonl"))
    ap.add_argument("--out", default=str(D / "rnaqa_class_mix.jsonl"))
    ap.add_argument("--ratio", type=float, default=1.0,
                    help="old rows per class row, per split")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()

    cls_rows = []
    with open(args.class_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cls_rows.append(json.loads(line))
    cls_split = Counter(r["split"] for r in cls_rows)
    print(f"[class] {len(cls_rows):,}  {dict(cls_split)}")

    old_by_split = {"train": [], "val": [], "test": []}
    with open(args.old_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            old_by_split[r.get("split", "train")].append(r)
    print(f"[old]   {sum(len(v) for v in old_by_split.values()):,}  "
          f"{ {k: len(v) for k, v in old_by_split.items()} }")

    rng = random.Random(args.seed)
    picked_old = []
    for sp in ("train", "val", "test"):
        want = int(cls_split[sp] * args.ratio)
        pool = old_by_split[sp]
        take = min(want, len(pool))
        if take < want:
            print(f"** {sp}: 需要 {want:,} 条旧数据，池里只有 {take:,} **")
        picked_old.extend(rng.sample(pool, take))

    rows = cls_rows + picked_old
    rng.shuffle(rows)

    print(f"\n{'split':6s} {'class':>7s} {'old':>7s} {'total':>7s}")
    for sp in ("train", "val", "test"):
        a = sum(1 for r in cls_rows if r["split"] == sp)
        b = sum(1 for r in picked_old if r["split"] == sp)
        print(f"{sp:6s} {a:7d} {b:7d} {a+b:7d}")
    print(f"{'total':6s} {len(cls_rows):7d} {len(picked_old):7d} {len(rows):7d}")

    # 每一条 task 都该有答案，空答案会让 answer-only loss 退化成 0 且不报错。
    bad = [i for i, r in enumerate(rows)
           if not r.get("answer") or not r.get("question") or not r.get("uid")]
    if bad:
        raise SystemExit(f"** {len(bad)} 条缺字段，首条下标 {bad[0]} **")

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[out] {args.out}  {len(rows):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
