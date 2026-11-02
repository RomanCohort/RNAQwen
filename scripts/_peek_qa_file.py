"""看一眼 RNA-QA 文件里的实际内容 + 被闸门拦下的原因分布。"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.records import read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--show", type=int, default=6)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--dropped", type=Path, default=None)
    args = ap.parse_args()

    recs = list(read_jsonl(args.inp))
    kinds = collections.Counter()
    variants = collections.Counter()
    for r in recs:
        for p in r.qa:
            kinds[p.get("kind", "?")] += 1
            variants[p.get("variant", "?")] += 1
    print(f"records={len(recs)} qa={sum(kinds.values())}")
    print("kinds   ", dict(kinds))
    print("variants", dict(variants))
    print()

    shown = 0
    for r in recs:
        if not r.qa:
            continue
        shown += 1
        print("=" * 96)
        print(f"{r.uid} | {r.provenance.get('ent_id','')} | {len(r.sequence)} nt")
        print(f"summary: {r.annotation[:200]}")
        for p in r.qa[: args.per]:
            print(f"  [{p.get('variant')}/{p.get('kind')}] Q: {p.get('question','')[:180]}")
            print(f"        A: {p.get('answer','')[:240]}")
        if shown >= args.show:
            break

    dp = args.dropped or args.inp.with_suffix(".dropped.jsonl")
    if dp.exists():
        reasons = collections.Counter()
        by_kind = collections.Counter()
        for ln in dp.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            d = json.loads(ln)
            reasons[d.get("reason", "?")] += 1
            by_kind[f"{d.get('variant','?')}/{d.get('kind','?')}"] += 1
        print()
        print(f"[dropped] {dp}  共 {sum(reasons.values())} 条")
        print("  reasons :", dict(reasons))
        print("  by kind :", dict(by_kind))
    return 0


if __name__ == "__main__":
    sys.exit(main())
