"""按 variant 分组打印某条记录的全部 QA，用来判断 ais / dc 两个变体各自的问法。"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.records import read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="看前几条记录")
    args = ap.parse_args()

    recs = list(read_jsonl(args.inp))
    for r in recs[args.idx: args.idx + args.n]:
        print("=" * 98)
        print(f"{r.uid} | {r.provenance.get('ent_id','')} | {len(r.sequence)} nt")
        print(f"[literature] {r.literature[:220]}")
        print(f"[annotation] {r.annotation[:220]}")
        by = collections.defaultdict(list)
        for p in r.qa:
            by[p.get("variant", "?")].append(p)
        for var in ("ais", "dc"):
            print(f"\n --- variant={var} ({len(by[var])} 组) ---")
            for p in by[var]:
                print(f"   Q: {p.get('question','')[:190]}")
                print(f"   A: {p.get('answer','')[:230]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
