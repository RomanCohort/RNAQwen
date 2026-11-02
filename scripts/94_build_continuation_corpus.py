"""Build the Rfam + free-text continuation corpus for the D arm.

The D arm trained on seqtask_full alone.  This adds the two sources that the
old C_merged run mixed in and that broke it, so the continuation is the direct
test of whether that inversion was intrinsic or an artifact of the small
merged corpus (rnaqa_merged.jsonl was 88,743 rows; seqtask_full alone is
1,496,820, which changes the loss-token balance on its own).

Split discipline, unchanged from 74_merge_datasets.py:
  seqtask_full  carries its own split (by RNA: 129,791 / 2,950 / 16,941)
  rfam          carries its own split (31,560 train / 7,892 test, no val)
  freetext      has no split at all -> pinned to train, never used for early
                stopping.  Its RNAs are the same 4,481 as the main cache, and
                some of those are seqtask val/test RNAs, so the seqtask ruler is
                mildly optimistic.  Same caveat as the old merge.

Two free-text variants:
  raw    - the file as built by 60_freetext_qa.py
  clean  - refusals dropped and exact-duplicate answers capped per template

Usage:
  python scripts/94_build_continuation_corpus.py --freetext raw   --out data/processed/rnaqa_cont_raw.jsonl
  python scripts/94_build_continuation_corpus.py --freetext clean --out data/processed/rnaqa_cont_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed"

REFUSAL = "the provided annotation does not state this."


def rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def clean_freetext(items, dup_cap):
    """Drop refusals, then cap how often one exact answer may repeat per template."""
    kept, seen = [], defaultdict(int)
    n_ref = n_dup = 0
    for r in items:
        a = (r.get("answer") or "").strip()
        if a.lower().startswith(REFUSAL[:40]):
            n_ref += 1
            continue
        key = (r.get("template"), a)
        seen[key] += 1
        if seen[key] > dup_cap:
            n_dup += 1
            continue
        kept.append(r)
    return kept, n_ref, n_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freetext", choices=["raw", "clean"], default="raw")
    ap.add_argument("--dup-cap", type=int, default=5)
    ap.add_argument("--up-rfam", type=int, default=1,
                    help="duplicate rfam TRAIN rows this many times")
    ap.add_argument("--up-freetext", type=int, default=1,
                    help="duplicate freetext TRAIN rows this many times")
    ap.add_argument("--seqtask-cap", type=int, default=0,
                    help="cap seqtask TRAIN rows by strided sampling (0 = keep all). "
                         "Needed because the replay set is what starves the new "
                         "sources: rfam+freetext are only about 3 percent of a "
                         "1.5M-row train list, so a few thousand steps would "
                         "barely touch them.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    src_n = Counter()
    tpl_n = Counter()
    split_n = Counter()
    ans_chars = defaultdict(int)
    written = 0

    with open(out, "w", encoding="utf-8") as fo:
        def emit(items, source, sp_of, upsample=1):
            nonlocal written
            for r in items:
                sp = sp_of(r)
                reps = upsample if sp == "train" else 1
                for _ in range(reps):
                    row = dict(r)
                    row["split"] = sp
                    row["source"] = source
                    fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1
                    src_n[source] += 1
                    tpl_n[(source, row.get("template"))] += 1
                    split_n[sp] += 1
                    ans_chars[source] += len(row.get("answer") or "")

        # seqtask_full: keep its own split; train rows may be strided down so the
        # new sources actually get drawn.
        st = list(rows(D / "rnaqa_seqtask_full.jsonl"))
        if args.seqtask_cap:
            tr = [r for r in st if r.get("split", "train") == "train"]
            rest = [r for r in st if r.get("split", "train") != "train"]
            if len(tr) > args.seqtask_cap:
                stride = len(tr) / args.seqtask_cap
                kept = [tr[int(i * stride)] for i in range(args.seqtask_cap)]
                print("  seqtask train strided: %d -> %d (stride %.2f)"
                      % (len(tr), len(kept), stride))
                st = kept + rest
        emit(st, "seqtask", lambda r: r.get("split", "train"))

        # rfam: keep its own split
        emit(rows(D / "rnaqa_rfam.jsonl"), "rfam",
             lambda r: r.get("split", "train"), args.up_rfam)

        # freetext: always train
        ft = list(rows(D / "rnaqa_freetext.jsonl"))
        n_ref = n_dup = 0
        if args.freetext == "clean":
            ft, n_ref, n_dup = clean_freetext(ft, args.dup_cap)
        emit(ft, "freetext", lambda r: "train", args.up_freetext)

    print("wrote %s   %d rows" % (out.name, written))
    print("  by source : %s" % dict(src_n))
    print("  by split  : %s" % dict(split_n))
    if args.freetext == "clean":
        print("  freetext cleaned: dropped %d refusals, %d over-cap duplicates"
              % (n_ref, n_dup))
    print("  mean answer chars: %s"
          % {k: round(v / src_n[k], 1) for k, v in ans_chars.items()})
    print()
    print("  templates:")
    for (src, tpl), c in tpl_n.most_common():
        print("    %-9s %7d  %s" % (src, c, str(tpl)[:66]))

    print()
    print("  NOTE: tokens are what the loss sees, so the source that owns the")
    print("  most answer characters owns the gradient.  Compare against the old")
    print("  C_merged balance (seqtask 16.0 / rfam 25.3 / freetext 58.8).")
    print("  split counts: %s" % dict(split_n))


if __name__ == "__main__":
    main()
