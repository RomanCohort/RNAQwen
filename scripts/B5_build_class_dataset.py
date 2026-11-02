"""Build the missing task: "which class of RNA is this".

Why
---
The D_qwen36_sos checkpoint was trained on ten templates and all ten ask for
local characters (first/last 4-8 nt, one dinucleotide, two reverse complements).
`reports/B4_generalization.md` shows what that buys: the model reads the ends of
a sequence well, and answers a class question at 0.333 with the shuffled control
at 0.500 -- i.e. it answers from the label prior, not from the RNA.

Meanwhile a linear probe on the same FSQ codes reaches 0.599 balanced on the
three-class reduction (`B2_probe_c3000.json`).  The information is in the code
channel; the LLM was simply never asked to use it.

This script writes that task in the format `50_routeC_e2e.py --data` expects:

    {"uid": <bare md5>, "sequence": ..., "question": ..., "answer": ...,
     "kind": "class", "source": "rfam_class", "template": ..., "split": ...}

`uid` must be the bare md5 (no "seq:" prefix) because the rfam hidden-state cache
is registered under those keys in `_rfa_full_index.pt`.

Split protocol
--------------
Family-level holdout only where it is actually possible.  Measured shares of the
largest single family per class:

    tRNA     81.1%  (2 families)      -> sequence-level
    rRNA     62.8%  (9 families)      -> sequence-level
    other    72.1%  (893, unassigned) -> sequence-level
    snRNA    34.2%  (11 families)     -> sequence-level
    ribozyme 32.6%  (27 families)     -> sequence-level
    snoRNA   24.6%  (290 families)    -> family-level
    miRNA     8.5%  (716 families)    -> family-level

A family-level split on rRNA would put SSU_rRNA_eukarya (62.8% of the class) on
one side and leave the other side with 1-2 family names -- which is what the
first version of this script did, and why rRNA's test set was 190 rows.  Those
classes now use a sequence-level split and are labelled as the weaker protocol.

Where a family split is used, families are assigned greedily by size so the
70/15/15 proportions hold in *sequences*, not in family names.

Class cap
---------
rRNA is 52% of the corpus.  Training on the natural distribution teaches the
prior, which is the exact failure mode B4 measured.  `--cap-per-class` truncates
each class (sampled with a fixed seed) so every class contributes equally.

Usage
-----
    python scripts/B5_build_class_dataset.py --dry-run        # stats only
    python scripts/B5_build_class_dataset.py                  # write file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEQS = ROOT / "data/processed/rfam_seqs.jsonl"
IDX = ROOT / "data/processed/_rfa_full_index.pt"
DEFAULT_OUT = ROOT / "data/processed/rnaqa_class.jsonl"

CLASSES = ["tRNA", "rRNA", "miRNA", "snoRNA", "snRNA", "ribozyme", "other"]

TEMPLATES = [
    "What type of RNA is this: tRNA, rRNA, miRNA, snoRNA, snRNA, ribozyme, or other?",
    "Which class does this RNA belong to: tRNA, rRNA, miRNA, snoRNA, snRNA, "
    "ribozyme, or other?",
]

# Not "acgu" in a set check: Selenocysteine tRNAs and modified bases exist in
# Rfam alignments.  Anything outside ACGU is skipped rather than silently mapped.
ACGU = set("ACGU")


def coarse(family: str, name: str) -> str:
    """Rfam family -> coarse class.  Keyword-based and explicit.

    A clan-based mapping would be better but needs the clan file, which is not in
    the repo.  Unmatched families land in `other` and stay visible as such.
    """
    n = (name or "").lower()
    if family == "unassigned":
        return "other"
    if n.startswith("trna"):
        return "tRNA"
    if "ssu_rrna" in n or "lsu_rrna" in n or n in ("5s_rrna", "5_8s_rrna", "5.8s_rrna"):
        return "rRNA"
    if n.startswith("mir"):
        return "miRNA"
    if n.startswith("sno"):
        return "snoRNA"
    if n in ("u1", "u2", "u4", "u5", "u6", "u11", "u12", "srp", "7sk",
             "metazoa_srp", "vault", "y_rna"):
        return "snRNA"
    if any(k in n for k in ("hammerhead", "ribozyme", "rnasep", "glms", "cobalamin",
                            "twister", "hairpin", "hdv", "group")):
        return "ribozyme"
    return "other"


FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}


def seq_split(uid: str) -> str:
    h = int(hashlib.md5(uid.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def family_split(fams: list[str], size: dict[str, int]) -> dict[str, str]:
    """Greedy size-balanced assignment of whole families to splits.

    Sorting by size and always filling the split with the largest remaining
    deficit keeps the *sequence* proportions near 70/15/15 even when family
    sizes span three orders of magnitude.
    """
    total = sum(size[f] for f in fams)
    target = {k: total * v for k, v in FRACS.items()}
    have = {k: 0 for k in FRACS}
    out = {}
    for f in sorted(fams, key=lambda x: -size[x]):
        k = max(FRACS, key=lambda s: target[s] - have[s])
        out[f] = k
        have[k] += size[f]
    return out


def family_protocol(fam: str, n_seqs: int, top1_frac: float) -> bool:
    """True when a family-level holdout is meaningful for this class."""
    return n_seqs >= 800 and top1_frac < 0.30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--cap-per-class", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    import torch

    idx = torch.load(IDX, map_location="cpu", weights_only=False)
    cache_uids = set(idx["uids"])
    print(f"[cache] {len(cache_uids):,} sequences in _rfa_full_index.pt")

    pools = defaultdict(list)
    fam_of_class = defaultdict(set)
    fam_size = Counter()
    dropped = Counter()
    n = 0
    with open(SEQS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n += 1
            u = d["uid"]
            key = u[4:] if u.startswith("seq:") else u
            if key not in cache_uids:
                dropped["not_in_cache"] += 1
                continue
            seq = d["sequence"].upper()
            if not (4 <= len(seq) <= 1022):
                dropped["bad_length"] += 1
                continue
            if not set(seq) <= ACGU:
                dropped["non_acgu"] += 1
                continue
            c = coarse(d["family"], d["family_name"])
            pools[c].append({"uid": key, "sequence": seq, "family": d["family"],
                             "cls": c})
            fam_of_class[c].add(d["family"])
            fam_size[d["family"]] += 1

    print(f"[read] {n:,} rows   dropped: {dict(dropped)}")
    print(f"\n{'class':10s} {'seq':>8s} {'families':>9s} {'capped':>8s}")
    rng = random.Random(args.seed)
    kept = {}
    for c in CLASSES:
        pool = pools[c]
        rng.shuffle(pool)
        kept[c] = pool[: args.cap_per_class]
        print(f"{c:10s} {len(pool):8d} {len(fam_of_class[c]):9d} {len(kept[c]):8d}")

    rows = []
    plan = {}
    for c in CLASSES:
        cls_total = len(pools[c])
        top1 = max((fam_size[f] for f in fam_of_class[c]), default=0) / max(1, cls_total)
        use_fam = family_protocol(c, cls_total, top1)
        if use_fam:
            fams = sorted(fam_of_class[c])
            assigned = family_split(fams, fam_size)
            plan[c] = ("family", assigned)
        else:
            plan[c] = ("sequence", None)
        for r in kept[c]:
            sp = (plan[c][1][r["family"]] if use_fam else seq_split(r["uid"]))
            for t in TEMPLATES:
                rows.append({"uid": r["uid"], "sequence": r["sequence"],
                             "question": t, "answer": c, "kind": "class",
                             "source": "rfam_class", "template": t, "split": sp,
                             "split_protocol": plan[c][0]})

    print(f"\n{'class':10s} {'protocol':9s} {'train':>7s} {'val':>7s} {'test':>7s}  "
          f"{'test fams':>9s}")
    for c in CLASSES:
        sub = [r for r in rows if r["answer"] == c]
        cnt = Counter(r["split"] for r in sub)
        if plan[c][0] == "family":
            tf = str(sum(1 for f, s in plan[c][1].items() if s == "test"))
        else:
            # Every test family also appears in train here; printing a number
            # would look like a holdout that does not exist.
            tf = "-"
        print(f"{c:10s} {plan[c][0]:9s} {cnt['train']:7d} {cnt['val']:7d} "
              f"{cnt['test']:7d}  {tf:>9s}")

    tot = Counter(r["split"] for r in rows)
    print(f"\n[total] {len(rows):,} rows  {dict(tot)}")

    empty = [c for c in CLASSES if not any(r["answer"] == c and r["split"] == "test"
                                           for r in rows)]
    if empty:
        print(f"** WARNING: no test rows for {empty} — split is degenerate **")
    soft = [c for c in CLASSES if plan[c][0] == "sequence"]
    if soft:
        print(f"** CAVEAT: {soft} use a sequence-level split — their test rows "
              f"come from families the model also trained on **")

    if args.dry_run:
        print("\n[dry-run] nothing written")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[out] {out}  {len(rows):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
