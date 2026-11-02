"""Build RNAQA v2, stage A: L2 (annotation) + L3 (structure).

Design constraints (see reports/rnaqa_v2_design.md):
  - only `sequence` is stored; codes are derived at training time
  - at most 8 questions per sequence
  - split by Rfam family, never by uid, so homologues do not straddle
  - every answer is a deterministic function of the sequence/annotation

L1 (mechanical string ops), L4-L6 are deliberately NOT included yet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RFAM = ROOT / "data/processed/rfam_seqs.jsonl"
STRUCT = ROOT / "data/processed/rfam_struct.jsonl"
OUT = ROOT / "data/processed/rnaqa_v2_l23.jsonl"
MAX_Q_PER_SEQ = 8
TEMPLATE_CAP = 0.20


SPLIT_MAP = {}


def build_split_map(fam_counts):
    """Assign whole families to train/val/test so that the SEQUENCE counts
    (not the family counts) land on 85/5/10.  Hashing families alone leaves the
    split wildly unbalanced because a few families hold most sequences."""
    test_target = sum(fam_counts.values()) * 0.10
    val_target = sum(fam_counts.values()) * 0.05
    order = sorted(fam_counts, key=lambda f: hashlib.md5(f.encode()).hexdigest())
    val_n = test_n = 0
    for fam in order:
        n = fam_counts[fam]
        if val_n < val_target:
            SPLIT_MAP[fam] = "val"; val_n += n
        elif test_n < test_target:
            SPLIT_MAP[fam] = "test"; test_n += n
        else:
            SPLIT_MAP[fam] = "train"
    print(f"[split] val={val_n:,} test={test_n:,} train={sum(fam_counts.values())-val_n-test_n:,}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    struct = {}
    with STRUCT.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("struct"):
                struct[r["uid"]] = r["struct"]
    print(f"[read] struct={len(struct):,}", flush=True)

    fam_counts = Counter()
    with RFAM.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("family") and r.get("sequence"):
                fam_counts[r["family"]] += 1
    build_split_map(fam_counts)

    items = []
    n_seq = 0
    with RFAM.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            uid = r["uid"]; seq = r["sequence"]
            fam = r.get("family") or ""
            fname = r.get("family_name") or ""
            label = r.get("label") or ""
            if not fam or not seq:
                continue
            n_seq += 1
            sp = SPLIT_MAP.get(fam, "train")
            base = dict(uid=uid, sequence=seq, split=sp, source="rfam",
                        meta={"family": fam, "family_name": fname, "type": label,
                              "len": len(seq)})
            qs = []
            if fname:
                qs.append(("Which Rfam family does this RNA belong to?",
                           fname, "family_name", "l2_family_name"))
            qs.append(("What is the Rfam accession of this RNA?",
                       fam, "family_acc", "l2_family_acc"))
            if label:
                qs.append(("What type of RNA is this sequence?",
                           label, "rna_type", "l2_rna_type"))
            st = struct.get(uid)
            if st:
                pf = float(st.get("paired_frac", 0.0))
                qs.append(("What fraction of nucleotides is base-paired in the minimum free energy structure?",
                           f"{pf:.2f}", "struct", "l3_paired_frac"))
                mfe = float(st.get("mfe", 0.0))
                qs.append(("What is the minimum free energy of this RNA in kcal/mol?",
                           f"{mfe:.1f}", "struct", "l3_mfe"))
                ls = int(st.get("longest_stem", 0))
                qs.append(("How many base pairs are in the longest helical stem?",
                           str(ls), "struct", "l3_longest_stem"))
                am = float(st.get("acc_mean", 0.0))
                qs.append(("What is the mean unpaired probability across this RNA?",
                           f"{am:.2f}", "struct", "l3_accessibility"))
            for q, ans, kind, tpl in qs[:MAX_Q_PER_SEQ]:
                items.append(dict(base, question=q, answer=ans, kind=kind,
                                  template=tpl))

    print(f"[gen] sequences={n_seq:,} items={len(items):,}", flush=True)

    # global template cap (drop from the most frequent templates)
    by_tpl = Counter(it["template"] for it in items)
    cap = int(len(items) * TEMPLATE_CAP)
    allowed = {t: min(c, cap) for t, c in by_tpl.items()}
    seen = Counter()
    kept = []
    for it in items:
        t = it["template"]
        if seen[t] >= allowed[t]:
            continue
        seen[t] += 1
        kept.append(it)
    print(f"[cap] {len(items):,} -> {len(kept):,} "
          f"(cap {cap:,} per template)", flush=True)

    out = Path(a.out)
    with out.open("w", encoding="utf-8") as w:
        for it in kept:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")

    sp = Counter(it["split"] for it in kept)
    src = Counter(it["kind"] for it in kept)
    fams = defaultdict(set)
    for it in kept:
        fams[it["split"]].add(it["meta"]["family"])
    manifest = dict(out=out.name, rows=len(kept), sequences=n_seq,
                    splits=dict(sp), kinds=dict(src),
                    families={k: len(v) for k, v in fams.items()},
                    templates=dict(Counter(it["template"] for it in kept)),
                    max_q_per_seq=MAX_Q_PER_SEQ, template_cap=TEMPLATE_CAP)
    (out.with_suffix(".manifest.json")).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out}  {out.stat().st_size/1e6:.0f} MB")
    print(json.dumps(manifest["splits"], ensure_ascii=False))
    print(json.dumps(manifest["kinds"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
