"""RNAQA v3 = v2 (L2 annotation + L3 structure) + L4 composite long answers.

Why L4: in v2 the median answer is 5 characters, so 98.7% of the corpus teaches
"pick a label", not "explain".  The long-form sources we already have are tiny
(4,481 annotation + 1,850-char literature; 10,832 freetext), which cannot move a
35B model.  But every one of the 92,685 structured Rfam entries supports a
deterministic multi-sentence answer built from its ViennaRNA features, and those
answers are *necessarily* sequence-dependent.

No API, no hallucination: every number in a composite answer is copied from
rfam_struct.jsonl, which was computed from the sequence itself.

Pure Python / no torch, so this can run while the GPU is busy.
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
OUT = ROOT / "data/processed/rnaqa_v3_l234.jsonl"
MAX_Q_PER_SEQ = 10
TEMPLATE_CAP = 0.15


def build_split_map(fam_counts):
    tot = sum(fam_counts.values())
    order = sorted(fam_counts, key=lambda f: hashlib.md5(f.encode()).hexdigest())
    mp, val_n, test_n = {}, 0, 0
    for fam in order:
        n = fam_counts[fam]
        if val_n < tot * 0.05:
            mp[fam] = "val"; val_n += n
        elif test_n < tot * 0.10:
            mp[fam] = "test"; test_n += n
        else:
            mp[fam] = "train"
    return mp, val_n, test_n


def long_answers(seq, fam, fname, label, st):
    """Deterministic, grounded, multi-sentence answers."""
    n = len(seq)
    out = []
    if fname:
        out.append((
            "Describe this RNA's family and length in a paragraph.",
            f"This sequence belongs to Rfam family {fam} ({fname}), which is a "
            f"{label or fname} type RNA. The submitted sequence is {n} nucleotides long.",
            "l4_family_paragraph", "l4_family_para"))
    if st:
        mfe = float(st.get("mfe", 0.0))
        mfe_nt = float(st.get("mfe_per_nt", 0.0))
        pf = float(st.get("paired_frac", 0.0))
        ls = int(st.get("longest_stem", 0))
        acc = float(st.get("acc_mean", 0.0))
        out.append((
            "Describe the secondary structure of this RNA in a paragraph.",
            f"In its minimum free energy structure this {n}-nt RNA has a folding "
            f"energy of {mfe:.1f} kcal/mol, or {mfe_nt:.3f} kcal/mol per nucleotide. "
            f"{pf*100:.0f}% of the bases are paired, the longest helical stem spans "
            f"{ls} base pairs, and the mean unpaired probability is {acc:.2f}.",
            "l4_structure_paragraph", "l4_struct_para"))
        if fname:
            out.append((
                "Give a complete profile of this RNA in one paragraph.",
                f"This {n}-nt sequence is a member of Rfam family {fam} ({fname}), "
                f"a {label or fname} type RNA. It folds with a minimum free energy of "
                f"{mfe:.1f} kcal/mol ({mfe_nt:.3f} per nucleotide); {pf*100:.0f}% of "
                f"bases are paired, the longest stem is {ls} bp, and the mean "
                f"unpaired probability is {acc:.2f}.",
                "l4_full_profile", "l4_full_para"))
    return out


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
    mp, vn, tn = build_split_map(fam_counts)
    print(f"[split] val={vn:,} test={tn:,} train={sum(fam_counts.values())-vn-tn:,}", flush=True)

    items, n_seq = [], 0
    with RFAM.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            uid, seq = r["uid"], r["sequence"]
            fam, fname, label = r.get("family") or "", r.get("family_name") or "", r.get("label") or ""
            if not fam or not seq:
                continue
            n_seq += 1
            sp = mp.get(fam, "train")
            base = dict(uid=uid, sequence=seq, split=sp, source="rfam",
                        meta=dict(family=fam, family_name=fname, type=label, len=len(seq)))
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
                qs.append(("What fraction of nucleotides is base-paired in the minimum free energy structure?",
                           f"{float(st.get('paired_frac', 0.0)):.2f}", "struct", "l3_paired_frac"))
                qs.append(("What is the minimum free energy of this RNA in kcal/mol?",
                           f"{float(st.get('mfe', 0.0)):.1f}", "struct", "l3_mfe"))
                qs.append(("How many base pairs are in the longest helical stem?",
                           str(int(st.get("longest_stem", 0))), "struct", "l3_longest_stem"))
                qs.append(("What is the mean unpaired probability across this RNA?",
                           f"{float(st.get('acc_mean', 0.0)):.2f}", "struct", "l3_accessibility"))
            qs.extend(long_answers(seq, fam, fname, label, st))
            for q, ans, kind, tpl in qs[:MAX_Q_PER_SEQ]:
                items.append(dict(base, question=q, answer=ans, kind=kind, template=tpl))

    print(f"[gen] sequences={n_seq:,} items={len(items):,}", flush=True)
    by_tpl = Counter(it["template"] for it in items)
    cap = int(len(items) * TEMPLATE_CAP)
    allowed = {t: min(c, cap) for t, c in by_tpl.items()}
    seen, kept = Counter(), []
    for it in items:
        t = it["template"]
        if seen[t] >= allowed[t]:
            continue
        seen[t] += 1
        kept.append(it)
    print(f"[cap] {len(items):,} -> {len(kept):,} (cap {cap:,}/template)", flush=True)

    out = Path(a.out)
    with out.open("w", encoding="utf-8") as w:
        for it in kept:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")

    sp = Counter(it["split"] for it in kept)
    tpl = Counter(it["template"] for it in kept)
    ans_tok = sum(max(1, len(it["answer"]) // 4) for it in kept)
    code_tok = sum(max(1, len(it["sequence"]) // 4) for it in kept)
    q_tok = sum(max(1, len(it["question"]) // 4) for it in kept)
    long_tpl = [t for t in tpl if t.startswith("l4_")]
    long_tok = sum(max(1, len(it["answer"]) // 4) for it in kept if it["template"] in long_tpl)
    print(f"[save] {out}  {out.stat().st_size/1e6:.0f} MB", flush=True)
    print(f"[split] {dict(sp)}", flush=True)
    print(f"[templates] {dict(tpl)}", flush=True)
    print(f"[tokens] answer={ans_tok/1e6:.1f}M code={code_tok/1e6:.1f}M question={q_tok/1e6:.1f}M", flush=True)
    print(f"[share] answer={100*ans_tok/(ans_tok+code_tok+q_tok):.1f}%  "
          f"long-answer={100*long_tok/(ans_tok+code_tok+q_tok):.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
