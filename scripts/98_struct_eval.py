"""Freeze a measured-structure eval set from the RCSB/NDB complexes.

Truth  = curated base pairs shipped in the mmCIF (_ndb_struct_na_base_pair)
Base   = ViennaRNA MFE, i.e. the free-solution prediction
Metric = base-pair precision / recall / F1 on the pair sets

Rows whose F1 is very low are the interesting ones: the RNA only folds that way
inside the receptor.  2B3J (a TLR7 ligand) is the canonical example.
"""
from __future__ import annotations

import json
from pathlib import Path

import RNA

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/processed/pdb_rna_struct.jsonl"
OUT = ROOT / "data/processed/pdb_struct_eval.jsonl"
SUM = ROOT / "reports/pdb_struct_eval_summary.json"


def pairs(db: str) -> set[tuple[int, int]]:
    st, out = [], set()
    for i, ch in enumerate(db):
        if ch == "(":
            st.append(i)
        elif ch == ")" and st:
            out.add((st.pop(), i))
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return dict(precision=round(p, 4), recall=round(r, 4),
                f1=round(2 * p * r / max(1e-9, p + r), 4), tp=tp, fp=fp, fn=fn)


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    singles = [r for r in rows if r["task"] == "pdb_struct"]
    out_rows = []
    tp = fp = fn = 0
    per_entry = {}
    for r in singles:
        truth = pairs(r["structure"])
        pred_db, dg = RNA.fold(r["sequence"])
        pred = pairs(pred_db)
        m = prf(len(truth & pred), len(pred - truth), len(truth - pred))
        tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
        rec = dict(r)
        rec.update({"baseline": "vienna_mfe", "pred": pred_db, "mfe": round(dg, 3),
                    "score": m})
        out_rows.append(rec)
        per_entry.setdefault(r["entry"], []).append(m["f1"])

    with OUT.open("w", encoding="utf-8") as fh:
        for rec in out_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "n_chains": len(singles),
        "n_entries": len(per_entry),
        "baseline": "ViennaRNA MFE",
        "overall": prf(tp, fp, fn),
        "worst_entries": sorted(((e, round(sum(v) / len(v), 3)) for e, v in per_entry.items()),
                                key=lambda x: x[1])[:8],
        "note": "duplexes (pdb_duplex) are kept in pdb_rna_struct.jsonl but not scored "
                "here: a single-chain MFE has no intermolecular pairs to compare.",
    }
    SUM.parent.mkdir(parents=True, exist_ok=True)
    SUM.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False))
    print("worst:", summary["worst_entries"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())