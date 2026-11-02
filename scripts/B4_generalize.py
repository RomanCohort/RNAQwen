"""Does RNAQwen generalise past its ten training templates?

The corrected score audit (reports/B3_score_audit.md) shows the model answers the
ten trained templates at 69%.  That says nothing about anything it was not
trained on, and the training corpus contains *only* local character questions.

This script asks four questions the checkpoint never saw, over real Rfam
sequences, through the already-running service on port 8100:

    first 10 nt      extrapolate past the trained 4/6/8
    last 10 nt       same, from the other end
    how many G       global counting — the FSQ 4x pooling should make this hard
    which class      rRNA / tRNA / other — never trained, but the code probe says
                     a linear classifier reaches 0.60 balanced, so a trained LLM
                     ought to be able to do better than chance if it reads codes

Each question is asked twice: with the real sequence (A) and with a different
sequence of similar length (B).  A-B is the only number that matters; a model
that ignores the RNA scores the same on both.

Usage:
    python scripts/B4_generalize.py --n-per-class 4
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

SEQS = ROOT / "data/processed/rfam_seqs.jsonl"
OUT = ROOT / "reports" / "family_class"
API = "http://127.0.0.1:8100/rna/ask"

ACGU = set("ACGU")


def coarse(family: str, name: str) -> str:
    n = (name or "").lower()
    if family == "unassigned":
        return "other"
    if n.startswith("trna"):
        return "tRNA"
    if "ssu_rrna" in n or "lsu_rrna" in n or n in ("5s_rrna", "5_8s_rrna", "5.8s_rrna"):
        return "rRNA"
    return "other"


def ask(seq: str, q: str, max_new: int = 24) -> str:
    body = json.dumps({"sequence": seq, "question": q,
                       "max_new_tokens": max_new, "temperature": 0.0}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as f:
        return json.loads(f.read().decode())["answer"]


def first_nt(ans: str, k: int) -> str:
    s = "".join(c for c in ans.upper() if c in ACGU)
    return s[:k]


def first_int(ans: str) -> int | None:
    m = re.search(r"-?\d+", ans.replace(",", ""))
    return int(m.group()) if m else None


def which_class(ans: str) -> str:
    a = ans.lower()
    if "trna" in a:
        return "tRNA"
    if "rrna" in a:
        return "rRNA"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=4)
    ap.add_argument("--max-nt", type=int, default=250,
                    help="keep the RNA side fast; the service caps at 1024")
    ap.add_argument("--tag", default="gen")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    pools = defaultdict(list)
    with open(SEQS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            s = d["sequence"]
            if not (40 <= len(s) <= args.max_nt):
                continue
            if not set(s) <= ACGU:
                continue
            pools[coarse(d["family"], d["family_name"])].append(s)

    rng = random.Random(7)
    pick = []
    for c in ("rRNA", "tRNA", "other"):
        rng.shuffle(pools[c])
        pick += [(c, s) for s in pools[c][: args.n_per_class]]
    print(f"[sample] {len(pick)} sequences "
          f"(len {min(len(s) for _, s in pick)}-{max(len(s) for _, s in pick)})",
          flush=True)

    # Length-matched control: pair each sequence with another of similar length.
    ctrl = {}
    ordered = sorted(range(len(pick)), key=lambda i: len(pick[i][1]))
    for rank, i in enumerate(ordered):
        j = ordered[(rank + 1) % len(ordered)]
        if j == i:
            j = ordered[(rank + 2) % len(ordered)]
        ctrl[i] = j

    tasks = [
        ("first10", "What are the first 10 nucleotides of this RNA?",
         lambda s: s[:10]),
        ("last10", "What are the last 10 nucleotides of this RNA?",
         lambda s: s[-10:]),
        ("countG", "How many G nucleotides are in this RNA?",
         lambda s: str(s.count("G"))),
        ("class", "Which type of RNA is this: tRNA, rRNA, or other?",
         None),
    ]

    res = {"n_seq": len(pick), "max_nt": args.max_nt, "tasks": {}}
    t0 = time.time()
    for key, q, gold_fn in tasks:
        okA = okB = 0
        rows = []
        for i, (cls, seq) in enumerate(pick):
            cseq = pick[ctrl[i]][1]
            try:
                aA = ask(seq, q)
                aB = ask(cseq, q)
            except Exception as e:
                print(f"  [{key}] {i} ERR {type(e).__name__}: {e}", flush=True)
                continue
            if key == "class":
                hitA = which_class(aA) == cls
                hitB = which_class(aB) == cls      # control asks about the real gold
            elif key == "countG":
                hitA = first_int(aA) == seq.count("G")
                hitB = first_int(aB) == seq.count("G")
            else:
                g = gold_fn(seq)
                hitA = first_nt(aA, 10) == g
                hitB = first_nt(aB, 10) == g
            okA += hitA
            okB += hitB
            rows.append({"cls": cls, "gold": (gold_fn(seq) if gold_fn else cls),
                         "A": aA, "B": aB, "hitA": bool(hitA), "hitB": bool(hitB)})
            print(f"  [{key}] {i+1}/{len(pick)} {cls:6s} "
                  f"gold={rows[-1]['gold']} A={aA!r} hit={hitA} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        n = len(rows)
        res["tasks"][key] = {"q": q, "n": n, "accA": okA / n if n else 0,
                             "accB": okB / n if n else 0, "rows": rows}
        print(f"[{key}] A={okA}/{n} B={okB}/{n}\n", flush=True)

    lp = OUT / f"B4_{args.tag}.json"
    lp.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {lp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
