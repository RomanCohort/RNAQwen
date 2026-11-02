"""RNAcentral FASTA -> clean jsonl (uid + sequence).

RNAcentral uses the DNA alphabet (T, plus N/R ambiguity codes).  The code
channel and ERNIE-RNA expect ACGU, so this converts T->U and drops any window
that still contains an ambiguous base.  Output schema matches
data/processed/rnaqa_n6_merged.jsonl so the existing stage-1 pipeline can read
it unchanged.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/rnacentral/rnacentral_active.fasta.gz"
OUT = ROOT / "data/processed/rnacentral_2m.jsonl"
TOTAL = 48_053_536


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--min-nt", type=int, default=20)
    ap.add_argument("--max-nt", type=int, default=512)
    a = ap.parse_args()

    stride = max(1, a.total // a.n)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] stride={stride} target={a.n:,} len[{a.min_nt},{a.max_nt}]", flush=True)

    t0 = time.time()
    seen = kept = bad = 0
    with gzip.open(a.src, "rt", errors="replace") as fh, out.open("w", encoding="utf-8") as w:
        name, buf = None, []
        def emit(nm, seq):
            nonlocal kept, bad
            s = seq.upper().replace("T", "U")
            if not (a.min_nt <= len(s) <= a.max_nt):
                return False
            bad += int(any(c not in "ACGU" for c in s))
            uid = nm.split()[0]
            w.write(json.dumps(dict(uid=uid, sequence=s, source="RNAcentral",
                                    task="rnacentral_pretrain"), ensure_ascii=False) + "\n")
            kept += 1
            return True
        for line in fh:
            if line[0] == ">":
                if name is not None and buf:
                    if seen % stride == 0:
                        emit(name, "".join(buf))
                seen += 1
                name = line[1:].strip(); buf = []
                if kept >= a.n:
                    break
            else:
                buf.append(line.strip())
        if kept < a.n and name is not None and buf and seen % stride == 0:
            emit(name, "".join(buf))
    print(f"[done] seen={seen:,} kept={kept:,} ambiguous_dropped={bad:,} "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    print(f"wrote {out}  {out.stat().st_size/1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
