"""Per-window RNA structure features via ViennaRNA.

Used two ways:
  * auxiliary supervision (predict ΔG / paired fraction from the code channel)
  * input features once we have an immune label to attach them to

Outputs one JSON object per input row with a `struct` block.  Nothing here is
ground-truth structure -- it is the ViennaRNA MFE / ensemble summary and must be
labelled as computed, not measured, in any report.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import RNA


def features(seq: str) -> dict:
    seq = seq.strip().upper().replace("T", "U")
    if len(seq) < 10 or any(c not in "ACGU" for c in seq):
        return {}
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe = fc.mfe()
    _, ens = fc.pf()
    bpp = fc.bpp()
    n = len(seq)
    paired = sum(1 for i in range(1, n + 1) for j in range(i + 1, n + 1)
                 if bpp[i][j] > 0.5)
    # longest helical run in the MFE dot-bracket
    best = cur = 0
    for ch in mfe_struct:
        if ch in "([{":
            cur += 1; best = max(best, cur)
        else:
            cur = 0
    # accessibility: mean unpaired probability over a 4-nt sliding window (min = most open)
    acc = [1.0 - sum(bpp[i][j] for j in range(1, n + 1) if j != i) for i in range(1, n + 1)]
    acc = [max(0.0, min(1.0, x)) for x in acc]
    k = 4
    windows = [sum(acc[i:i + k]) / k for i in range(0, max(1, n - k + 1))]
    return dict(
        mfe=round(mfe, 3),
        mfe_per_nt=round(mfe / n, 4),
        ens=round(ens, 3),
        paired_frac=round(2 * paired / n, 4),
        longest_stem=best,
        acc_min=round(min(windows), 4) if windows else 0.0,
        acc_mean=round(sum(acc) / n, 4),
        gc=round(sum(1 for c in seq if c in "GC") / n, 4),
        len=n,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=400)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()

    t0 = time.time()
    done = skipped = 0
    rows = []
    with open(a.src, encoding="utf-8") as fh, open(a.out, "w", encoding="utf-8") as out:
        idx = -1
        for line in fh:
            idx += 1
            if a.shards > 1 and idx % a.shards != a.shard:
                continue
            if a.limit and done >= a.limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            seq = r.get("sequence") or ""
            if len(seq) > a.max_len:
                skipped += 1
                continue
            f = features(seq)
            if not f:
                skipped += 1
                continue
            r["struct"] = f
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            done += 1
            if done % 100 == 0:
                print(f"  {done} rows  {time.time() - t0:.1f}s", flush=True)
    dt = time.time() - t0
    print(f"done {done} skipped {skipped} in {dt:.1f}s ({done / max(dt, 1e-9):.1f} rows/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())