"""Restrict RNAQA v3 to the sequences that already have cached ERNIE states.

`_rfa_full_index.pt` describes 145,201 Rfam sequences (47.5M positions, 68 GB)
but the matching `_rfa_full_hidden.f16` is gone; only the 19,726-sequence
`_rfam_hidden.f16` (9.56 GB) survives.  Its uids are a strict subset of the
full index, so a pilot can run on those without re-encoding anything.

Pure Python, no torch.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/processed/rnaqa_v3_l234.jsonl"))
    ap.add_argument("--uids", default=str(ROOT / "data/processed/_rfam_index.pt"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/rnaqa_v3_pilot.jsonl"))
    a = ap.parse_args()

    # torch.save uses a zip container, so raw pickle cannot read it.
    import torch
    d = torch.load(a.uids, map_location="cpu", weights_only=False)
    keep = set(d["uids"])
    print(f"[uids] cached={len(keep):,}", flush=True)

    n_in = n_out = 0
    seqs = set()
    tpl = Counter()
    sp = Counter()
    out = Path(a.out)
    with open(a.src, encoding="utf-8") as fh, out.open("w", encoding="utf-8") as w:
        for line in fh:
            n_in += 1
            r = json.loads(line)
            uid = r["uid"]
            key = uid[4:] if uid.startswith("seq:") else uid
            if key not in keep:
                continue
            # The rfam hidden cache is keyed by the bare md5 hash, so the uid
            # written here must lose the `seq:` prefix or every lookup misses.
            r["uid"] = key
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
            seqs.add(key)
            tpl[r["template"]] += 1
            sp[r["split"]] += 1
    print(f"[rows] {n_in:,} -> {n_out:,}   sequences={len(seqs):,}", flush=True)
    print(f"[split] {dict(sp)}", flush=True)
    for t, c in tpl.most_common():
        print(f"  {t:20s} {c:>7,}", flush=True)
    print(f"[save] {out}  {out.stat().st_size/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
