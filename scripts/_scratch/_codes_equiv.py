"""Are the codes the frontend computes the codes the adapter was trained on?

Training fed the LLM `rna_emb[code]` with `code` taken from cached ERNIE-RNA hidden
states (data/processed/_fsq_full_pos.pt).  The shipped path recomputes ERNIE-RNA from
the sequence and re-runs the same FSQ.  Codes are a step function of that hidden
state, so this is a clean pass/fail: either the shipped codes match the trained ones,
or the GGUF is being fed inputs the adapter never saw and 2.6 is a floor.

    python scripts/_scratch/_codes_equiv.py --n 600

--max-nt 1024 recomputes on the full cached sequence; the shipped frontend defaults
to 512, so the run also reports how many cached sequences are longer than that.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "gguf"
HID = ROOT / "data/processed/_fsq_full_pos.pt"


def spread(n: int, k: int) -> list:
    if k >= n:
        return list(range(n))
    st = n / k
    return [min(int(i * st), n - 1) for i in range(k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--max-nt", type=int, default=1024)
    ap.add_argument("--out", default="reports/codes_equiv.json")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("rp", GG / "rna_prompt.py")
    rp = importlib.util.module_from_spec(spec)
    sys.modules["rp"] = rp
    spec.loader.exec_module(rp)
    p = rp._parts()
    enc = p["enc"]
    rc = rp.rc

    recs = rc.load_records(None, sort=False)
    d = torch.load(HID, map_location="cpu", weights_only=False, mmap=True)
    lens, offs = d["lens"], d["offsets"]
    print(f"[data] records {len(recs):,}  cache {tuple(d['x'].shape)}", flush=True)

    lens_py = [int(x) for x in lens]
    over = sum(1 for s in (r["sequence"] for r in recs) if len(s) > 512)
    print(f"[data] sequences longer than the frontend's 512 nt default: {over:,}"
          f" / {len(recs):,}", flush=True)

    idxs = spread(len(recs), args.n)
    rows, mism_seq, mism_pos, tot_pos = [], 0, 0, 0
    for j, i in enumerate(idxs):
        r = recs[i]
        seq = (r["sequence"] or "").strip().upper()
        off, ln = int(offs[i]), lens_py[i]
        with torch.no_grad():
            hc = d["x"][off:off + ln].to(rp.DEV, dtype=torch.float32)
            code_cache = enc(hc)[0].tolist()
            code_front = rp.codes_of(seq, args.max_nt)
        n = min(len(code_cache), len(code_front))
        bad = [k for k in range(n) if code_cache[k] != code_front[k]]
        extra = abs(len(code_cache) - len(code_front))
        rows.append(dict(uid=r["uid"], nt=len(seq), cache_pos=ln, codes=n,
                         mismatch=len(bad) + extra, first=bad[0] if bad else None))
        mism_seq += bool(bad) or extra > 0
        mism_pos += len(bad)
        tot_pos += n
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(idxs)}  sequences with a flipped code: "
                  f"{mism_seq}  positions {mism_pos}/{tot_pos}", flush=True)

    print(f"\n对比 {len(idxs)} 条序列（max_nt={args.max_nt}）")
    print(f"  有翻转的序列 {mism_seq} / {len(idxs)}  ({mism_seq / len(idxs):.2%})")
    print(f"  翻转的码位   {mism_pos} / {tot_pos}  ({mism_pos / max(tot_pos, 1):.3%})")
    worst = sorted(rows, key=lambda x: -x["mismatch"])[:5]
    for w in worst:
        print(f"    {w['uid'][:40]:<42} nt {w['nt']:>4} 码 {w['codes']:>4} "
              f"翻转 {w['mismatch']:>4} 首个 {w['first']}")

    out = ROOT / args.out
    out.write_text(json.dumps(dict(n=len(idxs), max_nt=args.max_nt,
                                   sequences_with_flip=mism_seq,
                                   positions_flipped=mism_pos, positions=tot_pos,
                                   longer_than_512=over, rows=rows),
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
