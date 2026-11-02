"""Extract RNA chains + curated base pairs from RCSB/NDB mmCIF (v2).

Sequences come from _pdbx_poly_seq_scheme (mon_id in ACGU = RNA), base pairs
from _ndb_struct_na_base_pair (label chain + label seq id).  Dot-bracket is
therefore experimental, not predicted.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

RNA = {"A", "C", "G", "U"}


def rows(path: Path, prefix: str):
    with gzip.open(path, "rt", errors="replace") as fh:
        lines = fh.readlines()
    cols, i, n = None, 0, len(lines)
    while i < n:
        if lines[i].startswith(prefix + "."):
            cols = []
            while i < n and lines[i].startswith(prefix + "."):
                cols.append(lines[i].strip().split(".")[-1]); i += 1
            while i < n and not lines[i].startswith("#"):
                if not lines[i].strip():
                    i += 1; continue
                vals = []
                while i < n and not lines[i].startswith("#"):
                    parts = lines[i].rstrip("\n").split()
                    if not parts:
                        i += 1; break
                    vals.extend(parts); i += 1
                    if len(vals) >= len(cols):
                        break
                if len(vals) >= len(cols):
                    yield dict(zip(cols, vals[:len(cols)]))
            return
        i += 1


def to_db(ids: list[int], pairs: list[tuple[int, int]]) -> str:
    pos = {sid: k for k, sid in enumerate(ids)}
    s = ["."] * len(ids)
    for a, b in pairs:
        if a in pos and b in pos and pos[a] != pos[b]:
            lo, hi = sorted((pos[a], pos[b]))
            s[lo], s[hi] = "(", ")"
    return "".join(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-pairs", type=int, default=3)
    a = ap.parse_args()

    kept = 0; entries = set()
    with open(a.out, "w", encoding="utf-8") as out:
        for path in sorted(Path(a.dir).glob("*.cif.gz")):
            entry = path.name.split(".")[0]
            try:
                sch = list(rows(path, "_pdbx_poly_seq_scheme"))
                bps = list(rows(path, "_ndb_struct_na_base_pair"))
            except Exception as exc:          # noqa: BLE001
                print(f"  skip {entry}: {exc}")
                continue
            chains = defaultdict(list)
            for r in sch:
                if r.get("mon_id") in RNA:
                    try:
                        chains[r["asym_id"]].append((int(r["seq_id"]), r["mon_id"]))
                    except Exception:
                        pass
            # intra-chain pairs (single-strand secondary structure)
            pairs = defaultdict(list)
            for r in bps:
                if r.get("i_label_comp_id") not in RNA or r.get("j_label_comp_id") not in RNA:
                    continue
                if r.get("i_label_asym_id") != r.get("j_label_asym_id"):
                    continue
                try:
                    pairs[r["i_label_asym_id"]].append((int(r["i_label_seq_id"]), int(r["j_label_seq_id"])))
                except Exception:
                    pass
            for chain, items in chains.items():
                items.sort()
                ids = [x[0] for x in items]
                seq = "".join(x[1] for x in items)
                db = to_db(ids, pairs.get(chain, []))
                if db.count("(") < a.min_pairs:
                    continue
                kept += 1; entries.add(entry)
                out.write(json.dumps({
                    "uid": f"pdb:{entry}:{chain}", "entry": entry, "chain": chain,
                    "sequence": seq, "structure": db, "n_pairs": db.count("("),
                    "task": "pdb_struct", "source": "RCSB/NDB", "family": entry,
                }, ensure_ascii=False) + "\n")

            # inter-chain duplexes: dsRNA is the immunogenic form, keep it
            duplex = [r for r in bps
                      if r.get("i_label_comp_id") in RNA and r.get("j_label_comp_id") in RNA
                      and r.get("i_label_asym_id") != r.get("j_label_asym_id")]
            if duplex:
                involved = sorted({r["i_label_asym_id"] for r in duplex} |
                                  {r["j_label_asym_id"] for r in duplex})
                if len(involved) == 2 and all(c in chains for c in involved):
                    off = {}
                    seqs = []
                    acc = 0
                    for c in involved:
                        items = sorted(chains[c])
                        off[c] = {sid: acc + k for k, (sid, _m) in enumerate(items)}
                        seqs.append("".join(m for _s, m in items))
                        acc += len(items)
                    total = sum(len(s) for s in seqs)
                    s = ["."] * total
                    npair = 0
                    for r in duplex:
                        try:
                            ia = off[r["i_label_asym_id"]][int(r["i_label_seq_id"])]
                            jb = off[r["j_label_asym_id"]][int(r["j_label_seq_id"])]
                        except Exception:
                            continue
                        lo, hi = sorted((ia, jb))
                        if s[lo] == "." and s[hi] == ".":
                            s[lo], s[hi] = "(", ")"; npair += 1
                    if npair >= a.min_pairs:
                        kept += 1; entries.add(entry)
                        out.write(json.dumps({
                            "uid": f"pdb:{entry}:duplex:{'+'.join(involved)}", "entry": entry,
                            "chain": "+".join(involved), "sequence": "&".join(seqs),
                            "structure": "".join(s), "n_pairs": npair,
                            "task": "pdb_duplex", "source": "RCSB/NDB", "family": entry,
                        }, ensure_ascii=False) + "\n")
    print(f"entries={len(entries)} chains={kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())