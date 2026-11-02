"""Rfam structure pass: multiprocessing pool, parent writes (WMI-safe)."""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "sf", str(Path(__file__).resolve().parent / "94_structure_features.py"))
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


def work(args):
    i, line = args
    try:
        r = json.loads(line)
    except Exception:
        return None
    seq = r.get("sequence") or ""
    if not seq or len(seq) > 400:
        return None
    f = sf.features(seq)
    if not f:
        return None
    r["struct"] = f
    return r


def main() -> int:
    src = Path(sys.argv[1]); out = Path(sys.argv[2]); workers = int(sys.argv[3])
    t0 = time.time(); n = 0
    with open(src, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as w:
        lines = ((i, line) for i, line in enumerate(fh) if line.strip())
        with Pool(workers) as pool:
            for r in pool.imap(work, lines, chunksize=32):
                if r is None:
                    continue
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
                if n % 1000 == 0:
                    w.flush()
                    print(f"  {n} rows {time.time() - t0:.0f}s", flush=True)
        w.flush()
    print(f"done {n} rows in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())