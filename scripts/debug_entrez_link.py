"""调试：单条 PMID 走完 pubmed → nuccore 的链路，看每一步拿到什么。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.sources import Entrez, parse_fasta, parse_pubmed_xml  # noqa: E402


def main() -> int:
    pmid = sys.argv[1] if len(sys.argv) > 1 else "33087357"
    ez = Entrez()
    print("== 1. efetch pubmed ==")
    xml = ez.efetch("pubmed", [pmid], retmode="xml")
    lit = parse_pubmed_xml(xml)
    print("parsed:", json.dumps({k: (v[:80] if isinstance(v, str) else v) for k, v in lit.get(pmid, {}).items()},
                                ensure_ascii=False, indent=2))

    print("\n== 2. elink pubmed -> nuccore ==")
    raw = ez._get("elink.fcgi", {"dbfrom": "pubmed", "db": "nuccore", "retmode": "json", "id": pmid})
    print(raw[:800])
    links = ez.elink("pubmed", "nuccore", [pmid])
    print("parsed links:", links)
    nt_ids = sorted({n for v in links.values() for n in v})
    print("nt ids:", nt_ids[:20])

    if nt_ids:
        print("\n== 3. efetch nuccore fasta ==")
        fasta = ez.efetch("nuccore", nt_ids[:5], retmode="text", rettype="fasta")
        recs = parse_fasta(fasta)
        for acc, defn, seq in recs:
            print(f"  {acc:20s} len={len(seq):7d} {defn[:90]}")
            print(f"      {seq[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
