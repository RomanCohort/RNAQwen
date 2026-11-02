"""调试：反向链路 —— 短 RNA（nuccore）→ 文献（pubmed）→ 摘要。

正向链路（pubmed→nuccore）实测拿到的都是 2–4 kb 的完整 mRNA，
全部超过 RNA-FM 的 1024 nt 上限，被过滤光了。
所以改用反向：**先找短 RNA 序列**，再反向找它们的文献。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.sources import Entrez, parse_fasta, parse_pubmed_xml  # noqa: E402

QUERIES = {
    "refseq_ncrna_short": 'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
                          '("non-coding RNA"[Title] OR "lncRNA"[Title] OR "miRNA"[Title] '
                          'OR "snoRNA"[Title] OR "riboswitch"[Title] OR "ribozyme"[Title])',
    "mirna_precursor": '"miRNA"[Title] AND 60:300[slen] AND biomol_rna[prop]',
    "riboswitch": 'riboswitch[Title] AND 60:600[slen] AND biomol_rna[prop]',
    "trna": '"tRNA"[Title] AND 60:200[slen] AND biomol_rna[prop]',
}


def main() -> int:
    ez = Entrez()
    for name, q in QUERIES.items():
        ids = ez.esearch("nuccore", q, retmax=5)
        print(f"\n=== {name} === term={q[:90]}...")
        print(f"  前 5 个 nuccore id: {ids}")
        if not ids:
            continue
        fasta = parse_fasta(ez.efetch("nuccore", ids, retmode="text", rettype="fasta"))
        for acc, defn, seq in fasta:
            print(f"    {acc:16s} len={len(seq):5d} {defn[:80]}")
        links = ez.elink("nuccore", "pubmed", ids)
        n_links = sum(len(v) for v in links.values())
        print(f"    -> pubmed 链接数: {n_links}  ({ {k: v[:3] for k, v in list(links.items())[:3]} })")
        pmids = sorted({p for v in links.values() for p in v})[:3]
        if pmids:
            lit = parse_pubmed_xml(ez.efetch("pubmed", pmids, retmode="xml"))
            for pmid, meta in list(lit.items())[:2]:
                print(f"    PMID {pmid}: abstract={len(meta['abstract'])} 字符 | {meta['title'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
