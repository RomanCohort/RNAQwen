"""Screen GEO for RNA-immunity datasets (modification sites + innate sensing).

Searches GDS, pulls esummary metadata, scores each Series on the things that
actually decide whether we can turn it into training rows:

  * species (human / mouse only)
  * sample count
  * processed supplementary tables present (CSV/TSV/XLSX/TXT)
  * keyword match in title + summary
  * Series vs single-sample SubSeries

Output: reports/immune_geo_screen.json plus a CSV-friendly summary on stdout.

Usage: python scripts/91_immune_geo_screen.py
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "immune_geo_screen.json"
BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

QUERIES = {
    "m6a": 'm6A MeRIP-seq[All Fields] AND ("Homo sapiens"[Organism] OR "Mus musculus"[Organism])',
    "m5c": 'm5C[All Fields] AND RNA[All Fields] AND bisulfite[All Fields]',
    "rigi": 'RIG-I[All Fields] AND RNA[All Fields] AND (agonist[All Fields] OR activation[All Fields])',
    "tlr_rna": '(TLR7[All Fields] OR TLR8[All Fields]) AND RNA[All Fields]',
}
GOOD_TAXA = {"Homo sapiens", "Mus musculus", "human", "mouse"}
KEYWORDS = [
    "m6a", "m5c", "merip", "mrip", "bisulfite", "modification", "methylation",
    "rig-i", "rigi", "mda5", "tlr7", "tlr8", "innate", "interferon", "immune",
    "immunostimulatory", "agonist", "vaccine", "sensing",
]
TABULAR = ("CSV", "TSV", "XLSX", "TXT", "XLS")


def get(url: str, tries: int = 3) -> dict:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=40) as f:
                return json.loads(f.read().decode("utf-8"))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 + 2 * i)
    return {}


def esearch(term: str, retmax: int = 400) -> list[str]:
    u = f"{BASE}/esearch.fcgi?db=gds&retmode=json&retmax={retmax}&term={urllib.parse.quote(term)}"
    return get(u)["esearchresult"].get("idlist", [])


def esummary(ids: list[str]) -> dict:
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        u = f"{BASE}/esummary.fcgi?db=gds&retmode=json&id={','.join(chunk)}"
        res = get(u).get("result", {})
        for gid in chunk:
            if gid in res:
                out[gid] = res[gid]
        time.sleep(0.4)
    return out


def score(rec: dict) -> tuple[int, list[str]]:
    s, why = 0, []
    taxon = (rec.get("taxon") or "").strip()
    if taxon in GOOD_TAXA or any(t in taxon for t in ("sapiens", "musculus")):
        s += 2; why.append("species")
    n = int(rec.get("n_samples") or 0)
    if n >= 20:
        s += 3; why.append("n>=20")
    elif n >= 8:
        s += 2; why.append("n>=8")
    elif n >= 4:
        s += 1
    supp = (rec.get("suppfile") or "").upper()
    if any(t in supp for t in TABULAR):
        s += 2; why.append("tabular_supp")
    text = ((rec.get("title") or "") + " " + (rec.get("summary") or "")).lower()
    hits = sorted({k for k in KEYWORDS if k in text})
    if hits:
        s += min(3, len(hits)); why.append("kw:" + ",".join(hits[:3]))
    if rec.get("gdstype") == "Expression profiling by high throughput sequencing":
        s += 1; why.append("hts")
    if rec.get("gpl"):
        s += 1
    return s, why


def main() -> int:
    report = {}
    for tag, term in QUERIES.items():
        ids = esearch(term)
        recs = esummary(ids)
        rows = []
        for gid, rec in recs.items():
            if rec.get("entrytype") != "GSE":
                continue
            pts, why = score(rec)
            rows.append({
                "gse": rec.get("accession"),
                "title": (rec.get("title") or "")[:140],
                "taxon": rec.get("taxon"),
                "n_samples": int(rec.get("n_samples") or 0),
                "suppfile": rec.get("suppfile"),
                "gdsType": rec.get("gdstype"),
                "pubmed": rec.get("pubmedids"),
                "score": pts,
                "why": why,
                "summary": (rec.get("summary") or "")[:600],
            })
        rows.sort(key=lambda r: (-r["score"], -r["n_samples"]))
        report[tag] = rows
        print(f"=== {tag}: {len(rows)} Series ===")
        for r in rows[:8]:
            print(f"  {r['score']:2d} {r['gse']} n={r['n_samples']:<3d} {r['taxon'][:20]:20s} {r['title'][:78]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())