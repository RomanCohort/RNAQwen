"""P0 第一步：采集 (RNA 序列, 文献) 记录。

最终链路（实测选出来的，别改回去）
--------------------------------
1. `esearch nuccore` —— 找**短 RNA 序列**（RefSeq ncRNA + 长度闸门）
2. `efetch nuccore rettype=gb` —— GenBank 平文本，
   **一次请求同时拿到序列 + DEFINITION + ORGANISM + PUBMED**
3. `efetch pubmed` —— 批量拿摘要
4. 过滤 → 落 JSONL

为什么不用 elink：
    elink 传**多个** UID 时会合并成一个 linkset，不区分哪条链接属于哪条序列
    （实测 5 个 id 的 pubmed 链接全挤在一个 linkset 里）。
    GenBank 记录里自带 `PUBMED` 字段 —— 一一对应，无歧义，还省一半请求。

为什么不用正向 pubmed→nuccore：
    正向拿到的全是 2–4 kb 完整 mRNA（NM_*），越过 RNA-FM 的 1024 nt 上限，
    全部被长度闸门挡掉 —— 半小时一条都收不到。

过滤标准（写死，别让"数据质量差"变成事后无法归因的黑盒）
    * 序列长度 30–1024 nt（RNA-FM 上下文 1026 − 2 个特殊 token）
    * 字母表 ⊆ {A,C,G,U}（允许 T，归一成 U）
    * GenBank 里至少一个 PUBMED，且摘要 >= 200 字符

跑法：
    python scripts/10_collect.py --target 50   --out data/raw/rna_literature_50.jsonl
    python scripts/10_collect.py --target 1000 --out data/raw/rna_literature_1k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.records import RnaQaRecord, write_jsonl  # noqa: E402
from rnaqwen.data.sources import Entrez, parse_genbank, parse_pubmed_xml  # noqa: E402

# RNA-FM 的**实测**上限是 1022 nt（1024 个 token 含 BOS/EOS）。
# config 里的 max_position_embeddings=1026 不是真正生效的那个数 ——
# 见 scripts/01b_probe_rnafm_limit.py 的逐长度边界测试。
MIN_NT, MAX_NT = 30, 1022
MIN_ABSTRACT = 200

# 多路查询：保证 RNA 类型有分布，不是清一色 lncRNA
#
# ⚠️ 为什么条目这么多：`esearch` 的 relevance 排序是**确定性**的，
#    同一条 query 反复跑永远从同一片记录开始；而 RefSeq 记录的 GenBank `PUBMED`
#    字段又大量指向少数几篇"大论文"。结果就是 —— 不加闸门时 1000 条里只有 24 个
#    不同 pmid，其中一篇挂了 800 条（详见 PROGRESS.md 的"文献复用 98%"）。
#    对策是**横向铺开 query**（不同 RNA 类型 × 不同物种 × 不同年份），
#    而不是把同一条 query 的 retstart 一直往后翻。
DEFAULT_QUERIES: list[tuple[str, str]] = [
    (
        "ncrna_refseq",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        '("non-coding RNA"[Title] OR "lncRNA"[Title] OR "long non-coding"[Title] '
        'OR "circular RNA"[Title] OR "circRNA"[Title])',
    ),
    (
        "mirna_snrna",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        '("miRNA"[Title] OR "microRNA"[Title] OR "snoRNA"[Title] OR "snRNA"[Title])',
    ),
    (
        "ribozyme_aptamer",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 40:1000[slen] AND '
        '("ribozyme"[Title] OR "aptamer"[Title] OR "riboswitch"[Title] OR "IRES"[Title])',
    ),
    (
        "trna_rrna",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        '("tRNA"[Title] OR "rRNA"[Title] OR "ribosomal RNA"[Title] OR "SRP RNA"[Title])',
    ),
    (
        "viral_structured",
        'biomol_rna[prop] AND 60:1000[slen] AND srcdb_refseq[prop] AND '
        '("HCV"[Title] OR "SARS"[Title] OR "influenza"[Title] OR "HIV"[Title])',
    ),
    (
        "human_lncrna",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        'Homo sapiens[Organism] AND ("lncRNA"[Title] OR "antisense"[Title] OR "pseudogene"[Title])',
    ),
    (
        "mouse_lncrna",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        'Mus musculus[Organism] AND ("lncRNA"[Title] OR "antisense"[Title] OR "non-coding"[Title])',
    ),
    (
        "zebrafish_drosophila",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        '(Danio rerio[Organism] OR Drosophila[Organism]) AND ("non-coding"[Title] OR "lncRNA"[Title] OR "miRNA"[Title])',
    ),
    (
        "plant_fungal",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND '
        '(Arabidopsis[Organism] OR Oryza[Organism] OR Saccharomyces[Organism]) AND '
        '("non-coding"[Title] OR "lncRNA"[Title] OR "miRNA"[Title] OR "rRNA"[Title])',
    ),
    (
        "bacterial_srna",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 50:1000[slen] AND '
        '(Escherichia[Organism] OR Bacillus[Organism] OR Pseudomonas[Organism]) AND '
        '("sRNA"[Title] OR "small RNA"[Title] OR "riboswitch"[Title] OR "ncRNA"[Title])',
    ),
    # 年份切片：同一条 query 换时间段 = 换出一批完全不同的记录
    (
        "ncrna_2010_2015",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND 2010:2015[dp] AND '
        '("non-coding"[Title] OR "lncRNA"[Title] OR "miRNA"[Title])',
    ),
    (
        "ncrna_2016_2020",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND 2016:2020[dp] AND '
        '("non-coding"[Title] OR "lncRNA"[Title] OR "miRNA"[Title] OR "circRNA"[Title])',
    ),
    (
        "ncrna_2021_2026",
        'biomol_rna[prop] AND srcdb_refseq[prop] AND 60:1000[slen] AND 2021:2026[dp] AND '
        '("non-coding"[Title] OR "lncRNA"[Title] OR "miRNA"[Title] OR "circRNA"[Title] '
        'OR "snoRNA"[Title] OR "piRNA"[Title])',
    ),
]


def looks_like_rna(seq: str) -> bool:
    s = seq.upper().replace("T", "U")
    # 允许少量 IUPAC 简并碱基（N/R/Y…），但要求主体是 ACGU
    core = s.replace("N", "").replace("R", "").replace("Y", "").replace("K", "").replace("M", "")
    core = core.replace("S", "").replace("W", "").replace("B", "").replace("D", "").replace("H", "").replace("V", "")
    if not core:
        return False
    return set(core) <= set("ACGU") and len(core) / max(len(s), 1) >= 0.95


_STOP = {
    "rna", "mrna", "ncrna", "lncrna", "non", "coding", "long", "complete", "sequence", "gene",
    "transcript", "variant", "predicted", "homo", "sapiens", "mus", "musculus", "danio", "rerio",
    "drosophila", "melanogaster", "caenorhabditis", "elegans", "escherichia", "coli", "and",
    "the", "putative", "protein", "partial", "clone", "chromosome", "isoform", "precursor",
}


def extract_key_terms(definition: str, min_len: int = 4, max_terms: int = 6) -> list[str]:
    """从 DEFINITION 里抠出"这条 RNA 自己的标识"。

    优先取括号里的代号（如 `(Lncpnky)`、`(F09A5.15)`、`(mcmdc2as)`），
    再退回到长度 >= min_len 且不在停用词表里的词。
    这些词必须出现在摘要里，才算"这篇文献确实在讲这条 RNA"。
    """
    import re as _re2

    terms: list[str] = []
    for m in _re2.finditer(r"\(([^()]{2,40})\)", definition):
        inner = m.group(1).strip()
        if inner and not inner.isdigit():
            terms.append(inner.lower())
    for tok in _re2.findall(r"[A-Za-z0-9_\-\.]{4,}", definition or ""):
        t = tok.lower().strip(".")
        if t and t not in _STOP and not t.isdigit():
            terms.append(t)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:max_terms]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=50, help="目标记录数")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw" / "rna_literature.jsonl")
    ap.add_argument("--query", default=None, help="覆盖默认查询（单条）")
    ap.add_argument("--batch", type=int, default=100, help="每次 esearch 抓多少 nuccore id")
    ap.add_argument("--max-batches", type=int, default=8,
                    help="每条 query 最多翻几页。默认 8 —— 翻多了没用：同一条 query 的记录"
                         "高度聚集（同一个 pmid 反复出现），配额一满后面全是空转。"
                         "宁可横向换 query。")
    ap.add_argument("--max-pmid-per-seq", type=int, default=10)
    ap.add_argument("--max-per-pmid", type=int, default=2,
                    help="同一篇文献最多给几条记录。默认 2 —— 一篇论文讲一条 RNA 很常见，"
                         "给 2 条留一点余量；给 1 太紧会把产出压没。"
                         "设 0 表示不限（就是 50 条那一版的坏行为：1000 条只对应 24 篇文献）")
    ap.add_argument("--require-mention", action="store_true", default=True,
                    help="要求 DEFINITION 里的特征标识出现在摘要里（弱相关性检查）")
    ap.add_argument("--no-require-mention", dest="require_mention", action="store_false")
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()
    if args.max_per_pmid == 0:
        args.max_per_pmid = 10 ** 9

    queries = [("custom", args.query)] if args.query else DEFAULT_QUERIES
    ez = Entrez()
    records: list[RnaQaRecord] = []
    seen_acc: set[str] = set()
    seen_nt_id: set[str] = set()
    stats = {"nt_scanned": 0, "bad_len": 0, "bad_alpha": 0, "no_pubmed": 0,
             "short_abstract": 0, "no_abstract": 0, "dup": 0, "kept": 0,
             "pmid_quota": 0, "low_relevance": 0}
    pmid_counts: dict[str, int] = {}

    for qname, query in queries:
        if len(records) >= args.target:
            break
        print(f"\n===== query: {qname} =====", flush=True)
        for batch_i in range(args.max_batches):
            if len(records) >= args.target:
                break
            ids = ez.esearch("nuccore", query, retmax=args.batch, retstart=batch_i * args.batch)
            ids = [i for i in ids if i not in seen_nt_id]
            if not ids:
                print(f"  [batch {batch_i}] 没结果了，换下一条 query")
                break
            seen_nt_id.update(ids)
            stats["nt_scanned"] += len(ids)

            recs = parse_genbank(ez.efetch("nuccore", ids, retmode="text", rettype="gb"))
            wanted = []
            for r in recs:
                s = r.sequence.upper().replace("T", "U")
                if not (MIN_NT <= len(s) <= MAX_NT):
                    stats["bad_len"] += 1
                    continue
                if not looks_like_rna(r.sequence):
                    stats["bad_alpha"] += 1
                    continue
                if not r.pubmeds:
                    stats["no_pubmed"] += 1
                    continue
                wanted.append(r)
            if not wanted:
                print(f"  [batch {batch_i}] ids={len(ids)} 无可用记录（都卡在长度/字母表/无文献）")
                continue

            pmids = sorted({p for r in wanted for p in r.pubmeds[: args.max_pmid_per_seq]})
            lit = parse_pubmed_xml(ez.efetch("pubmed", pmids[:400], retmode="xml"))

            for r in wanted:
                if len(records) >= args.target:
                    break
                if r.accession in seen_acc:
                    stats["dup"] += 1
                    continue
                best_abs, best_pmid = "", ""
                for pmid in r.pubmeds[: args.max_pmid_per_seq]:
                    meta = lit.get(pmid)
                    if not meta:
                        continue
                    if len(meta["abstract"]) > len(best_abs):
                        best_abs, best_pmid = meta["abstract"], pmid
                if not best_pmid:
                    stats["no_abstract"] += 1
                    continue
                if len(best_abs) < MIN_ABSTRACT:
                    stats["short_abstract"] += 1
                    continue

                # ---- 相关性闸门（50 条那一版就是在这里翻车的，别删）----
                # 实测：不加闸门时 50 条记录只对应 **7 篇文献**，其中 42 条共用同一篇
                # C. elegans 基因组论文（PMID 9851916）—— 也就是说 84% 的"文献"跟这条
                # RNA 没关系。用这种数据训 QA，模型只会学会复述那篇论文。
                #
                # 两道闸：
                #   1. 每篇文献最多给 max_per_pmid 条记录（默认 1）
                #   2. 描述行的**特征标识**必须出现在摘要里（弱相关性检查）
                if pmid_counts.get(best_pmid, 0) >= args.max_per_pmid:
                    stats["pmid_quota"] += 1
                    continue
                if args.require_mention:
                    key = extract_key_terms(r.definition)
                    haystack = (best_abs + " " + lit[best_pmid].get("title", "")).lower()
                    if key and not any(t in haystack for t in key):
                        stats["low_relevance"] += 1
                        continue
                pmid_counts[best_pmid] = pmid_counts.get(best_pmid, 0) + 1
                seen_acc.add(r.accession)
                records.append(
                    RnaQaRecord(
                        uid=f"nt:{r.accession}|pmid:{best_pmid}",
                        sequence=r.sequence.upper().replace("T", "U"),
                        organism=r.organism,
                        definition=r.definition,
                        literature=best_abs,
                        pmid=best_pmid,
                        nuc_accession=r.accession,
                        provenance={
                            "source": "NCBI Entrez (nuccore[gb]→pubmed)",
                            "query_name": qname,
                            "query": query,
                            "collected": time.strftime("%Y-%m-%d"),
                            "title": lit[best_pmid]["title"],
                            "journal": lit[best_pmid]["journal"],
                            "year": lit[best_pmid]["year"],
                            "molecule_type": r.molecule_type,
                            "n_pubmed": len(r.pubmeds),
                        },
                    )
                )
                stats["kept"] += 1
            print(f"  [batch {batch_i}] ids={len(ids)} gb={len(recs)} 可用={len(wanted)} "
                  f"pmids={len(pmids)} 已收 {len(records)}/{args.target}", flush=True)
            time.sleep(args.sleep)

    n = write_jsonl(args.out, records)
    print(f"\n[ok] 写入 {n} 条 -> {args.out}")
    print("[stats]", json.dumps(stats, ensure_ascii=False))
    if records:
        lens = sorted(len(r.sequence) for r in records)
        print(f"[lens] min={lens[0]} median={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
        orgs: dict[str, int] = {}
        for r in records:
            orgs[r.organism] = orgs.get(r.organism, 0) + 1
        print("[organisms] top:", sorted(orgs.items(), key=lambda kv: -kv[1])[:6])
        r = records[0]
        print("[sample]")
        print("  uid       :", r.uid)
        print("  seq len   :", len(r.sequence))
        print("  seq head  :", r.sequence[:70])
        print("  definition:", r.definition[:110])
        print("  abstract  :", r.literature[:170], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
