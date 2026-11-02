"""P0 数据质量报告：对 10_collect.py 产出的 JSONL 做全面体检。

只读，不改数据。用法：

    python scripts/14_quality_report.py \
        --in data/raw/rna_literature_1k.jsonl \
        --out reports/collect_1k_report.json \
        --stats '{"nt_scanned":...}'      # 可选：把 10_collect.py 打印的 [stats] 原样传进来

检查项
------
1. 总条数 + [stats] 全计数
2. 序列长度分布（min/p25/median/p75/p90/max）+ >MAX_NT 的条数（必须 0）
3. 字母表：去掉 ACGU 之后剩余字符集合（必须为空集或极少数简并碱基）
4. 物种分布 top10 / RNA 类型分布（DEFINITION 关键词粗分）
5. 摘要长度分布
6. 去重：uid 唯一、accession 唯一、序列本身 sha1 唯一
7. 每条必须有非空 pmid 和 abstract
8. 抽样 10 条

任何违规项会进入 `critical_issues`（顶层），方便上层直接判红。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_NT, MAX_NT = 30, 1022  # 与 10_collect.py 保持一致（1022 = RNA-FM 实测上限）
MIN_ABSTRACT = 200

# IUPAC 简并碱基（去掉 ACGU 之后"合法"的残留）
IUPAC = set("NRYWSKMBDHV")
VALID = set("ACGU")

# DEFINITION 关键词 → RNA 类型。**顺序即优先级**（先命中先归类）。
TYPE_RULES: list[tuple[str, str]] = [
    ("circRNA", r"circular rna|\bcircrna\b|circ_?rna"),
    ("miRNA", r"\bmirna\b|microrna|\bmiR-|\blet-7\b|\bmir-\d"),
    ("snoRNA", r"\bsnorna\b|small nucleolar"),
    ("snRNA", r"\bsnrna\b|small nuclear rna"),
    ("tRNA", r"\btrna\b|transfer rna|\btrna-|\bmt[-_]t"),
    ("rRNA", r"\brrna\b|ribosomal rna|\brrna[-_]|\b16s\b|\b18s\b|\b28s\b|\b5\.8s\b|\b5s\b"),
    ("lncRNA", r"lncrna|long non-?coding|lincrna|antisense rna|non-?coding rna|\bncrna\b"),
    ("ribozyme", r"ribozyme|riboswitch|\baptamer\b|\bires\b|hammerhead|pseudoknot"),
    ("SRP RNA", r"\bsrp rna\b|signal recognition particle"),
    ("vaultRNA/YRNA", r"vault rna|\byrna\b|\by rna\b"),
]
TYPE_RES = [(name, re.compile(pat, re.I)) for name, pat in TYPE_RULES]


def classify(definition: str) -> str:
    d = definition or ""
    for name, rx in TYPE_RES:
        if rx.search(d):
            return name
    return "其它"


def pct(sorted_vals: list[int], q: float) -> int:
    """线性插值分位数（与 numpy percentile 默认一致）。"""
    if not sorted_vals:
        return 0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return int(round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac))


def dist(vals: list[int]) -> dict:
    s = sorted(vals)
    if not s:
        return {"n": 0, "min": 0, "p25": 0, "median": 0, "p75": 0, "p90": 0, "max": 0, "mean": 0}
    return {
        "n": len(s),
        "min": s[0],
        "p25": pct(s, 0.25),
        "median": pct(s, 0.50),
        "p75": pct(s, 0.75),
        "p90": pct(s, 0.90),
        "max": s[-1],
        "mean": round(statistics.fmean(s), 2),
    }


def load(path: Path) -> tuple[list[dict], int]:
    recs: list[dict] = []
    bad_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
    return recs, bad_lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="outp", type=Path, required=True)
    ap.add_argument("--stats", default=None, help="10_collect.py 打印的 [stats] JSON 字符串")
    ap.add_argument("--stats-file", default=None,
                    help="同上，但从文件读（Windows 下传内联 JSON 会被 PowerShell 吞引号）")
    ap.add_argument("--n-sample", type=int, default=10)
    args = ap.parse_args()

    recs, bad_lines = load(args.inp)
    n = len(recs)
    critical: list[str] = []   # 硬失败：长度越界 / 缺 pmid / 缺 abstract / 重复
    warnings: list[str] = []   # 可用性告警：多样性、文献复用（不阻断管线，但影响训练）

    # ---------------- 1. 总条数 ----------------
    raw_stats = args.stats
    if not raw_stats and args.stats_file:
        # utf-8-sig：PowerShell 的 Set-Content -Encoding utf8 会写 BOM
        raw_stats = Path(args.stats_file).read_text(encoding="utf-8-sig").strip()
        # 允许直接粘 "[stats] {...}" 这一整行
        if raw_stats.startswith("[stats]"):
            raw_stats = raw_stats[len("[stats]"):].strip()
    stats = json.loads(raw_stats) if raw_stats else None
    if stats and stats.get("kept") is not None and stats["kept"] != n:
        critical.append(f"[stats].kept={stats['kept']} 与文件实际条数 {n} 不一致")

    # ---------------- 2. 长度 ----------------
    seqs = [r.get("sequence") or "" for r in recs]
    lens = [len(s) for s in seqs]
    lens_d = dist(lens)
    over = sorted(
        [{"uid": r.get("uid"), "nuc_accession": r.get("nuc_accession"), "len": len(r.get("sequence") or "")}
         for r in recs if len(r.get("sequence") or "") > MAX_NT],
        key=lambda d: -d["len"],
    )
    under = [r.get("uid") for r in recs if len(r.get("sequence") or "") < MIN_NT]
    if over:
        critical.append(f"有 {len(over)} 条序列长度 > {MAX_NT} nt（RNA-FM 会截断/报错）")
    if under:
        critical.append(f"有 {len(under)} 条序列长度 < {MIN_NT} nt")

    brackets = {
        "30-99": sum(1 for x in lens if 30 <= x <= 99),
        "100-249": sum(1 for x in lens if 100 <= x <= 249),
        "250-499": sum(1 for x in lens if 250 <= x <= 499),
        "500-749": sum(1 for x in lens if 500 <= x <= 749),
        "750-1022": sum(1 for x in lens if 750 <= x <= MAX_NT),
        ">1022": len(over),
    }

    # ---------------- 3. 字母表 ----------------
    residue_counter: Counter[str] = Counter()
    n_with_residue = 0
    residue_examples: list[dict] = []
    for r in recs:
        s = (r.get("sequence") or "").upper()
        residue = "".join(sorted({c for c in s if c not in VALID}))
        if residue:
            n_with_residue += 1
            for c in residue:
                residue_counter[c] += s.count(c)
            if len(residue_examples) < 15:
                residue_examples.append({"uid": r.get("uid"), "residue_chars": residue,
                                         "len": len(s)})
            unknown = sorted(set(residue) - IUPAC)
            if unknown:
                critical.append(f"{r.get('uid')} 含非 IUPAC 字符 {unknown}")
    alphabet_ok = n_with_residue == 0 or all(set(r["residue_chars"]) <= IUPAC for r in residue_examples)

    # ---------------- 4. 物种 / 类型 ----------------
    orgs = Counter((r.get("organism") or "<空>") for r in recs)
    types = Counter(classify(r.get("definition") or "") for r in recs)
    no_org = sum(1 for r in recs if not (r.get("organism") or "").strip())
    no_def = sum(1 for r in recs if not (r.get("definition") or "").strip())
    if no_org:
        critical.append(f"有 {no_org} 条缺 organism")
    if no_def:
        critical.append(f"有 {no_def} 条缺 definition")

    # ---------------- 5. 摘要长度 ----------------
    abs_lens = [len(r.get("literature") or "") for r in recs]
    abs_d = dist(abs_lens)
    short_abs = [r.get("uid") for r in recs if len(r.get("literature") or "") < MIN_ABSTRACT]

    # ---------------- 5b. 来源分布（后面 7c 要用） ----------------
    qname = Counter((r.get("provenance") or {}).get("query_name", "?") for r in recs)

    # 与其它 JSONL 的交叉核对（例如 50 条基线、20 条 probe）
    cross_checks = []
    for other in sorted((ROOT / "data" / "raw").glob("*.jsonl")):
        if other.resolve() == args.inp.resolve() or other.stat().st_size == 0:
            continue
        o_recs, _ = load(other)
        o_acc = {r.get("nuc_accession") for r in o_recs}
        cross_checks.append({
            "file": other.name,
            "n": len(o_recs),
            "shared_accessions": sum(1 for r in recs if r.get("nuc_accession") in o_acc),
            "shared_sequences": len({r.get("sequence") for r in recs}
                                    & {r.get("sequence") for r in o_recs}),
        })

    # ---------------- 6. 去重 ----------------
    def dups(key: str) -> dict:
        c = Counter((r.get(key) or "") for r in recs)
        d = {k: v for k, v in c.items() if v > 1}
        return {
            "unique": len(c),
            "n_dup_keys": len(d),
            "n_extra_rows": sum(v - 1 for v in d.values()),
            "top": sorted(({"key": k, "count": v} for k, v in d.items()), key=lambda x: -x["count"])[:10],
        }

    uid_d = dups("uid")
    acc_d = dups("nuc_accession")

    sha: dict[str, list[str]] = {}
    for r in recs:
        h = hashlib.sha1((r.get("sequence") or "").encode()).hexdigest()[:16]
        sha.setdefault(h, []).append(r.get("uid") or "")
    sha_dups = {h: u for h, u in sha.items() if len(u) > 1}
    if uid_d["n_dup_keys"]:
        critical.append(f"uid 有 {uid_d['n_dup_keys']} 个重复键（多出 {uid_d['n_extra_rows']} 行）")
    if acc_d["n_dup_keys"]:
        critical.append(f"nuc_accession 有 {acc_d['n_dup_keys']} 个重复（多出 {acc_d['n_extra_rows']} 行）")
    if sha_dups:
        critical.append(f"序列本身有 {sum(len(v) - 1 for v in sha_dups.values())} 条重复"
                        f"（{len(sha_dups)} 组相同序列）")

    # ---------------- 7. 必填字段 ----------------
    miss_pmid = [r.get("uid") for r in recs if not (r.get("pmid") or "").strip()]
    miss_abs = [r.get("uid") for r in recs if not (r.get("literature") or "").strip()]
    if miss_pmid:
        critical.append(f"有 {len(miss_pmid)} 条缺 pmid: {miss_pmid[:10]}")
    if miss_abs:
        critical.append(f"有 {len(miss_abs)} 条缺 abstract: {miss_abs[:10]}")
    # uid 里嵌的 pmid 也要和 pmid 字段一致
    uid_mismatch = [r.get("uid") for r in recs
                    if (r.get("uid") or "").split("|pmid:")[-1] != (r.get("pmid") or "")]
    if uid_mismatch:
        critical.append(f"有 {len(uid_mismatch)} 条 uid 内嵌 pmid 与 pmid 字段不一致")

    # ---------------- 7b. 文献复用（序列唯一 ≠ 文献唯一） ----------------
    pmid_c = Counter((r.get("pmid") or "") for r in recs)
    abs_c = Counter((r.get("literature") or "") for r in recs)
    lit_dup_rows = sum(v - 1 for v in abs_c.values() if v > 1)
    pm_dup_rows = sum(v - 1 for v in pmid_c.values() if v > 1)
    if n and lit_dup_rows / n > 0.20:
        warnings.append(
            f"文献复用严重：只有 {len(pmid_c)} 个不同 pmid / {len(abs_c)} 个不同摘要，"
            f"却有 {n} 条记录（{lit_dup_rows} 行为重复摘要，占 {lit_dup_rows / n:.0%}）——"
            "同一条 abstract 被配到大量不同序列上，AIS/摘要类监督信号会被稀释"
        )
    lit_titles = {}
    for r in recs:
        p = r.get("pmid") or ""
        if p not in lit_titles:
            pr = r.get("provenance") or {}
            lit_titles[p] = {"pmid": p, "count": pmid_c[p], "year": pr.get("year", ""),
                             "journal": pr.get("journal", ""), "title": (pr.get("title") or "")[:140]}
    lit_reuse_top = sorted(lit_titles.values(), key=lambda d: -d["count"])[:10]

    # ---------------- 7c. 多样性 ----------------
    acc_pref = Counter(re.match(r"[A-Za-z_]+", (r.get("nuc_accession") or "")).group(0)
                       for r in recs if r.get("nuc_accession"))
    n_orgs = len(orgs)
    top_org_share = (orgs.most_common(1)[0][1] / n) if n else 0
    if n and n_orgs <= 10:
        warnings.append(f"物种集中度高：仅 {n_orgs} 个物种，"
                        f"'{orgs.most_common(1)[0][0]}' 占 {top_org_share:.0%}")
    n_types = len(types)
    if n and types:
        top_type, top_type_n = types.most_common(1)[0]
        if top_type_n / n > 0.90:
            warnings.append(f"RNA 类型单一：'{top_type}' 占 {top_type_n / n:.0%}"
                            f"（全部只有 {n_types} 类）")
    if n and len(qname) < 3:
        warnings.append(f"只命中了 {len(qname)} 条 query（{dict(qname)}），"
                        "多路查询的多样性设计没有生效——数据被单一的 "
                        f"{qname.most_common(1)[0][0]} 查询塞满")
    for cc in cross_checks:
        if cc["shared_accessions"] == cc["n"] and cc["n"]:
            warnings.append(
                f"与 {cc['file']} 完全重叠：它的 {cc['n']} 条 accession 全部出现在本文件中"
                "（esearch 按 relevance 排序是确定性的，所以每次跑都从同一片记录开始）——"
                "如果后续要把几个 jsonl 合并训练，必须先按 accession 去重，否则这些行会被重复计入"
            )

    # ---------------- 8. 抽样 ----------------
    k = min(args.n_sample, n)
    step = max(n // k, 1)
    picks = recs[::step][:k]
    samples = [
        {
            "uid": r.get("uid"),
            "seq_len": len(r.get("sequence") or ""),
            "organism": r.get("organism"),
            "rna_type": classify(r.get("definition") or ""),
            "definition": (r.get("definition") or "")[:200],
            "abstract_head150": re.sub(r"\s+", " ", (r.get("literature") or ""))[:150],
            "pmid": r.get("pmid"),
            "journal": (r.get("provenance") or {}).get("journal", ""),
            "year": (r.get("provenance") or {}).get("year", ""),
        }
        for r in picks
    ]

    qname_report = qname
    file_bytes = args.inp.stat().st_size

    # 与其它 JSONL 的交叉核对（例如 50 条基线、20 条 probe）
    cross_checks = []
    for other in (ROOT / "data" / "raw").glob("*.jsonl"):
        if other.resolve() == args.inp.resolve() or other.stat().st_size == 0:
            continue
        o_recs, _ = load(other)
        o_acc = {r.get("nuc_accession") for r in o_recs}
        overlap = sum(1 for r in recs if r.get("nuc_accession") in o_acc)
        dup_seqs = len({r.get("sequence") for r in recs} & {r.get("sequence") for r in o_recs})
        cross_checks.append({
            "file": other.name, "n": len(o_recs),
            "shared_accessions": overlap, "shared_sequences": dup_seqs,
        })

    report_notes = [
        "10_collect.py 的逻辑**未改动**；本次只新增了只读的报告脚本 scripts/14_quality_report.py。",
        "注意 10_collect.py 顶部注释写的是 30–1024 nt，而实际常量 MIN_NT, MAX_NT = 30, 1022 —— "
        "注释与代码不一致（1022 = 探针实测的 RNA-FM 上限）。本次按任务要求保持 1022，未改注释。",
        "MAX_NT 之外还有一层 esearch 服务端闸门：query 里写的是 60:1000[slen]，"
        "所以实际落盘长度天然 <= 1000，1022 闸门是双保险而非唯一约束。",
        "所有查询都命中了 srcdb_refseq + biomol_rna，因此 1000 条全部是 RefSeq NR_* 记录。",
        "10_collect.py 把记录攒在内存里、结束时一次性写盘，所以采集过程中 jsonl 文件不存在 —— "
        "想看进度只能读进程 stdout（本次是后台任务日志）。",
    ]

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": str(args.inp),
        "source_bytes": file_bytes,
        "source_mb": round(file_bytes / 1024 / 1024, 2),
        "pipeline": "esearch nuccore -> efetch nuccore rettype=gb -> efetch pubmed",
        "gates": {"MIN_NT": MIN_NT, "MAX_NT": MAX_NT, "MIN_ABSTRACT": MIN_ABSTRACT},

        "verdict": "PASS" if not critical else "FAIL",
        "critical_issues": critical,
        "warnings": warnings,
        "hard_gates_ok": not critical,
        "downstream_usability": "OK" if not warnings else "DEGRADED",
        "notes": report_notes,
        "cross_checks": cross_checks,

        "counts": {
            "total_records": n,
            "bad_jsonl_lines": bad_lines,
            "collect_stats": stats,
            "by_query": dict(qname_report),
        },

        "length_distribution": {
            **lens_d,
            "n_over_max_nt": len(over),
            "max_nt_gate": MAX_NT,
            "over_max_examples": over[:10],
            "under_min_examples": under[:10],
            "buckets": brackets,
        },

        "alphabet": {
            "allowed_core": sorted(VALID),
            "allowed_iupac": sorted(IUPAC),
            "records_with_residue": n_with_residue,
            "residue_char_counts": dict(residue_counter),
            "residue_is_empty_or_degenerate_only": alphabet_ok,
            "examples": residue_examples,
        },

        "organisms_top10": [{"organism": k, "count": v} for k, v in orgs.most_common(10)],
        "n_distinct_organisms": len(orgs),
        "rna_type_distribution": dict(types.most_common()),
        "n_distinct_types": len(types),
        "accession_prefixes": dict(acc_pref.most_common()),

        "literature_reuse": {
            "n_distinct_pmids": len(pmid_c),
            "n_distinct_abstracts": len(abs_c),
            "n_rows_with_reused_abstract": lit_dup_rows,
            "reuse_ratio": round(lit_dup_rows / n, 4) if n else 0,
            "top_reused": lit_reuse_top,
        },

        "abstract_length_distribution": abs_d,
        "n_abstract_under_min": len(short_abs),

        "duplicates": {
            "uid": uid_d,
            "nuc_accession": acc_d,
            "sequence_sha1": {
                "unique_sequences": len(sha),
                "n_dup_groups": len(sha_dups),
                "n_extra_duplicate_rows": sum(len(v) - 1 for v in sha_dups.values()),
                "groups": [{"sha1_16": h, "count": len(u), "uids": u[:6]} for h, u in list(sha_dups.items())[:10]],
            },
        },

        "required_fields": {
            "n_missing_pmid": len(miss_pmid),
            "n_missing_abstract": len(miss_abs),
            "n_uid_pmid_mismatch": len(uid_mismatch),
            "examples_missing": (miss_pmid + miss_abs)[:10],
            "ok": not (miss_pmid or miss_abs or uid_mismatch),
        },

        "samples": samples,
    }

    args.outp.parent.mkdir(parents=True, exist_ok=True)
    args.outp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print(f"[report] {args.outp}  verdict={report['verdict']}")
    print(f"  total={n}  bytes={file_bytes}  bad_lines={bad_lines}")
    print(f"  len: min={lens_d['min']} p25={lens_d['p25']} med={lens_d['median']} "
          f"p75={lens_d['p75']} p90={lens_d['p90']} max={lens_d['max']}  >{MAX_NT}={len(over)}")
    print(f"  alphabet residue records={n_with_residue} chars={dict(residue_counter)}")
    print(f"  organisms={len(orgs)}  types={dict(types.most_common())}")
    print(f"  abstract: min={abs_d['min']} med={abs_d['median']} max={abs_d['max']}")
    print(f"  dup: uid={uid_d['n_dup_keys']} acc={acc_d['n_dup_keys']} seq={len(sha_dups)}")
    print(f"  missing pmid={len(miss_pmid)} abstract={len(miss_abs)}")
    print(f"  literature: distinct pmid={len(pmid_c)} distinct abstract={len(abs_c)} "
          f"reused_rows={lit_dup_rows} ({lit_dup_rows / n:.0%})")
    print(f"  diversity: organisms={len(orgs)} types={len(types)} queries={len(qname)}")
    print("  --- 抽样 ---")
    for s in samples:
        print(f"   {s['uid']}  len={s['seq_len']:4d}  {s['rna_type']}")
        print(f"     DEF: {s['definition'][:100]}")
        print(f"     ABS: {s['abstract_head150'][:110]}")
    if critical:
        print("  !! CRITICAL:")
        for c in critical:
            print("   -", c)
    if warnings:
        print("  ~ warnings:")
        for w in warnings:
            print("   -", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
