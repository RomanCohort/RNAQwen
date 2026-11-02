"""把数据源换成**真正的 RNAcentral**（HF 上有镜像），一次解决"RNA 不够多"。

为什么换
--------
之前走的是 NCBI 代理链路（nuccore → GenBank → pubmed），实测两个硬伤：
  1. 文献复用 84%~98% —— 一条 RefSeq 记录挂的 PMID 大量集中在少数几篇"大论文"上；
  2. 产出极慢 —— 加相关性闸门后，扫 100 条只留 5~20 条。

而 RNA-GPT 原版的源就是 **RNAcentral**。本机到 EBI 不通（rnacentral.org / ftp.ebi.ac.uk 全 SSL EOF），
但 **HuggingFace 上有 RNAcentral 官方与社区镜像**，实测可达：

    multimolecule/rnacentral.1024   3.7 GB   upi / sequence / type / description      （10M~100M 条）
    RNAcentral/litsumm-v1.5         8.2 MB   urs_taxid / ent_id / context / summary   （4,618 条）

`litsumm` 是 RNAcentral 官方出的**文献摘要集**：每条 RNA 给出
`context`（文献原文片段，带 PMC 号）+ `summary`（LLM 合成的文献综述）。
这正好就是原版 D&C 注释那一层 —— 而且带 `truthful` / `problem_summary` 质量标记。

于是管线变成：

    litsumm(urs_taxid, summary, context)
        ⋈ rnacentral.1024(upi, sequence, type, description)     按 URS id 连接
        → (sequence, type, description, summary, context)
        → 本地 Qwen 拆 QA → RNA-QA

**4,618 条待配对 RNA ≈ PLAN §0 说的原版"5K RNA"训练子集**，量级对上了。

跑法：
    python scripts/16_build_from_rnacentral.py \
        --litsumm data/external/litsumm_v1_5_data.parquet \
        --rnacentral data/external/rnacentral_1024.parquet \
        --out data/raw/rnacentral_litsumm_joined.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.records import RnaQaRecord, write_jsonl  # noqa: E402

MAX_NT = 1022   # RNA-FM 实测上限（不是 1024）
MIN_NT = 30
MIN_SUMMARY = 80


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--litsumm", type=Path, required=True)
    ap.add_argument("--rnacentral", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-nt", type=int, default=MIN_NT)
    ap.add_argument("--max-nt", type=int, default=MAX_NT)
    ap.add_argument("--require-truthful", action="store_true", default=True,
                    help="只要 litsumm 里 truthful=True 且 problem_summary=False 的（默认开）")
    ap.add_argument("--no-require-truthful", dest="require_truthful", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    t0 = time.time()
    lit = pq.read_table(args.litsumm, columns=[
        "urs_taxid", "ent_id", "context", "summary", "truthful", "problem_summary", "selection_method",
    ]).to_pydict()
    n_lit = len(lit["urs_taxid"])
    print(f"[join] litsumm 行数 {n_lit}")

    rows = []
    stats = {"no_ursid": 0, "untruthful": 0, "short_summary": 0}
    for i in range(n_lit):
        urs_taxid = (lit["urs_taxid"][i] or "").strip()
        if not urs_taxid or "_" not in urs_taxid:
            stats["no_ursid"] += 1
            continue
        if args.require_truthful and (lit["truthful"][i] is False or lit["problem_summary"][i] is True):
            stats["untruthful"] += 1
            continue
        summary = (lit["summary"][i] or "").strip()
        if len(summary) < MIN_SUMMARY:
            stats["short_summary"] += 1
            continue
        rows.append({
            "urs": urs_taxid.split("_")[0],
            "urs_taxid": urs_taxid,
            "taxid": urs_taxid.split("_", 1)[1] if "_" in urs_taxid else "",
            "ent_id": lit["ent_id"][i] or "",
            "context": lit["context"][i] or "",
            "summary": summary,
            "selection_method": lit["selection_method"][i] or "",
        })
    print(f"[join] 通过质量闸门 {len(rows)} 条；丢弃 {stats}")

    urs_ids = sorted({r["urs"] for r in rows})
    print(f"[join] 需要去 rnacentral.1024 里查 {len(urs_ids)} 个 URS id")

    t1 = time.time()
    table = ds.dataset(args.rnacentral, format="parquet").to_table(
        filter=pc.field("upi").isin(pa.array(urs_ids))
    )
    print(f"[join] rnacentral 扫描完成，命中 {table.num_rows} 行（{time.time()-t1:.1f}s）")
    seq_by_urs = {}
    cols = table.to_pydict()
    for upi, seq, typ, desc in zip(cols["upi"], cols["sequence"], cols["type"], cols["description"]):
        seq_by_urs[upi] = (seq, typ, desc)
    print(f"[join] 唯一 URS {len(seq_by_urs)} 个")

    records, stats2 = [], {"no_sequence": 0, "bad_len": 0, "bad_alpha": 0, "duplicate_seq": 0}
    seen_seq: set[str] = set()
    for r in rows:
        hit = seq_by_urs.get(r["urs"])
        if hit is None:
            stats2["no_sequence"] += 1
            continue
        seq, typ, desc = hit
        s = (seq or "").upper().replace("T", "U")
        if not (args.min_nt <= len(s) <= args.max_nt):
            if len(s) > args.max_nt:
                s = s[: args.max_nt]          # 截到 RNA-FM 上限；记录了这件事
                stats2.setdefault("truncated", 0)
                stats2["truncated"] = stats2.get("truncated", 0) + 1
            else:
                stats2["bad_len"] += 1
                continue
        core = s.replace("N", "")
        if not core or set(core) - set("ACGU"):
            stats2["bad_alpha"] += 1
            continue
        if s in seen_seq:
            stats2["duplicate_seq"] += 1
            continue
        seen_seq.add(s)

        records.append(RnaQaRecord(
            uid=f"urs:{r['urs']}|taxid:{r['taxid']}",
            sequence=s,
            organism=f"taxid:{r['taxid']}" if r["taxid"] else "",
            rna_type=typ or "",
            definition=f"{r['ent_id']} — {desc}" if desc else r["ent_id"],
            literature=r["context"],          # 文献原文片段
            annotation=r["summary"],          # RNAcentral 官方的文献综述 = D&C 那一层
            provenance={
                "source": "RNAcentral (HF mirror)",
                "litsumm": "RNAcentral/litsumm-v1.5",
                "sequences": "multimolecule/rnacentral.1024",
                "ent_id": r["ent_id"],
                "selection_method": r["selection_method"],
                "joined": time.strftime("%Y-%m-%d"),
            },
        ))
        if args.limit and len(records) >= args.limit:
            break

    print(f"[join] 成对成功 {len(records)} 条；丢弃 {stats2}")
    n = write_jsonl(args.out, records)
    print(f"\n[ok] 写入 {n} 条 -> {args.out}  总用时 {(time.time()-t0)/60:.1f} min")
    if records:
        lens = sorted(len(r.sequence) for r in records)
        types: dict[str, int] = {}
        for r in records:
            types[r.rna_type] = types.get(r.rna_type, 0) + 1
        print(f"[lens] min={lens[0]} median={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
        print(f"[types] {sorted(types.items(), key=lambda kv: -kv[1])[:8]}")
        r = records[0]
        print("\n[sample]")
        print("  uid       :", r.uid)
        print("  type      :", r.rna_type, "|", len(r.sequence), "nt")
        print("  definition:", r.definition[:110])
        print("  summary   :", r.annotation[:260])
        print("  context   :", r.literature[:160], "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
