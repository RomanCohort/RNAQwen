"""rna_literature_1k 能不能并进训练集？以及被丢掉的那些能不能捞回来。

三个问题：
  1. 这 1000 条有没有 annotation / qa，够不够格进 Stage 1 / Stage 2
  2. 和现有 3488 条**重不重复**（按 uid / URS / 序列）
  3. 质量：pmid 复用率（PROGRESS 记过「1000 条里只有 24 个不同 pmid」那个坑）
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")


def load(p):
    out = []
    with open(ROOT / p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


base = load("data/processed/rnaqa_rnacentral_all.jsonl")
print("=" * 90)
print("现有训练集 rnaqa_rnacentral_all: %d 条" % len(base))
print("=" * 90)

def keys_of(rec):
    return {
        "uid": rec.get("uid"),
        "urs": rec.get("urs_id") or rec.get("uid"),
        "seq": (rec.get("sequence") or "")[:60],
        "acc": rec.get("nuc_accession"),
    }


base_uids = {r.get("uid") for r in base}
base_accs = {r.get("nuc_accession") for r in base}
base_seqs = {(r.get("sequence") or "")[:60] for r in base}
print("  现有 uid 数 %d，accession 数 %d" % (len(base_uids), len(base_accs)))

for cand in ["data/raw/rna_literature_1k.jsonl", "data/raw/rna_literature_50.jsonl"]:
    recs = load(cand)
    print()
    print("=" * 90)
    print("%s : %d 条" % (cand, len(recs)))
    print("=" * 90)

    has_ann = sum(1 for r in recs if (r.get("annotation") or "").strip())
    has_lit = sum(1 for r in recs if (r.get("literature") or "").strip())
    has_qa = sum(1 for r in recs if r.get("qa"))
    nqa = sum(len(r.get("qa") or []) for r in recs)
    print("  有 annotation : %d" % has_ann)
    print("  有 literature : %d" % has_lit)
    print("  有 qa         : %d  （共 %d 组 QA）" % (has_qa, nqa))

    prov = Counter(
        json.dumps(r.get("provenance"), sort_keys=True, ensure_ascii=False)
        if isinstance(r.get("provenance"), (dict, list))
        else str(r.get("provenance"))
        for r in recs
    )
    print("  provenance    :")
    for k, v in prov.most_common(5):
        print("       %-70s %d" % (k[:70], v))

    lens = sorted(len(r.get("sequence") or "") for r in recs)
    if lens:
        print("  序列长度      : min %d  median %d  max %d" % (lens[0], lens[len(lens)//2], lens[-1]))

    pmids = [r.get("pmid") for r in recs if r.get("pmid")]
    print("  pmid 复用率   : %d 条记录 / %d 个不同 pmid  (%.2f 条/pmid)"
          % (len(pmids), len(set(pmids)), len(pmids) / max(len(set(pmids)), 1)))

    ov_u = sum(1 for r in recs if r.get("uid") in base_uids)
    ov_a = sum(1 for r in recs if r.get("nuc_accession") in base_accs)
    ov_s = sum(1 for r in recs if (r.get("sequence") or "")[:60] in base_seqs)
    print("  与现有集重叠  : uid %d / accession %d / 序列前60 %d" % (ov_u, ov_a, ov_s))
    print("  -> 可新增约   : %d 条" % (len(recs) - ov_s))

print()
print("=" * 90)
print("被丢掉的记录（看能不能捞回来）")
print("=" * 90)
for p in ["data/processed/rnaqa_n6.dropped.jsonl",
          "data/processed/rnaqa_rnacentral.dropped.jsonl",
          "data/processed/rnaqa_rnacentral_new.dropped.jsonl"]:
    recs = load(p)
    reasons = Counter(r.get("reason") or r.get("error") or "?" for r in recs)
    print("  %-48s %5d 条" % (p.split("/")[-1], len(recs)))
    for k, v in reasons.most_common(6):
        print("       %-56s %d" % (str(k)[:56], v))
