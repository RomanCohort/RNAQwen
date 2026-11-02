"""准备扩数据的输入：从 rna_literature_1k 里切出真正新增的那些，并补上 rna_type。

三个动作：
  1. 按序列前 60 字符去掉与现有 3488 条重叠的（实测只有 7 条）
  2. 补 rna_type —— 这批是空的，但 provenance.molecule_type 有值
  3. 写出去，给 11_generate_annotations.py 用

不做的话：rna_type 空着会让下游丢类型条件；重叠的会污染训练/评测划分。
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
SRC = ROOT / "data/raw/rna_literature_1k.jsonl"
BASE = ROOT / "data/processed/rnaqa_rnacentral_all.jsonl"
OUT = ROOT / "data/raw/rna_literature_1k_new.jsonl"

base = [json.loads(l) for l in open(BASE, encoding="utf-8") if l.strip()]
base_seqs = {(r.get("sequence") or "")[:60] for r in base}

src = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
print("源 %d 条，现有集 %d 条" % (len(src), len(base)))

kept, dropped = [], []
for r in src:
    if (r.get("sequence") or "")[:60] in base_seqs:
        dropped.append(r)
        continue
    # 补 rna_type
    prov = r.get("provenance") or {}
    mt = (prov.get("molecule_type") or "").strip() if isinstance(prov, dict) else ""
    if not (r.get("rna_type") or "").strip():
        r["rna_type"] = mt
    kept.append(r)

print("去掉重叠 %d 条，保留 %d 条" % (len(dropped), len(kept)))

mt_counts = Counter(
    ((r.get("provenance") or {}).get("molecule_type") or "(空)") if isinstance(r.get("provenance"), dict) else "(?)"
    for r in kept
)
print("molecule_type 分布:", dict(mt_counts))
n_type = sum(1 for r in kept if (r.get("rna_type") or "").strip())
print("补上 rna_type 的: %d / %d" % (n_type, len(kept)))

lens = sorted(len(r.get("sequence") or "") for r in kept)
print("序列长度 min %d median %d max %d" % (lens[0], lens[len(lens)//2], lens[-1]))

missing_def = sum(1 for r in kept if not (r.get("definition") or "").strip())
missing_lit = sum(1 for r in kept if not (r.get("literature") or "").strip())
print("缺 definition: %d   缺 literature: %d" % (missing_def, missing_lit))

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\n[ok] -> %s  (%d 条)" % (OUT, len(kept)))
