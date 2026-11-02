"""第 1、2 步（容错版）：id_mapping 已确认损坏，读到哪算哪；parquet join 独立跑。

id_mapping.tsv.gz 在 2000 万行后 zlib 报 invalid block type —— 大小对、内容坏，
和 RNAFM 那次是同一个失败模式。所以：
  · 第 1 步：读到损坏处为止，统计仍然是**真实的前缀数据**（2000 万行不是小样本）
  · 第 2 步：不依赖 id_mapping，独立跑
"""
from __future__ import annotations

import gzip
import json
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDMAP = ROOT / "data/external/id_mapping.tsv.gz"
PARQ = ROOT / "data/external/rnacentral_1024.parquet"
FULL = ROOT / "data/processed/rnaqa_seq30_full.jsonl"

TARGET = {"RFAM", "GENECARDS", "MALACARDS", "LNCIPEDIA", "LNCBOOK", "NONCODE",
          "MIRBASE", "MIRGENEDB", "PDB", "REFSEQ", "ENSEMBL", "ENSEMBL_METAZOA"}
t0 = time.time()

print("=" * 74)
print("第 1 步  扫 id_mapping.tsv.gz（容错）")
print("=" * 74)
db_lines = Counter()
db_urs = defaultdict(set)
n = 0
truncated = False
with gzip.open(IDMAP, "rt", encoding="utf-8", errors="replace") as f:
    try:
        for line in f:
            n += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            db_lines[p[1]] += 1
            if p[1] in TARGET:
                db_urs[p[1]].add(p[0])
            if n % 20_000_000 == 0:
                print(f"    {n:,} 行  {time.time()-t0:.0f}s", flush=True)
    except (zlib.error, EOFError, OSError) as e:
        truncated = True
        print(f"    **在 {n:,} 行处损坏**: {type(e).__name__}: {str(e)[:80]}")

print(f"\n读到 {n:,} 行{'（已损坏，这是能读到的全部）' if truncated else ''}"
      f"   用时 {time.time()-t0:.0f}s")
print(f"数据库种类 {len(db_lines)}")
print()
print(f"{'数据库':<20} {'行数':>13} {'覆盖 URS':>11}  生物学")
for db, c in db_lines.most_common(22):
    u = len(db_urs.get(db, ()))
    print(f"  {db:<18} {c:>13,} {u:>11,}   {'是' if db in TARGET else ''}")

print()
print("=" * 74)
print("第 2 步  我们的序列 -> parquet 按序列 join 拿 URS（不依赖 id_mapping）")
print("=" * 74)
our = {}
with open(FULL, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        s = r["sequence"]
        if s not in our:
            our[s] = r["uid"]
print(f"我们的唯一序列 {len(our):,}")

import pyarrow.parquet as pq
pf = pq.ParquetFile(PARQ)
print(f"parquet {pf.metadata.num_rows:,} 行 / {pf.metadata.num_row_groups} 组")

seq2urs = {}
t1 = time.time()
for i in range(pf.metadata.num_row_groups):
    d = pf.read_row_group(i, columns=["upi", "sequence"]).to_pydict()
    for upi, s in zip(d["upi"], d["sequence"]):
        if s in our and s not in seq2urs:
            seq2urs[s] = upi
    print(f"    rg {i+1}/{pf.metadata.num_row_groups}  命中 {len(seq2urs):,}  "
          f"{time.time()-t1:.0f}s", flush=True)

rate = len(seq2urs) / len(our)
print(f"\n**命中 {len(seq2urs):,} / {len(our):,} = {100*rate:.1f}%**   用时 {time.time()-t1:.0f}s")

print()
print("=" * 74)
print("第 3 步  能拿到生物学标注的条数")
print("=" * 74)
hit = set(seq2urs.values())
print(f"有 URS 的序列 {len(hit):,}")
tot = set()
print(f"\n{'数据库':<20} {'命中':>10}")
for db in sorted(TARGET):
    s = db_urs.get(db)
    if not s:
        continue
    inter = hit & s
    tot |= inter
    print(f"  {db:<18} {len(inter):>10,}")
print()
print(f"  **至少一个生物学标注: {len(tot):,} = {100*len(tot)/len(our):.2f}%**")

(ROOT / "reports/idmap_coverage.json").write_text(json.dumps(dict(
    idmap_lines_read=n, idmap_truncated=truncated,
    db_lines=dict(db_lines.most_common(40)),
    db_urs={k: len(v) for k, v in db_urs.items()},
    our_seqs=len(our), matched_urs=len(seq2urs), match_rate=round(rate, 4),
    bio_covered=len(tot),
    bio_by_db={db: len(hit & db_urs[db]) for db in TARGET if db_urs.get(db)},
), ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[save] reports/idmap_coverage.json   总用时 {(time.time()-t0)/60:.1f} min")
