#!/bin/bash
# 导出 FASTA + 测 cmsearch 速度
set -e
cd /mnt/c/qwen-api/RNAQwen

echo "=== 导出唯一序列到 FASTA ==="
python3 - <<'PY'
import json
seen = {}
with open('data/processed/rnaqa_seq30_full.jsonl', encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)
        u = r['uid']
        if u not in seen:
            seen[u] = r['sequence']
        if len(seen) >= 145201:
            break
with open('data/external/rfam/all.fa', 'w') as f:
    for u, s in seen.items():
        f.write(f'>{u}\n{s}\n')
print(f'写到 {len(seen):,} 条')
PY
ls -l data/external/rfam/all.fa | awk '{print $5/1048576 " MB"}'
grep -c '^>' data/external/rfam/all.fa

echo
echo "=== 取前 200 条测速度 ==="
awk '/^>/{n++} n<=200' data/external/rfam/all.fa > data/external/rfam/test200.fa
grep -c '^>' data/external/rfam/test200.fa

echo
echo "=== cmsearch --rfam 计时 ==="
time cmsearch --rfam --noali --cpu 8 \
  -o data/external/rfam/test200.out \
  data/external/rfam/Rfam.cm data/external/rfam/test200.fa 2>&1 | tail -4

echo
echo "=== 命中了多少 ==="
grep -c '^RF' data/external/rfam/test200.out 2>/dev/null || echo 0
tail -3 data/external/rfam/test200.out 2>/dev/null || true
