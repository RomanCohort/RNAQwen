#!/bin/bash
# Rfam 全量搜索 —— 分片并行。单进程 --cpu 吃不满核心，切成 N 片同时跑更划算。
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam
mkdir -p chunks out

NCHUNK=${1:-8}
NCORE=$(nproc)
echo "核心数 $NCORE   分片 $NCHUNK"

if [ ! -f all.fa ]; then echo "缺 all.fa"; exit 1; fi

# 只在需要时切片
if [ ! -f chunks/c00.fa ]; then
  echo "切片中..."
  awk -v n="$NCHUNK" -v d="chunks" '
    /^>/ { i = (++k % n) }
    { print > (d "/c" sprintf("%02d", i) ".fa") }
  ' all.fa
fi
ls chunks/*.fa | wc -l
wc -l chunks/*.fa | tail -1

echo
echo "=== 并行启动（每片 --cpu $((NCORE/NCHUNK+1))）==="
for f in chunks/*.fa; do
  b=$(basename "$f" .fa)
  if [ -s "out/$b.tbl" ]; then echo "  跳过 $b"; continue; fi
  cmscan --rfam --noali --cpu $((NCORE/NCHUNK+1)) \
    --tblout "out/$b.tbl" \
    -o "out/$b.out" Rfam.cm "$f" &
  echo "  启动 $b (PID $!)"
done
wait

echo
echo "=== 汇总 ==="
cat out/*.tbl | grep -v '^#' > hits.tbl
echo "总命中行 $(wc -l < hits.tbl)"
echo "有命中的序列 $(awk '{print $3}' hits.tbl | sort -u | wc -l)"
echo "命中的家族 $(awk '{print $2}' hits.tbl | sort -u | wc -l)"
echo
echo "=== 前 15 条 ==="
head -15 hits.tbl
