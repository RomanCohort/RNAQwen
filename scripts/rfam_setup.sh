#!/bin/bash
# Rfam 家族归属 —— 在 WSL 里跑
set -e
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "=== 工具 ==="
cmsearch -h 2>&1 | head -2

echo
echo "=== Rfam.cm 状态 ==="
ls -l Rfam.cm 2>/dev/null || echo "Rfam.cm 不存在，需要解压"
if [ ! -f Rfam.cm ]; then
  echo "解压中..."
  gunzip -c Rfam.cm.gz > Rfam.cm
fi
ls -l Rfam.cm
echo "模型个数: $(grep -c '^NAME' Rfam.cm)"

echo
echo "=== cmpress 建索引 ==="
if [ ! -f Rfam.cm.i1f ]; then
  cmpress Rfam.cm
fi
ls -l Rfam.cm.i1* 2>/dev/null | head -5

echo
echo "=== 完成 ==="
