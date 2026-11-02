#!/bin/bash
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "=== test200.out 到底是什么 ==="
ls -l test200.out
echo "--- 头 20 行 ---"
head -20 test200.out
echo "--- 尾 20 行 ---"
tail -20 test200.out
echo "--- 行数 ---"
wc -l test200.out
echo "--- 有没有 'No hits' ---"
grep -c "No hits" test200.out || true

echo
echo "=== 换成 cmscan（对「序列扫模型库」才是对的工具）==="
time cmscan --rfam --noali --cpu 8 --fmt 2 \
  -o test200.cmscan.out \
  Rfam.cm test200.fa 2>&1 | tail -5

echo
echo "=== cmscan 命中数（fmt 2 = 表格输出）==="
wc -l test200.cmscan.out
head -5 test200.cmscan.out
echo "..."
grep -v '^#' test200.cmscan.out | head -10
