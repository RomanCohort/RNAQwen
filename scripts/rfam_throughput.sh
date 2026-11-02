#!/bin/bash
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "########## 先确认 tRNA 命中哪个家族 ##########"
cmscan --rfam --noali --cpu 4 -o known2.out Rfam.cm known.fa 2>&1 | tail -1
awk '/^Query:.*known_tRNA/,/^Query:.*known_5S/' known2.out | head -20

echo
echo "########## 测吞吐：200 条 ##########"
time cmscan --rfam --noali --cpu 8 --tblout test200.tbl -o test200.cmscan.out Rfam.cm test200.fa 2>&1 | tail -2

echo
echo "--- tblout 命中数（表格式输出，每命中一行）---"
grep -vc '^#' test200.tbl || echo 0
echo "--- 命中前 10 行 ---"
grep -v '^#' test200.tbl | head -10
echo "--- 有命中的序列数 ---"
grep -v '^#' test200.tbl | awk '{print $3}' | sort -u | wc -l

echo
echo "########## 推算全量时间 ##########"
echo "200 条 -> 见上面 real 时间；145,201 条按比例"
