#!/bin/bash
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "########## 对照 1: cmscan --rfam（Rfam 官方推荐的本地用法）##########"
time cmscan --rfam --noali --cpu 4 -o known.cmscan.out Rfam.cm known.fa 2>&1 | tail -3
echo "--- 命中 ---"
grep -E "total hits|Total CM hits" known.cmscan.out | tail -2
grep -E "^RF[0-9]{5}" known.cmscan.out | head -5
sed -n '/Query:/,+3p' known.cmscan.out | head -12

echo
echo "########## 对照 2: 加 --cut_ga（Rfam 的 curated 阈值）##########"
cmscan --rfam --cut_ga --noali --cpu 4 -o known.ga.out Rfam.cm known.fa 2>&1 | tail -3
grep -E "Total CM hits" known.ga.out | tail -1
grep -E "^RF[0-9]{5}" known.ga.out | head -5

echo
echo "########## 对照 3: 不用 --rfam（普通模式）##########"
cmscan --noali --cpu 4 -o known.norfam.out Rfam.cm known.fa 2>&1 | tail -3
grep -E "Total CM hits" known.norfam.out | tail -1
grep -E "^RF[0-9]{5}" known.norfam.out | head -5

echo
echo "########## 命中行到底长什么样（从 cmscan 输出里直接看）##########"
grep -n -A3 "cm hits reported" known.cmscan.out | head -10
