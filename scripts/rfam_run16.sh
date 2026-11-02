#!/bin/bash
# 加到 16 片。cmscan 的 --cpu 并行效率差，多个进程比单进程多核更划算。
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "=== 机器 ==="
echo "核心 $(nproc)"
free -g | sed -n '2p'
cat /proc/loadavg

echo
echo "=== 当前进度 ==="
cat out/*.tbl 2>/dev/null | grep -vc '^#'
echo "cmscan 进程 $(pgrep -c cmscan || echo 0)"

echo
echo "=== 停掉旧的 8 片 ==="
pkill -f "cmscan --rfam" || true
sleep 3
echo "剩余 $(pgrep -c cmscan || echo 0)"

echo
echo "=== 切 16 片（保留已完成片的结果）==="
mkdir -p out16
awk -v n=16 -v d=out16 '
  /^>/ { i = (++k % n) }
  { print > (d "/c" sprintf("%02d", i) ".fa") }
' all.fa
ls out16/*.fa | wc -l

echo
echo "=== 并行启动 16 片，每片 --cpu 1 ==="
for f in out16/*.fa; do
  b=$(basename "$f" .fa)
  if [ -s "out16/$b.tbl" ] && [ "$(grep -vc '^#' out16/$b.tbl 2>/dev/null)" -gt 0 ]; then
    echo "  跳过 $b"; continue
  fi
  cmscan --rfam --noali --cpu 1 --tblout "out16/$b.tbl" -o "out16/$b.out" Rfam.cm "$f" &
done
echo "启动了 $(pgrep -c cmscan || echo 0) 个"
wait

echo
echo "=== 汇总 ==="
cat out16/*.tbl | grep -v '^#' > hits16.tbl
echo "命中行 $(wc -l < hits16.tbl)"
echo "有命中的序列 $(awk '{print $3}' hits16.tbl | sort -u | wc -l)"
echo "命中的家族 $(awk '{print $2}' hits16.tbl | sort -u | wc -l)"
