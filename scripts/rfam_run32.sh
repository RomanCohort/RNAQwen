#!/bin/bash
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

echo "=== 停 16 片 ==="
pkill -f "cmscan --rfam" || true
sleep 3
echo "剩余 $(pgrep -c cmscan || echo 0)"

echo
echo "=== 切 32 片（= 核心数）==="
rm -rf out32; mkdir -p out32
awk -v n=32 -v d=out32 '
  /^>/ { i = (++k % n) }
  { print > (d "/c" sprintf("%02d", i) ".fa") }
' all.fa
echo "片数 $(ls out32/*.fa | wc -l)"
python3 - <<'PY'
import glob
tot = 0
for f in sorted(glob.glob('out32/*.fa')):
    n = sum(1 for l in open(f) if l.startswith('>'))
    tot += n
print(f'总序列 {tot:,}')
PY

echo
echo "=== 启动 32 片，每片 --cpu 1 ==="
for f in out32/*.fa; do
  b=$(basename "$f" .fa)
  cmscan --rfam --noali --cpu 1 --tblout "out32/$b.tbl" -o "out32/$b.out" Rfam.cm "$f" &
done
sleep 20
echo "cmscan 进程 $(pgrep -c cmscan || echo 0)"
echo "负载 $(cat /proc/loadavg)"

wait

echo
echo "=== 完成，汇总 ==="
cat out32/*.tbl | grep -v '^#' > hits32.tbl
echo "命中行 $(wc -l < hits32.tbl)"
echo "有命中的序列 $(awk '{print $3}' hits32.tbl | sort -u | wc -l)"
echo "命中的家族 $(awk '{print $2}' hits32.tbl | sort -u | wc -l)"
echo
echo "前 20 条:"; head -20 hits32.tbl
