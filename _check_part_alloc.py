"""查 .part 是不是被预分配了（未写入区域=零）。这正是上次损坏的机制。"""
import os, time
from pathlib import Path

P = Path(r"C:\qwen-api\RNAQwen\models\rnafm\model.safetensors.part")

for i in range(3):
    st = P.stat()
    print(f"  [t{i}] 逻辑大小 = {st.st_size:,} B ({st.st_size/1e6:.1f} MB)")
    time.sleep(1)

# 在几个位置采样，看是不是零
with open(P, "rb") as f:
    for off_mb in [1, 10, 30, 50, 55, 60, 100, 200, 300, 350]:
        off = off_mb * 1024 * 1024
        if off + 4096 > P.stat().st_size:
            continue
        f.seek(off)
        b = f.read(4096)
        nz = sum(1 for x in b if x != 0)
        print(f"    @{off_mb:4d} MB : 非零字节 {nz:5d}/4096  {'零填充' if nz == 0 else '有数据'}")

print("\n=== NTFS 稀疏 / 分配情况 ===")
print("  (fsutil sparse queryflag)")
