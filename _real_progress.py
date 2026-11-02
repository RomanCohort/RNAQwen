"""真实进度：.part 逻辑大小被预分配污染了，不能看它。
直接数「有多少 1MB 块是真有数据的」。"""
from pathlib import Path
import time

P = Path(r"C:\qwen-api\RNAQwen\models\rnafm\model.safetensors.part")
TOTAL = 398117588
CH = 1024 * 1024

t0 = time.time()
have, tot = 0, 0
with open(P, "rb") as f:
    while True:
        b = f.read(CH)
        if not b:
            break
        tot += 1
        # 整块全零 => 还没写
        if any(b):
            have += 1
size = P.stat().st_size
print("逻辑大小     : %d B (%.1f MB)" % (size, size / 1e6))
print("扫描块数     : %d" % tot)
print("有数据的块   : %d  (%.1f%%)" % (have, 100.0 * have / max(tot, 1)))
print("估算已下     : %.1f MB / %.1f MB" % (have, TOTAL / 1e6))
print("扫描耗时     : %.1fs" % (time.time() - t0))
print()
print("⚠️ 注意：含数据块的尾部可能有部分是零，这个数略高估。")
print("   真实进度以 aria2 自己的计数器为准。")
