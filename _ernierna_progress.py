"""ernierna 真实进度：数非零 MB 块。逻辑大小不可信。"""
import time
from pathlib import Path

P = Path(r"C:\qwen-api\RNAQwen\models\ernierna\model.safetensors.part")
EXPECT = 345071800
CH = 1 << 20


def scan():
    have = tot = 0
    with open(P, "rb") as f:
        while True:
            b = f.read(CH)
            if not b:
                break
            tot += 1
            if any(b):
                have += 1
    return have, tot


a = scan()
print("逻辑大小 : %d B  (不可信)" % P.stat().st_size)
print("真实数据 : %d MB / %d MB 块   %.1f%%" % (a[0], a[1], 100.0 * a[0] / max(a[1], 1)))
print("期望     : ~%d MB" % (EXPECT // CH))
print()
print("等 90 秒再测一次 ...")
time.sleep(90)
b = scan()
d = b[0] - a[0]
print("真实数据 : %d MB   (90 秒内 +%d MB)" % (b[0], d))
if d > 0:
    remain = (EXPECT // CH) - b[0]
    print("速率     : %.0f KB/s" % (d * 1024 / 90.0))
    print("剩余     : ~%d MB  ->  ETA ~%.0f 分钟" % (remain, remain * 90.0 / d / 60))
else:
    print("**90 秒零进展 —— 又卡住了**")
