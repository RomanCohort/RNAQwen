"""比三条路拿 multimolecule/rnafm 的实速。

  HF 官方       https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors
  hf-mirror     https://hf-mirror.com/multimolecule/rnafm/resolve/main/model.safetensors   <- 同一份 LFS 对象
  ModelScope    ZhejiangLab-LifeScience/rnafm  <- **不同产物**，只做速度参考，不能顶替
  服务器 dl     https://www.romancohort.cloud/dl/rnafm-model.safetensors   <- 服务器代取 + 直连取件
"""
import time
import urllib.request

UA = {"User-Agent": "curl/8.0"}
SEC = 20

TARGETS = [
    ("HF 官方", "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors", True),
    ("hf-mirror", "https://hf-mirror.com/multimolecule/rnafm/resolve/main/model.safetensors", True),
    ("ModelScope(异版本)", "https://modelscope.cn/models/ZhejiangLab-LifeScience/rnafm/resolve/master/model.safetensors", False),
    ("服务器 dl 取件", "https://www.romancohort.cloud/dl/rnafm-model.safetensors", True),
]

print("=" * 92)
print("各源实速（%d 秒，Range 32MB）" % SEC)
print("=" * 92)
best = None
for label, url, same_artifact in TARGETS:
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-33554431"})
        t0 = time.time()
        got = 0
        with urllib.request.urlopen(req, timeout=30) as r:
            while time.time() - t0 < SEC:
                b = r.read(256 * 1024)
                if not b:
                    break
                got += len(b)
        el = max(time.time() - t0, 0.01)
        kbs = got / el / 1024
        ok = got > 65536
        mark = "✅" if same_artifact else "⚠️异版本"
        print("  %-22s %s  %8.1f KB/s  (%d 字节 / %.0fs)" % (label, mark, kbs, got, el))
        if ok and same_artifact and (best is None or kbs > best[1]):
            best = (label, kbs, url)
    except Exception as e:
        print("  %-22s 失败  %s: %s" % (label, type(e).__name__, str(e)[:60]))

print()
print("=" * 92)
if best:
    print("同产物里最快的: %s  (%.1f KB/s)" % (best[0], best[1]))
    print("  398,117,588 B / %.1f KB/s = %.0f 分钟" % (best[1], 398117588 / 1024 / best[1] / 60))
    print()
    print("  命令:")
    print("    python scripts/00d_fetch_verified.py --repo multimolecule/rnafm \\")
    print("        --dest models/rnafm --sha256 ec69333eb894dbefb580e690b0cbe1cbae9845fefd2bc78da914cd14226a2828 \\")
    print("        --size 398117588 --host %s" % ("hf-mirror.com" if "mirror" in best[0] else "huggingface.co"))
else:
    print("没有可用的同产物源")
