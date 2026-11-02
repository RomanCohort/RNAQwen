"""测各镜像的实速：每个只拉前 24 MB，8 秒看速度。"""
import time, urllib.request

PATH = "multimolecule/rnafm/resolve/main/model.safetensors"
ENDPOINTS = [
    ("hf-mirror",       f"https://hf-mirror.com/{PATH}"),
    ("hf-mirror-raw",   f"https://hf-mirror.com/{PATH}?download=true"),
    ("hf.co",           f"https://huggingface.co/{PATH}"),
    ("hf.co-dl",        f"https://huggingface.co/{PATH}?download=true"),
    ("modelscope-cdn",  "https://modelscope.cn/api/v1/models/multimolecule/rnafm"),
]

for name, url in ENDPOINTS:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-25165823"})
    try:
        t0 = time.time()
        # ⚠️ urllib 的 timeout 只吃标量。元组是 requests 的用法 —— 用错会 TypeError。
        with urllib.request.urlopen(req, timeout=20) as r:
            got, last = 0, time.time()
            while time.time() - t0 < 8.0:
                b = r.read(512 * 1024)
                if not b:
                    break
                got += len(b)
                last = time.time()
            el = max(time.time() - t0, 0.01)
            print("  %-16s %7.2f MB/s   (%5.1f MB / %.1fs)  status=%s  len=%s"
                  % (name, got / el / 1e6, got / 1e6, el, r.status,
                     r.headers.get("Content-Length")))
    except Exception as e:
        print("  %-16s FAIL  %s: %s" % (name, type(e).__name__, str(e)[:80]))
