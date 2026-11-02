"""测 Google Drive 上 ERNIE-RNA_pretrain.pt 能不能下、多快、多大。

GDrive 大文件会返回一个「病毒扫描」确认页，需要带 confirm token + uuid cookie 二次请求。
gdown 就是干这个的；这里手写一遍，顺便把真实大小和速度量出来。
"""
import http.cookiejar
import re
import time
import urllib.request

FILE_ID = "1CmNxJxgjDhRoBdlDODFFjDNNzJMHVzmO"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"}

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

print("=" * 74)
print("1) 取确认页 / token")
print("=" * 74)
url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
try:
    with opener.open(urllib.request.Request(url, headers=UA), timeout=40) as r:
        html = r.read().decode("utf-8", "ignore")
        final = r.url
    print(f"  状态 ok  url={final[:80]}  html {len(html)} 字符")
except Exception as e:
    print("  失败:", type(e).__name__, str(e)[:100])
    raise SystemExit(1)

name = re.search(r'<span class="uc-name-size"><a[^>]*>([^<]+)</a>', html)
print("  文件名猜测:", name.group(1) if name else "(没解析到)")

confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html)
uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html)
print("  confirm token:", (confirm.group(1)[:20] + "...") if confirm else "无（可能不需要）")
print("  uuid:", uuid.group(1) if uuid else "无")

if confirm:
    url2 = (f"https://drive.usercontent.google.com/download?id={FILE_ID}"
            f"&export=download&confirm={confirm.group(1)}")
    if uuid:
        url2 += f"&uuid={uuid.group(1)}"
elif "drive.usercontent.google.com" in final or "export=download" in final:
    url2 = final
else:
    url2 = url
print()

print("=" * 74)
print("2) HEAD 拿大小")
print("=" * 74)
try:
    req = urllib.request.Request(url2, headers=UA, method="HEAD")
    with opener.open(req, timeout=40) as r:
        print("  status", r.status)
        for k, v in r.headers.items():
            if k.lower() in ("content-length", "content-type", "content-disposition"):
                print(f"  {k}: {v}")
except Exception as e:
    print("  HEAD 失败:", type(e).__name__, str(e)[:100])

print()
print("=" * 74)
print("3) 实拉 20 秒测速")
print("=" * 74)
try:
    req = urllib.request.Request(url2, headers={**UA, "Range": "bytes=0-52428799"})
    t0 = time.time()
    got = 0
    with opener.open(req, timeout=60) as r:
        print("  status", r.status, " content-length:", r.headers.get("Content-Length"))
        while time.time() - t0 < 20:
            b = r.read(256 * 1024)
            if not b:
                break
            got += len(b)
    el = max(time.time() - t0, 0.01)
    print(f"  拉到 {got/1e6:.2f} MB / {el:.1f}s = {got/el/1e6:.2f} MB/s")
    if got < 1024:
        print("  **几乎拉不到东西** —— GDrive 这条路可能被限流/需要登录")
except Exception as e:
    print("  失败:", type(e).__name__, str(e)[:120])

print()
print("=" * 74)
print("4) 对照：已知的 multimolecule/ernierna 声明值")
print("=" * 74)
print("  size   345,071,800")
print("  sha256 02274cb79699fb642a881fd1de6df56786291503aa9e2dcaec63e3ae46ab08eb")
print("  ⚠️ 这是**另一份 artifact**（作者原始 .pt），sha256 必然不同。")
print("     用它需要做键名映射进 ErnieRnaModel，并且必须重跑 01c 健全性检查。")
