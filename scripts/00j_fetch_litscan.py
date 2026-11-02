"""下 RNAcentral litscan 全部 48 个文件（~53 MB）。

这是论文 "Lit Scan" 的源头：每条约含一条文献摘句 + 指向的 URS id。
拿它来给新序列生成 literature -> annotation -> QA，和现有 3488 条同源。
"""
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
DEST = ROOT / "data" / "external" / "litscan"
LIST = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/litscan/"
UA = {"User-Agent": "Mozilla/5.0"}


def get(url, timeout=90, tries=6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(3 + i * 3)
    raise last


DEST.mkdir(parents=True, exist_ok=True)
print("拉目录 ...", flush=True)
html = get(LIST).decode("utf-8", "ignore")
names = sorted(set(re.findall(r'href="(RNAcentral_annotations_[^"]+\.gz)"', html)))
print("找到 %d 个文件" % len(names), flush=True)

ok = skip = fail = 0
total = 0
for i, n in enumerate(names, 1):
    p = DEST / n
    if p.exists() and p.stat().st_size > 10000:
        skip += 1
        total += p.stat().st_size
        continue
    try:
        data = get(LIST + n)
        p.write_bytes(data)
        ok += 1
        total += len(data)
        print("  [%2d/%d] %s  %d B" % (i, len(names), n, len(data)), flush=True)
    except Exception as e:
        fail += 1
        print("  [%2d/%d] %s 失败 %s" % (i, len(names), n, type(e).__name__), flush=True)

print("\n完成: 新下 %d / 跳过 %d / 失败 %d   合计 %.1f MB"
      % (ok, skip, fail, total / 1e6), flush=True)
print("目录: %s" % DEST, flush=True)
