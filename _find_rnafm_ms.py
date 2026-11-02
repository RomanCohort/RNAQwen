"""在 ModelScope 上找 RNA-FM。

两条路并行：
  A) 直接探候选 repo id（用已知存在的 /repo/files 端点，404 vs 200 很好判断）
  B) 抓搜索页 HTML / 试几种搜索 API 形状
不碰 HF 那条下载。
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}


def fetch(url, data=None, timeout=25):
    req = urllib.request.Request(url, data=data, headers=UA,
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


# ---------- A) 探候选 repo ----------
CANDIDATES = [
    "multimolecule/rnafm", "AI-ModelScope/rnafm", "iic/rnafm", "damo/rnafm",
    "ml4bio/RNA-FM", "ml4bio/rnafm", "RNA-FM/rnafm", "rnafm/rnafm",
    "bioscience/rnafm", "BioScience/rnafm", "ShanghaiTech/rnafm",
    "OpenBMB/rnafm", "BAAI/rnafm", "qwen/rnafm", "modelscope/rnafm",
    "AI-ModelScope/RNA-FM", "iic/RNA-FM", "damo/RNA-FM",
    "YuanLab/rnafm", "XLab/rnafm", "OmniGen/rnafm", "LifeScience/rnafm",
    "ai4s/rnafm", "bio/rnafm", "genomics/rnafm", "RNA/rnafm",
    "AI-ModelScope/multimolecule-rnafm", "multimolecule/RNA-FM",
    "iic/multimolecule-rnafm", "damo/multimolecule-rnafm",
    "svjack/rnafm", "wangrongsheng/rnafm", "AI-ModelScope/rna_fm",
]


def probe_file_list(rid):
    url = (f"https://modelscope.cn/api/v1/models/{rid}/repo/files"
           f"?Revision=master&Root=")
    try:
        st, body = fetch(url)
        return rid, st, body[:200]
    except urllib.error.HTTPError as e:
        return rid, e.code, b""
    except Exception as e:
        return rid, -1, str(e)[:80].encode()


print("=" * 74)
print("A) 探候选 repo id（200 = 存在）")
print("=" * 74)
found = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for rid, st, body in ex.map(probe_file_list, CANDIDATES):
        mark = "✅" if st == 200 else "  "
        print(f"  {mark} {st:>4}  {rid}")
        if st == 200:
            found.append((rid, body))

# ---------- B) 搜索 ----------
print()
print("=" * 74)
print("B) 搜索")
print("=" * 74)

for kw in ["rnafm", "RNA-FM", "multimolecule"]:
    url = "https://modelscope.cn/models?" + urllib.parse.urlencode({"name": kw, "page": 1})
    try:
        st, body = fetch(url)
        html = body.decode("utf-8", "ignore")
        repos = sorted(set(re.findall(r'href="/(?:models/)?([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)"', html)))
        repos = [r for r in repos if "rna" in r.lower() or "fm" in r.lower()][:20]
        print(f"  [HTML] {kw:14} status={st} len={len(html)} 命中={len(repos)}")
        for r in repos[:15]:
            print("        ", r)
    except Exception as e:
        print(f"  [HTML] {kw:14} FAIL {type(e).__name__}: {str(e)[:70]}")

# 试几种搜索 API 形状
print()
SEARCH_SHAPES = [
    ("POST", "https://modelscope.cn/api/v1/dolphin/models", {"Content-Type": "application/json"}),
    ("POST", "https://www.modelscope.cn/api/v1/dolphin/models", {"Content-Type": "application/json"}),
    ("POST", "https://modelscope.cn/api/v1/models/search", {"Content-Type": "application/json"}),
    ("GET", "https://modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&SortBy=Default&Name=RNA-FM", None),
    ("GET", "https://modelscope.cn/api/v1/models?pageNumber=1&pageSize=20&name=rnafm", None),
]
for method, url, hdr in SEARCH_SHAPES:
    try:
        data = json.dumps({"PageSize": 20, "PageNumber": 1, "SortBy": "Default",
                           "Name": "rnafm", "Target": "", "SingleCriterion": []}).encode() \
            if method == "POST" else None
        st, body = fetch(url, data=data)
        txt = body.decode("utf-8", "ignore")
        print(f"  [{method}] {st}  {url[:70]}")
        print(f"       {txt[:300]}")
    except urllib.error.HTTPError as e:
        print(f"  [{method}] {e.code}  {url[:70]}")
    except Exception as e:
        print(f"  [{method}] FAIL {type(e).__name__}  {url[:70]}")

print()
print("=" * 74)
print("结论")
print("=" * 74)
if found:
    for rid, body in found:
        print(f"  ✅ 找到 {rid}")
        print(f"     {body[:400]}")
else:
    print("  候选里一个都没有。")
