"""从 NGDC 数据库总目录里筛 ncRNA / RNA 相关的国内库。"""
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"}

URL = "https://ngdc.cncb.ac.cn/databasecommons/"
req = urllib.request.Request(URL, headers=UA)
with urllib.request.urlopen(req, timeout=30) as r:
    html = r.read().decode("utf-8", "ignore")
print("页面 %d 字节" % len(html))

# 抓所有链接 + 文本
links = re.findall(r'href="([^"]+)"[^>]*>([^<]{2,80})<', html)
print("链接 %d 条" % len(links))

KEY = re.compile(r'rna|noncode|lncrna|microrna|mirna|circrna|ncrna|transcript', re.I)
hits = [(t.strip(), h) for h, t in links if KEY.search(t)]
seen = set()
print()
print("=" * 90)
print("RNA / ncRNA 相关条目")
print("=" * 90)
for t, h in hits:
    if t in seen:
        continue
    seen.add(t)
    full = h if h.startswith("http") else ("https://ngdc.cncb.ac.cn" + h if h.startswith("/") else h)
    print("  %-40s %s" % (t[:40], full[:90]))

if not hits:
    # 也许内容在表格/JSON 里，直接扫全文里的 RNA 词
    print()
    print("链接里没命中，扫全文关键词:")
    for kw in ["noncode", "lncrna", "ncrna", "microrna", "circrna", "transcriptome"]:
        idx = [m.start() for m in re.finditer(kw, html, re.I)][:3]
        for i in idx:
            print("   [%s] ...%s..." % (kw, " ".join(html[max(0,i-80):i+80].split())))
