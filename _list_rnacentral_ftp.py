"""列 RNAcentral 在 EBI FTP 上的转储目录，找 description 和 id_mapping。"""
import re
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"}

BASE = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/"


def listing(url, label):
    print("=" * 92)
    print(label)
    print(url)
    print("=" * 92)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=40) as r:
            html = r.read().decode("utf-8", "ignore")
    except Exception as e:
        print("  失败:", type(e).__name__, str(e)[:80])
        return []
    rows = re.findall(r'href="([^"]+)"[^>]*>([^<]*)</a>\s*([\d\-]+ [\d:]+)?\s*([\d\.KMG]+)?', html)
    out = []
    for href, name, date, size in rows:
        if href.startswith("?") or href.startswith("/") or href in ("../",):
            continue
        out.append((name.strip(), size.strip(), href))
        print("  %-52s %10s  %s" % (name.strip()[:52], size.strip(), date.strip()))
    return out


for sub, label in [
    ("", "根目录"),
    ("current_release/", "current_release"),
]:
    listing(BASE + sub, label)
    print()
