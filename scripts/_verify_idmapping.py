"""补验 id_mapping.tsv.gz：拿 EBI 官方 md5 清单核对。

为什么必须做：下载时的完成判据是「大小已满」，而 aria2 的 .part 被预分配过
（只下了 14 MiB，文件就已经 2.33 GB）。**大小对不等于内容对** ——
models/rnafm 那次就是这么坏的（398,117,588 B 逐字节正确，155/205 张量全零）。
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

D = Path(r"C:\qwen-api\RNAQwen\data\external")
UA = {"User-Agent": "Mozilla/5.0"}
URL = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/md5/md5.tsv.gz"

p = D / "md5.tsv.gz"
if not p.exists() or p.stat().st_size < 1000:
    for i in range(10):
        try:
            req = urllib.request.Request(URL, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                p.write_bytes(r.read())
            print("  md5.tsv.gz -> %d B" % p.stat().st_size)
            break
        except Exception as e:
            print("  retry %d: %s: %s" % (i, type(e).__name__, str(e)[:60]))
    else:
        raise SystemExit("  **md5 清单下不来，无法校验**")
else:
    print("  md5.tsv.gz 已在（%d B）" % p.stat().st_size)

want = None
with gzip.open(p, "rt", errors="replace") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2 and parts[-1].split("/")[-1] == "id_mapping.tsv.gz":
            want = parts[0]
            break
print("  官方 md5 :", want)

f = D / "id_mapping.tsv.gz"
print("  文件大小 : %d B" % f.stat().st_size)
h = hashlib.md5()
with open(f, "rb") as fh:
    for b in iter(lambda: fh.read(8 << 20), b""):
        h.update(b)
got = h.hexdigest()
print("  实算 md5 :", got)
if want and got == want:
    print("\n  -> 一致 ✅ id_mapping.tsv.gz 内容可信")
else:
    print("\n  -> **不一致 ❌** 文件是坏的（预分配空洞）。删掉重下。")
