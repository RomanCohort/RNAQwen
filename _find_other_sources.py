"""找 RNA 注释数据的其他源。

为什么需要：rnacentral.org/api 从本机（隧道）和服务器（新加坡）**都是 403**。
那不是网络问题，是端点问题。所以要找别的路。

候选：
  A. RNAcentral 换端点（去掉参数的 /api/v1/rna/、EBI 的镜像路径）
  B. RNAcentral FTP 全量转储（EBI 的 ftp.ebi.ac.uk）
  C. NCBI Entrez —— **这条最近是通的**（2026-10-08 采到 1000 条）
  D. Rfam 家族描述
  E. 本地已有的 RNAcentral 序列集（multimolecule/rnacentral.1024）里有没有描述
"""
import json
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36",
      "Accept": "*/*"}

TESTS = [
    ("A1 RNAcentral /api/v1/rna/ (无参)", "https://rnacentral.org/api/v1/rna/"),
    ("A2 RNAcentral /api/v1/rna/URS.../", "https://rnacentral.org/api/v1/rna/URS000075B951/"),
    ("A3 EBI 路径", "https://www.ebi.ac.uk/Tools/ebi/search/rnacentral/api/v1/rna/?external_id=NR_136892.1"),
    ("A4 RNAcentral 首页", "https://rnacentral.org/"),
    ("B1 EBI FTP RNAcentral", "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/"),
    ("B2 EBI FTP current", "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/"),
    ("D1 Rfam FTP", "https://ftp.ebi.ac.uk/pub/databases/Rfam/"),
    ("C1 NCBI eutils", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?retmode=json"),
    ("C2 NCBI nuccore NR_136892.1",
     "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NR_136892.1&rettype=gb&retmode=text"),
]

print("=" * 96)
print("源可达性（本机，走隧道）")
print("=" * 96)
for label, url in TESTS:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read(1200)
        txt = body.decode("utf-8", "ignore")
        print("  [%s] %-38s %d 字节" % (r.status, label, len(body)))
        snippet = " ".join(txt.split())[:150]
        print("        %s" % snippet)
    except urllib.error.HTTPError as e:
        print("  [%s] %-38s HTTPError" % (e.code, label))
    except Exception as e:
        print("  [ERR] %-38s %s: %s" % (label, type(e).__name__, str(e)[:60]))
