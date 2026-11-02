"""测 RNAcentral API 能不能按 RefSeq accession 反查到 URS + 策展描述。

如果能 -> 那 993 条新序列可以补上**同类**的 annotation，是最干净的扩数据路径。
"""
import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# 从 rna_literature_1k 里取的真实 accession
ACCS = ["NR_136892.1", "NR_131335.1", "NR_146480.1", "NR_184134.1"]


def get(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


print("=" * 92)
print("1) 按 accession 反查（RNAcentral xref / api）")
print("=" * 92)
for acc in ACCS:
    got = False
    for tmpl in [
        "https://rnacentral.org/api/v1/rna/?external_id={}",
        "https://rnacentral.org/api/v1/rna/?external_id={}&format=json",
        "https://www.ebi.ac.uk/Tools/ebi/search/rnacentral/api/v1/rna/?external_id={}",
    ]:
        url = tmpl.format(urllib.parse.quote(acc))
        try:
            d = get(url)
            n = d.get("count")
            print("  [OK] %s -> count=%s  (%s)" % (acc, n, url[:60]))
            res = d.get("results") or []
            if res:
                r0 = res[0]
                print("       urs=%s" % r0.get("rnacentral_id"))
                print("       rna_type=%s  length=%s" % (r0.get("rna_type"), r0.get("length")))
                desc = (r0.get("description") or "").replace("\n", " ")
                print("       description: %s" % desc[:260])
            got = True
            break
        except Exception as e:
            print("  [--] %s  %s: %s" % (acc, type(e).__name__, str(e)[:60]))
    if not got:
        print("  [FAIL] %s 三种写法都不行" % acc)
    time.sleep(1)
