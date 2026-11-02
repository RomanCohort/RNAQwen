"""探 ModelScope 上 ZhejiangLab-LifeScience 组织下的模型（国内直连 7.8 MB/s）。"""
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def probe(rid):
    url = f"https://modelscope.cn/api/v1/models/{rid}/repo/files?Revision=master&Root="
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.load(r)
        files = (d.get("Data") or {}).get("Files") or []
        return rid, 200, files
    except urllib.error.HTTPError as e:
        return rid, e.code, []
    except Exception as e:
        return rid, -1, str(e)[:50]


NAMES = [
    # 我们真正想要的
    "ernierna", "ERNIE-RNA", "ernie-rna", "ERNIE_RNA", "ernierna-ss",
    # 已知存在（做对照）
    "rnafm", "rnaernie",
    # 同体系的其他 backbone
    "rinalmo", "rnamsm", "utrlm", "rnabert", "dnabert2", "hyenadna",
    "progen2", "esmc", "optmrl", "rnafm-ss",
]
ORGS = ["ZhejiangLab-LifeScience", "AI-ModelScope", "multimolecule"]

cands = [f"{o}/{n}" for o in ORGS for n in NAMES]
print(f"探 {len(cands)} 个候选 ...")
print()
found = []
with ThreadPoolExecutor(max_workers=10) as ex:
    for rid, st, files in ex.map(probe, cands):
        if st == 200:
            found.append((rid, files))
            print(f"  ✅ {rid}")
        elif st not in (404,):
            print(f"  ?  {st} {rid}")

print()
print("=" * 76)
for rid, files in found:
    print(f"=== {rid} ===")
    for f in files:
        n = f.get("Name")
        if n in ("model.safetensors", "pytorch_model.bin", "config.json"):
            print("   %-22s size=%-12s sha256=%s" % (n, f.get("Size"), f.get("Sha256")))
    # config
    try:
        url = f"https://modelscope.cn/models/{rid}/resolve/master/config.json"
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            c = json.load(r)
        print("   model_type=%s hidden=%s vocab=%s pos=%s arch=%s" % (
            c.get("model_type"), c.get("hidden_size"), c.get("vocab_size"),
            c.get("max_position_embeddings"), c.get("architectures")))
    except Exception as e:
        print("   config 读取失败:", str(e)[:60])
    print()
