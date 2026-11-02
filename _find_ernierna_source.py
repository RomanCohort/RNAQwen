"""给 multimolecule/ernierna 定源。

判据不是"哪个快"，是"是不是同一份字节"：
  如果 ModelScope 那份声明的 sha256 == HF 声明的 sha256，
  那源的选择就无关紧要了 —— 同一份 artifact，谁下都一样。
"""
import json, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
HF_ID = "multimolecule/ernierna"
WANT = "model.safetensors"


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


print("=" * 74)
print("1) HF 声明值")
print("=" * 74)
hf_sha = hf_size = None
try:
    d = get_json(f"https://huggingface.co/api/models/{HF_ID}?blobs=true")
    for s in d.get("siblings", []):
        if s["rfilename"] == WANT:
            lfs = s.get("lfs") or {}
            hf_sha, hf_size = lfs.get("sha256"), lfs.get("size")
            print(f"  {WANT}: size={hf_size} sha256={hf_sha}")
    for rev in ["main", "master"]:
        try:
            c = get_json(f"https://huggingface.co/{HF_ID}/resolve/{rev}/config.json")
            print(f"  config: model_type={c.get('model_type')} hidden={c.get('hidden_size')} "
                  f"vocab={c.get('vocab_size')} pos={c.get('max_position_embeddings')}")
            break
        except Exception:
            pass
except Exception as e:
    print("  HF API 失败:", e)

print()
print("=" * 74)
print("2) ModelScope 候选")
print("=" * 74)
MS_CANDS = [
    "ZhejiangLab-LifeScience/ernierna",
    "ZhejiangLab-LifeScience/ERNIE-RNA",
    "multimolecule/ernierna",
    "AI-ModelScope/ernierna",
    "iic/ernierna",
    "damo/ernierna",
    "ZhejiangLab-LifeScience/rnaernie",
    "AI-ModelScope/ERNIE-RNA",
]


def probe(rid):
    url = f"https://modelscope.cn/api/v1/models/{rid}/repo/files?Revision=master&Root="
    try:
        d = get_json(url)
        files = (d.get("Data") or {}).get("Files") or []
        return rid, 200, files
    except urllib.error.HTTPError as e:
        return rid, e.code, []
    except Exception as e:
        return rid, -1, str(e)[:60]


found = []
with ThreadPoolExecutor(max_workers=8) as ex:
    for rid, st, files in ex.map(probe, MS_CANDS):
        mark = "✅" if st == 200 else "  "
        print(f"  {mark} {st:>4}  {rid}")
        if st == 200:
            found.append((rid, files))

print()
for rid, files in found:
    print(f"=== {rid} ===")
    for f in files:
        n = f.get("Name")
        if n in (WANT, "config.json"):
            print("   %-24s size=%-12s sha256=%s" % (n, f.get("Size"), f.get("Sha256")))

print()
print("=" * 74)
print("3) 判定")
print("=" * 74)
if hf_sha:
    print(f"  HF 声明       size={hf_size}  sha256={hf_sha}")
match = False
for rid, files in found:
    for f in files:
        if f.get("Name") == WANT:
            same = (f.get("Sha256") == hf_sha)
            match = match or same
            print(f"  {rid:<44} sha256={f.get('Sha256')}  "
                  f"{'== HF ✅ 同一份字节' if same else '!= HF'}")
print()
if match:
    print("  -> ModelScope 那份与 HF 完全同字节。可以从 ModelScope 下（快），也可以等 HF，无区别。")
else:
    print("  -> 没有逐字节相同的副本。要么等 HF 下完，要么另行判断。")
