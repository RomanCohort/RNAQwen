"""看 ModelScope 那份 rnafm 的 config / README，判断是不是同一套权重。"""
import json, urllib.request

RID = "ZhejiangLab-LifeScience/rnafm"
UA = {"User-Agent": "Mozilla/5.0"}

for fn in ["config.json", "configuration.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]:
    url = f"https://modelscope.cn/models/{RID}/resolve/master/{fn}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode("utf-8", "ignore")
        print(f"===== {fn} =====")
        print(txt[:1200])
        print()
    except Exception as e:
        print(f"===== {fn} FAIL {type(e).__name__}: {str(e)[:80]}\n")

print("===== HF 的 config.json（对比）=====")
try:
    req = urllib.request.Request(
        "https://huggingface.co/multimolecule/rnafm/resolve/main/config.json", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.read().decode("utf-8", "ignore")[:1200])
except Exception as e:
    print("  FAIL:", e)

print("\n===== ModelScope README 头部 =====")
try:
    req = urllib.request.Request(f"https://modelscope.cn/models/{RID}/resolve/master/README.md", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        print(r.read().decode("utf-8", "ignore")[:2500])
except Exception as e:
    print("  FAIL:", e)
