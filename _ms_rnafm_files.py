"""ModelScope 上 ZhejiangLab-LifeScience/rnafm 的文件清单 + 大小。"""
import json, urllib.request

RID = "ZhejiangLab-LifeScience/rnafm"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
HF_SIZE = 398117588
HF_SHA = "ec69333eb894dbefb580e690b0cbe1cbae9845fefd2bc78da914cd14226a2828"

print("=== repo files (master) ===")
for rev in ["master", "main"]:
    url = f"https://modelscope.cn/api/v1/models/{RID}/repo/files?Revision={rev}&Root="
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        files = (d.get("Data") or {}).get("Files") or []
        print(f"  Revision={rev}: {len(files)} 个文件")
        for f in files:
            sz = f.get("Size") or 0
            flag = ""
            if f.get("Name") == "model.safetensors":
                flag = "  <== 大小和 HF 一致!" if sz == HF_SIZE else f"  <== 大小 {sz} != HF {HF_SIZE}"
            print("    %-34s %14s  %s%s" % (f.get("Name"), sz, f.get("Type"), flag))
        break
    except Exception as e:
        print(f"  Revision={rev} FAIL {type(e).__name__}: {str(e)[:100]}")

print()
print("=== 直接 HEAD resolve URL ===")
for fn in ["model.safetensors", "pytorch_model.bin", "config.json"]:
    url = f"https://modelscope.cn/models/{RID}/resolve/master/{fn}"
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            print("  %-24s status=%s  Content-Length=%s" % (
                fn, r.status, r.headers.get("Content-Length")))
    except Exception as e:
        print("  %-24s FAIL %s" % (fn, str(e)[:80]))

print()
print("=== 期望 ===")
print("  size  ", HF_SIZE)
print("  sha256", HF_SHA)
