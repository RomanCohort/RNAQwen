"""在 ModelScope 上搜 RNA-FM 的镜像。"""
import json, urllib.request, urllib.parse

def get(url, data=None, headers=None):
    h = {"User-Agent": "curl/8.0"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

# 1) 简单 GET 搜索
for name in ["rnafm", "RNA-FM", "rna_fm", "multimolecule"]:
    for tmpl in ["https://modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&Name={}",
                 "https://www.modelscope.cn/api/v1/models?Name={}"]:
        url = tmpl.format(urllib.parse.quote(name))
        try:
            d = get(url)
            ms = (d.get("Data") or {}).get("Models") or d.get("Data") or []
            if isinstance(ms, dict):
                ms = ms.get("Models", [])
            print(f"[GET] {name:14} -> {len(ms)} 个")
            for m in ms[:12]:
                if isinstance(m, dict):
                    print("      ", m.get("Path") or m.get("Name") or m.get("ModelId"))
            break
        except Exception as e:
            print(f"[GET] {name:14} FAIL {type(e).__name__}: {str(e)[:70]}")

# 2) dolphin POST 搜索（ModelScope 网页搜索走这个）
print("\n=== dolphin POST 搜索 ===")
try:
    body = json.dumps({"PageSize": 20, "PageNumber": 1, "Name": "rnafm",
                       "SortBy": "Default"}).encode()
    d = get("https://modelscope.cn/api/v1/dolphin/models", data=body,
            headers={"Content-Type": "application/json"})
    print(json.dumps(d, ensure_ascii=False)[:1500])
except Exception as e:
    print("  FAIL:", type(e).__name__, str(e)[:200])
