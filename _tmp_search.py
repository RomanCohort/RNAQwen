import json, urllib.request, urllib.error
def post(url, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json","User-Agent":"Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:600]
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}".encode()

for name in ["rnafm","RNA-FM"]:
    s,b = post("https://www.modelscope.cn/api/v1/dolphin/models", {"PageSize":20,"PageNumber":1,"Name":name,"SortBy":"Default"})
    print("=== search", name, s)
    txt = b.decode("utf-8","replace") if isinstance(b,bytes) else b
    try:
        j = json.loads(txt)
        models = (j.get("Data") or {}).get("Model",{}).get("Models") or (j.get("Data") or {}).get("Models") or []
        for m in models:
            print("   ", m.get("Path","?"), "/", m.get("Name","?"), "|", (m.get("ChineseName") or "")[:40])
        if not models: print("   raw:", txt[:500])
    except Exception as e:
        print("   raw:", txt[:500])
