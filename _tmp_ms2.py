import json, urllib.request, urllib.error
def get(url, hdr=None):
    h={"User-Agent":"Mozilla/5.0","Accept":"*/*"}
    if hdr: h.update(hdr)
    req=urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=45) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:400]
    except Exception as e: return "ERR", f"{type(e).__name__}: {e}".encode()

tests = [
 ("GET","https://www.modelscope.cn/api/v1/models?Name=rnafm&PageSize=20&PageNumber=1"),
 ("GET","https://modelscope.cn/api/v1/dolphin/models?Name=rnafm"),
 ("GET","https://www.modelscope.cn/api/v1/dolphin/models?Name=rnafm&PageSize=20&PageNumber=1"),
 ("GET","https://www.modelscope.cn/api/v1/models/multimolecule/rnafm"),
 ("GET","https://www.modelscope.cn/api/v1/models?name=rnafm"),
]
for m,u in tests:
    s,b=get(u)
    print(m,u,"->",s)
    if isinstance(b,bytes): print("    ", b[:350].decode("utf-8","replace"))
