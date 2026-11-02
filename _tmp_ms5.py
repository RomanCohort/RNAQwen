import urllib.request, urllib.error, re, json
def get(url, data=None, hdr=None):
    h={"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*"}
    if hdr: h.update(hdr)
    req=urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:200]
    except Exception as e: return "ERR", str(e).encode()

# ModelScope does not host rnafm under tested names; try the hub search API variants
cands = [
 "https://www.modelscope.cn/api/v1/hub/models?Name=rnafm",
 "https://modelscope.cn/api/v1/dolphin/models?PageSize=20&PageNumber=1&Name=rnafm",
 "https://www.modelscope.cn/openapi/v1/models?name=rnafm",
 "https://www.modelscope.cn/api/v1/models/AI-ModelScope/RNA-FM",
 "https://www.modelscope.cn/api/v1/models/multimolecule/RNA-FM",
 "https://www.modelscope.cn/api/v1/models/iic/RNA-FM",
]
for u in cands:
    s,b=get(u)
    txt = b.decode("utf-8","replace") if isinstance(b,bytes) else str(b)
    print(s, u.split("/api")[-1][:70], "|", txt[:120].replace("\n"," "))
