import json, urllib.request, urllib.error, urllib.parse
def get(url, data=None, hdr=None):
    h = {"User-Agent":"Mozilla/5.0","Accept":"application/json"}
    if hdr: h.update(hdr)
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:400]
    except Exception as e:
        return "ERR", f"{type(e).__name__}: {e}".encode()

for repo in ["multimolecule/rnafm","AI-ModelScope/rnafm","AI-ModelScope/RNA-FM","iic/rnafm","damo/rnafm","multimolecule/RNA-FM"]:
    u = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=True"
    s, b = get(u)
    print(repo, "->", s, (b[:200].decode("utf-8","replace") if isinstance(b,bytes) else b))
