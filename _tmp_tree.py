import json, urllib.request, urllib.error
def get(url):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:300]
    except Exception as e: return "ERR", str(e).encode()
s,b = get("https://huggingface.co/api/models/multimolecule/rnafm/tree/main?recursive=true")
print("tree:", s)
if isinstance(b,bytes) and s==200:
    for f in json.loads(b.decode()):
        lfs = f.get("lfs") or {}
        print(f"  {f['path']:24s} size={f.get('size')} sha256={lfs.get('oid','-')}")
