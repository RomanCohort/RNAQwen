import json, urllib.request, urllib.error
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:400]
    except Exception as e: return "ERR", f"{type(e).__name__}: {e}".encode()
s,b = get("https://huggingface.co/api/models/multimolecule/rnafm")
print("api:", s)
if isinstance(b,bytes):
    try:
        j=json.loads(b.decode())
        for f in j.get("siblings",[]): print("   ", f.get("rfilename"))
    except Exception: print(b[:400])
