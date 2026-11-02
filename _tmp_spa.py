import urllib.request, urllib.error, re
def get(url, hdr=None):
    h={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36","Accept":"*/*"}
    if hdr: h.update(hdr)
    req=urllib.request.Request(url, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=40) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:200]
    except Exception as e: return "ERR", str(e).encode()
# find the SPA's api endpoint
s,b = get("https://www.modelscope.cn/models?name=rnafm")
if isinstance(b,bytes):
    t=b.decode("utf-8","replace")
    print("endpoints:", sorted(set(re.findall(r'/api/v1/[A-Za-z0-9_/\-\.]+', t)))[:20])
    js = sorted(set(re.findall(r'src="([^"]+\.js)"', t)))
    print("js bundles:", js[:6])
    for u in js[:3]:
        if u.startswith("/"): u = "https://www.modelscope.cn" + u
        s2,b2 = get(u)
        if isinstance(b2,bytes) and s2==200:
            t2 = b2.decode("utf-8","replace")
            eps = sorted(set(re.findall(r'/api/v1/[A-Za-z0-9_/\-\.\{\}\$]{2,60}', t2)))
            hits = [e for e in eps if "model" in e.lower() or "dolphin" in e.lower() or "search" in e.lower()]
            print(" ", u[-40:], "->", hits[:15])
