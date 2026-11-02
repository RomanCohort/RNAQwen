import urllib.request, urllib.error, re
def get(url):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:300]
    except Exception as e: return "ERR", f"{type(e).__name__}: {e}".encode()
s,b = get("https://www.modelscope.cn/models?name=rnafm")
print("search page:", s, len(b) if isinstance(b,bytes) else b)
if isinstance(b,bytes):
    t=b.decode("utf-8","replace")
    hits=set(re.findall(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]*[Rr][Nn][Aa][-]?[Ff][Mm][A-Za-z0-9_.-]*', t))
    print("  repo-like hits:", list(hits)[:20])
    print("  'rnafm' count:", t.lower().count("rnafm"), " 'RNA-FM' count:", t.count("RNA-FM"))
