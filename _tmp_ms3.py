import json, urllib.request, urllib.error, urllib.parse
def get(url):
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)","Accept":"text/html,application/json,*/*"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()[:500]
    except Exception as e: return "ERR", f"{type(e).__name__}: {e}".encode()

for u in ["https://www.modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&Name=rnafm&SortBy=Default",
          "https://www.modelscope.cn/api/v1/models?PageSize=20&PageNumber=1&Keyword=rnafm",
          "https://www.modelscope.cn/api/v1/dolphin/models/search?Name=rnafm",
          "https://modelscope.cn/api/v1/models/AI-ModelScope/RNA-FM/repo/files?Revision=master&Recursive=True"]:
    s,b=get(u); print(s, u)
    if isinstance(b,bytes): print("   ", b[:250].decode("utf-8","replace"))
