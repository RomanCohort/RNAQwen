import urllib.request, json, sys

HF = "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors"

print("=== HEAD resolve URL ===")
try:
    req = urllib.request.Request(HF, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        for k, v in r.headers.items():
            kl = k.lower()
            if kl in ("x-linked-etag", "x-linked-size", "etag", "content-length", "x-repo-commit"):
                print("  %-18s %s" % (k, v))
except Exception as e:
    print("  HEAD 失败:", e)

print("\n=== API 带 blobs ===")
for url in ["https://huggingface.co/api/models/multimolecule/rnafm?blobs=true",
            "https://huggingface.co/api/models/multimolecule/rnafm"]:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        print(" ", url)
        for s in d.get("siblings", []):
            if "safetensors" in s.get("rfilename", ""):
                print("   ", json.dumps(s, ensure_ascii=False))
        break
    except Exception as e:
        print("  ", url, "失败:", e)

print("\n=== 本地 ===")
print("  size   398117588")
print("  sha256 38f3ae9aadbf4d3a5ff6873476fe9d16600b83a54711637ec3760e8e83a2408d")
