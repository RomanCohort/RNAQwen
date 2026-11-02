"""查 ModelScope / HF 上 rnafm 的可下载源，挑最快的。"""
import urllib.request, json

CANDIDATES = [
    ("modelscope", "multimolecule/rnafm"),
    ("modelscope", "AI-ModelScope/rnafm"),
    ("modelscope", "ml4bio/RNA-FM"),
    ("hf", "multimolecule/rnafm"),
]

print("=== ModelScope ===")
for kind, rid in CANDIDATES:
    if kind != "modelscope":
        continue
    for url in [f"https://modelscope.cn/api/v1/models/{rid}/repo/files?Revision=master&Root=",
                f"https://www.modelscope.cn/api/v1/models/{rid}"]:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            print(f"  [OK] {rid}  <- {url}")
            print("       ", json.dumps(d, ensure_ascii=False)[:600])
            break
        except Exception as e:
            print(f"  [--] {rid}  {type(e).__name__}: {str(e)[:90]}")

print("\n=== HF ===")
for kind, rid in CANDIDATES:
    if kind != "hf":
        continue
    try:
        with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{rid}?blobs=true", timeout=30) as r:
            d = json.load(r)
        for s in d.get("siblings", []):
            if "safetensors" in s.get("rfilename", ""):
                print("  ", s["rfilename"], s.get("lfs", {}))
    except Exception as e:
        print("  HF 失败:", e)
