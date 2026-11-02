"""看 Bruce-ywj/ERNIE-RNA 的 checkpoint/ 目录，以及另外两条下载的进度。"""
import json, time, urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
REPO = "Bruce-ywj/ERNIE-RNA"


def fetch(url, tries=4, timeout=25):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(2 + i * 2)
    print(f"    [失败 {type(last).__name__}: {str(last)[:60]}]")
    return None


def walk(path=""):
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    txt = fetch(url)
    if not txt:
        return
    for i in json.loads(txt):
        sz = i.get("size", 0)
        print("  %-6s %-58s %10s" % (i["type"], i["path"], sz if i["type"] == "file" else ""))
        if i["type"] == "file" and sz < 20000 and i["name"].lower().endswith((".md", ".txt", ".json")):
            body = fetch(f"https://raw.githubusercontent.com/{REPO}/main/{i['path']}")
            if body:
                lines = [l.strip() for l in body.splitlines() if l.strip()]
                print("        └─ 内容:", " | ".join(lines[:6])[:300])
        if i["type"] == "dir" and i["path"].count("/") < 2:
            walk(i["path"])


print("=" * 74)
print("checkpoint/ 目录树")
print("=" * 74)
walk("checkpoint")

print()
print("=" * 74)
print("README 里 model folder 那段完整原文")
print("=" * 74)
md = fetch(f"https://raw.githubusercontent.com/{REPO}/main/README.md")
if md:
    lines = md.splitlines()
    for n, l in enumerate(lines):
        if "two subfolders" in l or "model folder" in l:
            for k in range(max(0, n - 12), min(len(lines), n + 10)):
                print("   ", lines[k][:170])

print()
print("=" * 74)
print("另外两条下载的真实进度")
print("=" * 74)
for tag, p in [("hf.co        ", r"models\ernierna\model.safetensors.part"),
               ("hf-mirror.com", r"models\ernierna_mirror\model.safetensors.part")]:
    P = Path(r"C:\qwen-api\RNAQwen") / p
    if not P.exists():
        print(f"  {tag}  .part 不存在")
        continue
    have = 0
    with open(P, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            if any(b):
                have += 1
    print("  %s  真实 %3d MB / 329 MB  (%.1f%%)" % (tag, have, 100.0 * have / 329))
