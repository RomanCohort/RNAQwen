"""Bruce-ywj/ERNIE-RNA 的权重在哪 —— 加重试，这条链路抖。"""
import json, time, urllib.request, re

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}
REPO = "Bruce-ywj/ERNIE-RNA"


def fetch(url, tries=5, timeout=25, binary=False):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
            return b if binary else b.decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(2 + i * 2)
    print(f"    [重试 {tries} 次都失败] {type(last).__name__}: {str(last)[:70]}")
    return None


print("=" * 74)
print(f"1) {REPO}")
print("=" * 74)
info = fetch(f"https://api.github.com/repos/{REPO}")
if info:
    d = json.loads(info)
    print(f"  默认分支 {d['default_branch']}  大小 {d['size']} KB  ★{d['stargazers_count']}")
    print(f"  描述: {d.get('description')}")
    branch = d["default_branch"]
else:
    branch = "main"

print()
print("=" * 74)
print("2) 根目录")
print("=" * 74)
items = fetch(f"https://api.github.com/repos/{REPO}/contents/")
if items:
    for i in json.loads(items):
        sz = i.get("size", 0)
        mark = "  <== 大文件" if sz > 1_000_000 else ""
        print("  %-8s %-40s %12s%s" % (i["type"], i["name"], sz, mark))

print()
print("=" * 74)
print("3) releases / 资产")
print("=" * 74)
rels = fetch(f"https://api.github.com/repos/{REPO}/releases")
if rels:
    rs = json.loads(rels)
    print(f"  {len(rs)} 个 release")
    for r in rs:
        print(f"    tag={r['tag_name']}  {r.get('name')}")
        for a in r.get("assets", []):
            print("      %-46s %14d B" % (a["name"], a["size"]))
            print("        %s" % a["browser_download_url"])
if not rels or not json.loads(rels):
    print("  （没有 release）")

print()
print("=" * 74)
print("4) README 里的下载线索（Google Drive / Zenodo / 直链）")
print("=" * 74)
for name in ["README.md", "readme.md", "README.MD"]:
    md = fetch(f"https://raw.githubusercontent.com/{REPO}/{branch}/{name}")
    if md:
        print(f"  --- {name} ({len(md)} 字符) ---")
        # 找所有 URL
        urls = sorted(set(re.findall(r'https?://[^\s\)\]"\'<>]+', md)))
        weightish = [u for u in urls if any(k in u.lower() for k in
                     ("drive.google", "zenodo", "dropbox", "1drv", "figshare",
                      "download", ".pth", ".pt", ".bin", ".safetensors", "releases", "oss", "pan."))]
        print("  权重/下载相关 URL:")
        for u in weightish or ["（没找到显式下载链接）"]:
            print("     ", u)
        # 找 checkpoint 关键词
        print("  含 checkpoint/weight 的行:")
        for line in md.splitlines():
            if re.search(r'checkpoint|weight|pretrain|download|模型|权重', line, re.I):
                print("     ", line.strip()[:150])
        break

print()
print("=" * 74)
print("5) 试试常见权重路径")
print("=" * 74)
for p in ["ERNIE-RNA_pretrained.pth", "pretrained/ERNIE-RNA.pth",
          "checkpoints/ERNIE-RNA.pth", "weights/ERNIE-RNA.pth",
          "ERNIE-RNA.pth", "model.pth"]:
    url = f"https://raw.githubusercontent.com/{REPO}/{branch}/{p}"
    try:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as r:
            print("  [OK] %-38s %s" % (p, r.headers.get("Content-Length")))
    except Exception as e:
        print("  [--] %-38s %s" % (p, type(e).__name__))
