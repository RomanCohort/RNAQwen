"""找 GitHub 上的 ERNIE-RNA 权重。"""
import json, urllib.request, urllib.parse

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"}


def gj(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


print("=" * 74)
print("1) GitHub 搜索 ERNIE-RNA 仓库")
print("=" * 74)
for q in ["ERNIE-RNA", "ERNIE-RNA+language+model", "rna+foundation+model+ERNIE"]:
    try:
        d = gj(f"https://api.github.com/search/repositories?q={q}&per_page=10")
        print(f"  [{q}] {d.get('total_count')} 个")
        for it in d.get("items", [])[:8]:
            print("    %-46s ★%-5s %s" % (it["full_name"], it["stargazers_count"],
                                          (it.get("description") or "")[:60]))
    except Exception as e:
        print(f"  [{q}] FAIL {type(e).__name__}: {str(e)[:70]}")

print()
print("=" * 74)
print("2) 逐个仓库看 releases / 大文件")
print("=" * 74)
REPOS = ["Bruce-Wang/ERNIE-RNA", "WeijieYin/ERNIE-RNA", "ml4bio/RNA-FM",
         "Bruce-Wang/RNA-FM", "yikunpku/ERNIE-RNA"]

for repo in REPOS:
    print(f"\n--- {repo} ---")
    try:
        info = gj(f"https://api.github.com/repos/{repo}")
        print(f"    存在  默认分支={info['default_branch']} 大小={info['size']} KB  ★{info['stargazers_count']}")
    except Exception as e:
        print(f"    [--] {type(e).__name__}: {str(e)[:60]}")
        continue

    # releases
    try:
        rels = gj(f"https://api.github.com/repos/{repo}/releases")
        print(f"    releases: {len(rels)}")
        for r in rels[:5]:
            print(f"      tag={r['tag_name']}")
            for a in r.get("assets", []):
                print("        asset %-44s %12d B" % (a["name"], a["size"]))
                print("              %s" % a["browser_download_url"])
    except Exception as e:
        print(f"    releases 查询失败: {str(e)[:60]}")

    # 仓库根目录的大文件
    try:
        items = gj(f"https://api.github.com/repos/{repo}/contents/")
        big = [(i["name"], i.get("size", 0)) for i in items
               if i.get("size", 0) > 1_000_000 or i["name"].endswith((".pth", ".pt", ".bin", ".safetensors", ".ckpt"))]
        print(f"    根目录里的大/权重文件: {big if big else '无'}")
        # 看看有没有 pretrained / weights 目录
        dirs = [i["name"] for i in items if i["type"] == "dir"]
        print(f"    目录: {dirs}")
    except Exception as e:
        print(f"    内容查询失败: {str(e)[:60]}")
