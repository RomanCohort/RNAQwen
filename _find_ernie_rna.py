"""找 ERNIE-RNA：HF 搜索 + 候选 repo 结构探测。"""
import json, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def get_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get_text(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


print("=" * 74)
print("HF 搜索")
print("=" * 74)
seen = {}
for q in ["ERNIE-RNA", "ernie-rna", "ERNIE_RNA", "ernierna"]:
    for api in [f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&limit=50",
                f"https://huggingface.co/api/models?search={urllib.parse.quote(q)}&limit=50&full=true"]:
        try:
            d = get_json(api)
            for m in d:
                seen[m["modelId"]] = m
            print(f"  [{q}] {len(d)} 个")
            break
        except Exception as e:
            print(f"  [{q}] FAIL {type(e).__name__}: {str(e)[:70]}")

print()
print("=== 命中列表（按下载量）===")
for mid, m in sorted(seen.items(), key=lambda kv: -(kv[1].get("downloads") or 0)):
    print("  %-52s dl=%-8s likes=%-5s tags=%s" % (
        mid, m.get("downloads"), m.get("likes"),
        [t for t in (m.get("tags") or []) if t not in ("transformers", "pytorch", "safetensors")][:6]))

print()
print("=" * 74)
print("逐个看 config.json / 文件清单")
print("=" * 74)


def inspect(mid):
    out = {"id": mid}
    for rev in ["main", "master"]:
        try:
            cfg = json.loads(get_text(f"https://huggingface.co/{mid}/resolve/{rev}/config.json"))
            out["config"] = cfg
            break
        except Exception as e:
            out["config_err"] = str(e)[:60]
    try:
        d = get_json(f"https://huggingface.co/api/models/{mid}?blobs=true")
        out["files"] = [(s["rfilename"], (s.get("lfs") or {}).get("size") or s.get("size"))
                        for s in d.get("siblings", [])]
    except Exception as e:
        out["files_err"] = str(e)[:60]
    return out


with ThreadPoolExecutor(max_workers=6) as ex:
    for r in ex.map(inspect, list(seen)[:14]):
        print(f"\n--- {r['id']} ---")
        c = r.get("config")
        if c:
            keys = ["model_type", "architectures", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "intermediate_size", "vocab_size",
                    "max_position_embeddings"]
            for k in keys:
                if k in c:
                    print("   %-26s %s" % (k, c[k]))
            print("   config 全部键:", sorted(c.keys()))
        else:
            print("   config:", r.get("config_err"))
        fs = r.get("files") or []
        if fs:
            print("   文件:")
            for n, s in fs[:20]:
                print("     %-40s %s" % (n, s))
        else:
            print("   files:", r.get("files_err"))
