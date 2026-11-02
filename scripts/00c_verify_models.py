"""校验本地权重文件的 sha256 是否等于 ModelScope 声明的值。

为什么需要这一步：Qwen3-8B 的 5 个 shard 是**分多次、断点续传**下完的
（中途还碰到过磁盘写满、连接卡死重连）。续传用的是 HTTP Range ——
只要服务端两次返回的字节流不一致（换 CDN 节点 / 换 revision），
拼起来的文件就是**静默损坏**的：大小对得上、safetensors 也能加载、
但权重是错的，模型只会吐乱码。

这正是本机遇到的情况（397/399 个张量 load 成功，输出 "React matplotlib sys" 乱码）。

用法：
    python scripts/00c_verify_models.py --repo Qwen/Qwen3-8B --dest models/Qwen3-8B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files"
CHUNK = 8 * 1024 * 1024


def list_files(repo: str, revision: str = "master") -> list[dict]:
    url = f"{MS_API.format(repo=repo)}?Revision={urllib.parse.quote(revision)}&Recursive=True"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "rnaqwen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode())
    files = (payload.get("Data") or {}).get("Files") or []
    out = []
    for f in files:
        if f.get("Type") == "tree" or f.get("Size") is None:
            continue
        out.append({"path": f["Path"], "size": int(f["Size"]),
                    "sha256": (f.get("Sha256") or f.get("Sha256Hash") or "").lower()})
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--revision", default="master")
    ap.add_argument("--include", default=None, help="只校验匹配的子串（逗号分隔）")
    args = ap.parse_args()

    remote = list_files(args.repo, args.revision)
    if args.include:
        pats = [p.strip() for p in args.include.split(",") if p.strip()]
        remote = [f for f in remote if any(p in f["path"] for p in pats)]

    bad: list[str] = []
    missing: list[str] = []
    print(f"{'file':50s} {'local':>12s} {'remote':>12s}  sha256")
    for f in remote:
        local = args.dest / f["path"]
        if not local.exists():
            missing.append(f["path"])
            print(f"{f['path']:50s} {'-':>12s} {f['size']:>12d}  MISSING")
            continue
        size = local.stat().st_size
        ok_size = size == f["size"]
        digest = sha256_of(local)
        ok_hash = (digest == f["sha256"]) if f["sha256"] else None
        flag = "OK" if (ok_size and ok_hash is not False) else "BAD"
        if not ok_size or ok_hash is False:
            bad.append(f["path"])
        print(f"{f['path']:50s} {size:>12d} {f['size']:>12d}  {flag} {digest[:16]} vs {f['sha256'][:16]}")

    print()
    if bad:
        print(f"[FAIL] {len(bad)} 个文件校验不通过：")
        for b in bad:
            print(f"   - {b}")
        print("       -> 删掉这些文件重新下载（**不要**续传），再跑一次校验。")
        return 1
    if missing:
        print(f"[PARTIAL] {len(missing)} 个文件缺失：{missing}")
        return 1
    print("[OK] 全部文件与 ModelScope 声明一致。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
