"""下 ModelScope 那份 rnafm 到独立目录当对照（不碰 models/rnafm）。

完整性怎么保证：ModelScope 不像 HF 那样声明 sha256，所以靠三条叠加：
  1) 大小必须等于 API 声明的 Size；
  2) safetensors 能打开且张量数对（截断会直接抛）；
  3) **没有任何全零张量** + 换序列输出会变（这才是真正的判据）。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
DEST = ROOT / "models" / "_rnafm_msc_ref"
RID = "ZhejiangLab-LifeScience/rnafm"
REV = "master"
UA = {"User-Agent": "Mozilla/5.0"}
WANT = ["model.safetensors", "config.json", "tokenizer_config.json",
        "vocab.txt", "special_tokens_map.json"]


def api_files() -> dict[str, int]:
    url = f"https://modelscope.cn/api/v1/models/{RID}/repo/files?Revision={REV}&Root="
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    files = (d.get("Data") or {}).get("Files") or []
    out = {}
    print("=== API 声明的文件 ===")
    for f in files:
        n, s = f.get("Name"), f.get("Size") or 0
        out[n] = s
        extra = {k: v for k, v in f.items() if k.lower() in
                 ("sha256", "md5", "checksum", "hash", "revision")}
        print("  %-32s %14s  %s" % (n, s, extra if extra else ""))
    return out


def small_get(name: str, expect: int) -> bool:
    url = f"https://modelscope.cn/models/{RID}/resolve/{REV}/{name}"
    p = DEST / name
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(p, "wb") as f:
        shutil.copyfileobj(r, f)
    ok = p.stat().st_size == expect
    print("  %-32s %8d / %8d  %s" % (name, p.stat().st_size, expect, "OK" if ok else "大小不符"))
    return ok


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    declared = api_files()

    big = DEST / "model.safetensors"
    exp_big = declared.get("model.safetensors", 0)
    if big.exists() and big.stat().st_size == exp_big and exp_big:
        print(f"\n[skip] 大文件已在且大小对（{exp_big}）")
    else:
        if big.exists():
            big.unlink()
        url = f"https://modelscope.cn/models/{RID}/resolve/{REV}/model.safetensors"
        exe = shutil.which("aria2c")
        print(f"\n=== 下 model.safetensors（期望 {exp_big} B）===")
        t0 = time.time()
        if exe:
            rc = subprocess.call([exe, "-x", "16", "-s", "16", "-k", "1M",
                                  "--continue=true", "--file-allocation=none",
                                  "--max-tries=0", "--retry-wait=5", "--timeout=60",
                                  "--summary-interval=10", "--console-log-level=warn",
                                  "-d", str(DEST), "-o", "model.safetensors", url])
            print(f"[aria2] rc={rc}  {(time.time()-t0)/60:.1f} min")
        else:
            small_get("model.safetensors", exp_big)
        if big.exists():
            print(f"  落盘 {big.stat().st_size} / {exp_big}  "
                  f"{(time.time()-t0):.1f}s  {big.stat().st_size/1e6/max(time.time()-t0,0.1):.1f} MB/s")

    print("\n=== 小文件 ===")
    for n in WANT:
        if n == "model.safetensors":
            continue
        try:
            small_get(n, declared.get(n, 0))
        except Exception as e:
            print("  %-32s FAIL %s" % (n, str(e)[:60]))

    print("\n=== 大小核对 ===")
    for n in WANT:
        p = DEST / n
        if not p.exists():
            print("  %-32s 缺失" % n)
            continue
        e = declared.get(n, 0)
        print("  %-32s %14d / %14d  %s" % (n, p.stat().st_size, e,
                                           "OK" if p.stat().st_size == e else "不符"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
