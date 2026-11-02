"""后台挂机下载 RNAcentral 数据（断点续传，绝不因中断丢失进度）。

为什么单独写一个：
  实测 EBI FTP ~12 KB/s，2.66 GB 要几十小时。这种长任务必须：
    1) --continue=true 断点续传
    2) 循环重试，**绝不删 .part**
    3) 最后用官方 md5 校验（EBI 提供 md5/md5.tsv.gz）

目标：
  data/external/id_mapping.tsv.gz          RefSeq/Ensembl/... -> URS 映射  (2.66 GB)
  data/external/rnacentral_active.fasta.gz 全部活跃序列
  data/external/md5.tsv.gz                 官方校验和（小，先拿）

再往外走一步（描述字段）需要 database_files/ 的 PostgreSQL 转储，那个要 pg_restore，
先不碰 —— 把能直接用的拿下来。
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
DEST = ROOT / "data" / "external"
BASE = "https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release"
UA = {"User-Agent": "Mozilla/5.0"}

TARGETS = [
    ("md5/md5.tsv.gz", 2 << 20),
    ("id_mapping/id_mapping.tsv.gz", 2660438847),
    ("sequences/rnacentral_active.fasta.gz", None),
]


def md5_of(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch_small(rel: str) -> Path | None:
    url = f"{BASE}/{rel}"
    p = DEST / Path(rel).name
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            p.write_bytes(r.read())
        print(f"  [ok] {rel} -> {p.name} ({p.stat().st_size} B)", flush=True)
        return p
    except Exception as e:
        print(f"  [--] {rel} 失败 {type(e).__name__}: {str(e)[:60]}", flush=True)
        return None


def fetch_big(rel: str, expect: int | None, attempts: int = 2000) -> bool:
    """aria2c 循环重试续传。返回是否最终拿到完整大小。"""
    url = f"{BASE}/{rel}"
    name = Path(rel).name
    part = DEST / (name + ".part")
    final = DEST / name
    exe = shutil.which("aria2c")
    if not exe:
        print("  [STOP] 没有 aria2c")
        return False

    for i in range(1, attempts + 1):
        if final.exists() and expect and final.stat().st_size == expect:
            print(f"  [done] {name} 已完成", flush=True)
            return True
        cmd = [exe, "-x", "16", "-s", "16", "-k", "1M",
               "--continue=true", "--file-allocation=none",
               "--disable-ipv6=true", "--max-tries=0", "--retry-wait=20",
               "--timeout=60", "--lowest-speed-limit=1K",
               "--summary-interval=300", "--console-log-level=warn",
               "-d", str(DEST), "-o", part.name, url]
        if i == 1:
            print(f"[aria2] {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        sz = part.stat().st_size if part.exists() else 0
        pct = f"{100.0*sz/expect:.1f}%" if expect else f"{sz/1e6:.0f}MB"
        print(f"[aria2] 第 {i} 次 rc={rc}  {(time.time()-t0)/60:.1f}min  "
              f"{sz:,} B ({pct})", flush=True)
        if rc == 0 and part.exists():
            part.rename(final)
            print(f"  [done] {name} -> {final}", flush=True)
            return True
        if expect and part.exists() and part.stat().st_size >= expect:
            part.rename(final)
            print(f"  [done] {name}（大小已满）", flush=True)
            return True
        time.sleep(10)
    return False


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print(f"RNAcentral 后台下载 -> {DEST}")
    print("=" * 80, flush=True)

    md5file = fetch_small("md5/md5.tsv.gz")
    declared = {}
    if md5file:
        import gzip
        try:
            with gzip.open(md5file, "rt", errors="replace") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        declared[parts[-1].lstrip("./")] = parts[0]
            print(f"  读到 {len(declared)} 条官方 md5", flush=True)
        except Exception as e:
            print(f"  解析 md5 失败: {e}", flush=True)

    for rel, expect in TARGETS:
        if rel.startswith("md5/"):
            continue
        print(f"\n=== {rel}  (期望 {expect} B) ===", flush=True)
        ok = fetch_big(rel, expect)
        name = Path(rel).name
        p = DEST / name
        if p.exists() and name in declared:
            got = md5_of(p)
            print(f"  [verify] md5 {got}  期望 {declared[name]}  "
                  f"{'一致 ✅' if got == declared[name] else '**不一致**'}", flush=True)
    print("\n[all done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
