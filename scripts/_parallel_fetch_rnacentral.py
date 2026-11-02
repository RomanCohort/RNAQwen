"""多连接 Range 分片下载 rnacentral.1024（单连接被 HF 限速到 ~200-470 KB/s）。

为什么要它
----------
`_redownload_rnacentral.py` 是**单连接**顺序追加，实测 ~470 KB/s，3.7 GB 要 2 小时。
实测 HF 的限速是**按连接**的：24 并发能到 ~2.9 MB/s（约 6 倍）。
所以这里把剩余区间切成固定大小的分片，每片一个 `.part` 文件、独立可续传，
全部下完后按 offset 顺序追加到主文件，最后校验 sha256。

设计要点
--------
* 分片网格固定在首次启动时的 `base`（= 当时主文件已有的字节数），存进 `meta.json`；
  重跑时前缀可能变长，已完全落进主文件的分片会跳过。
* 每片单独可续传：`.part` 文件已有多少就从哪里接着要。
* 只有在**全部**分片完整后才会碰主文件 —— 不会把半成品拼进去。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import requests

P = Path(r"C:\qwen-api\RNAQwen\data\external\rnacentral_1024.parquet")
PARTS = Path(r"C:\qwen-api\RNAQwen\data\external\_rnacentral_parts")
URL = "https://huggingface.co/datasets/multimolecule/rnacentral.1024/resolve/main/data.parquet"
TARGET = 3_695_330_389
WANT = "cf1f389a2accda43ee3d118c5e9f485be584c7511df12bca1d7b9fe5816ba7b4"

_lock = threading.Lock()
_done_bytes = 0
_t0 = 0.0


def human(n: float) -> str:
    return f"{n/1024/1024:.1f} MB"


def fetch_chunk(start: int, end: int, retries: int = 40) -> tuple[int, int]:
    """下载 [start, end] 闭区间到独立 .part，返回 (bytes, attempts)。"""
    path = PARTS / f"{start:012d}.part"
    want = end - start + 1
    if path.exists() and path.stat().st_size == want:
        return want, 0
    have = path.stat().st_size if path.exists() else 0
    last = ""
    for attempt in range(retries):
        try:
            r = requests.get(URL, headers={"Range": f"bytes={start + have}-{end}"},
                             stream=True, timeout=(20, 30))
            if r.status_code != 206:
                raise RuntimeError(f"status={r.status_code}（期望 206）")
            with path.open("ab") as fh:
                for c in r.iter_content(512 * 1024):
                    fh.write(c)
                    have += len(c)
            if have >= want:
                return have, attempt + 1
            last = f"短读 {have}/{want}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {str(exc)[:90]}"
        time.sleep(min(1.0 + attempt * 0.4, 8.0))
        have = path.stat().st_size if path.exists() else 0
    raise RuntimeError(f"分片 {start} 重试 {retries} 次仍失败：{last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--assemble", action="store_true", default=True)
    ap.add_argument("--no-assemble", dest="assemble", action="store_false")
    args = ap.parse_args()

    PARTS.mkdir(parents=True, exist_ok=True)
    meta_path = PARTS / "meta.json"
    chunk = args.chunk_mb * 1024 * 1024
    cur = P.stat().st_size if P.exists() else 0

    if cur >= TARGET:
        print(f"[dl] 主文件已是 {cur} 字节，跳过下载")
    else:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            base = int(meta["base"])
        else:
            base = cur
            meta_path.write_text(json.dumps({"base": base, "target": TARGET, "chunk": chunk}))
        print(f"[dl] 主文件前缀 {cur} 字节；分片网格 base={base}；目标 {TARGET}；"
              f"待下 {human(TARGET-base)}；并发 {args.workers}；片大小 {args.chunk_mb} MB", flush=True)

        if cur != base:
            print(f"[dl] 警告：当前前缀 {cur} != 网格 base {base}，只拼接 end>=cur 的分片")

        chunks = [(s, min(s + chunk - 1, TARGET - 1)) for s in range(base, TARGET, chunk)]
        todo = [(s, e) for s, e in chunks if e >= cur]
        print(f"[dl] 共 {len(chunks)} 片，本次处理 {len(todo)} 片", flush=True)

        global _done_bytes, _t0
        _done_bytes = sum((PARTS / f"{s:012d}.part").stat().st_size
                          for s, _ in chunks if (PARTS / f"{s:012d}.part").exists())
        _t0 = time.time()
        n_ok = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_chunk, s, e): (s, e) for s, e in todo}
            for fut in cf.as_completed(futs):
                s, e = futs[fut]
                try:
                    got, att = fut.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[dl] 分片 {s} 失败：{exc}", flush=True)
                    continue
                n_ok += 1
                with _lock:
                    _done_bytes += e - s + 1
                    el = max(time.time() - _t0, 1e-6)
                    rate = _done_bytes / el
                    if n_ok % 10 == 0 or n_ok == len(todo):
                        eta_min = (TARGET - cur - _done_bytes) / max(rate, 1) / 60
                        print(f"[dl] {n_ok}/{len(todo)} 片  {human(_done_bytes)}  "
                              f"{rate/1024/1024:.2f} MB/s  ETA {eta_min:.1f} min", flush=True)

        got_all = sum((PARTS / f"{s:012d}.part").stat().st_size
                      for s, _ in chunks if (PARTS / f"{s:012d}.part").exists())
        want_all = TARGET - base
        print(f"[dl] 分片合计 {got_all} / {want_all}", flush=True)
        if got_all != want_all:
            print("[dl] 分片不完整，停止（不拼主文件）")
            return 1

    # ---- 拼接 ----
    if P.stat().st_size < TARGET and args.assemble:
        chunk = json.loads(meta_path.read_text())["chunk"] if meta_path.exists() else chunk
        base = json.loads(meta_path.read_text())["base"] if meta_path.exists() else 0
        cur = P.stat().st_size
        print(f"[asm] 追加 [{cur}, {TARGET}) 到主文件 ...", flush=True)
        with P.open("ab") as out:
            for s in range(base, TARGET, chunk):
                e = min(s + chunk - 1, TARGET - 1)
                if e < cur:
                    continue
                part = PARTS / f"{s:012d}.part"
                if not part.exists():
                    print(f"[asm] 缺分片 {part} → 中止")
                    return 1
                with part.open("rb") as fh:
                    shutil.copyfileobj(fh, out, 4 * 1024 * 1024)

    size = P.stat().st_size
    print(f"[chk] 大小 {size} / {TARGET}  {'OK' if size == TARGET else '不一致'}", flush=True)
    h = hashlib.sha256()
    t0 = time.time()
    with P.open("rb") as fh:
        while True:
            b = fh.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    got = h.hexdigest()
    print(f"[chk] sha256 {got}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"[chk] 期望   {WANT}", flush=True)
    print(f"[chk] 大小匹配={size == TARGET}  sha256匹配={got == WANT}", flush=True)
    return 0 if (size == TARGET and got == WANT) else 1


if __name__ == "__main__":
    sys.exit(main())
