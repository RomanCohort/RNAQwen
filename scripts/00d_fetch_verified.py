"""通用「带哈希闸门」的模型取数。

为什么不用 `hf_hub_download` / `snapshot_download`：
  2026-10-09 事故 —— `models/rnafm/model.safetensors` 大小 398,117,588 与上游**逐字节
  相同**，但 205 个张量里 155 个全零（aria2 预分配 + 中断留下的零填充空洞），
  `from_pretrained` 一声不响。**唯一能挡住它的是 sha256。**

规矩（写死在这里，别在调用点上重复实现）：
  1) 下到 `.part`；
  2) **强制** sha256 比对上游声明值，不符绝不改名；
  3) 大小已满但哈希错 => 判定为「预分配空洞」，连 .part.aria2 一起清掉
     （这种情况续传救不了：控制文件会说"分片都完成了"，每次续传立刻返回同一个坏文件）；
  4) 通过后才原子改名。

跑法：
    python scripts/00d_fetch_verified.py \
        --repo multimolecule/ernierna --dest models/ernierna \
        --sha256 02274cb7... --size 345071800
不给 --sha256 时**拒绝大文件下载**（防止又走回"下来就用"的老路）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0"}

# 大文件之外要一起拉的小文件（缺了模型加载不起来）
SMALL = ["config.json", "tokenizer_config.json", "vocab.txt",
         "special_tokens_map.json", "configuration.json"]


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def real_progress(path: Path, chunk: int = 1 << 20) -> tuple[int, int, int]:
    """数「有多少 MB 块真有数据」。

    ⚠️ **逻辑大小不能当进度。** aria2 多连接按偏移写文件，逻辑大小会跳到最高已写位置，
    中间可能是零。实测一份 325 MB 的 .part 里 255/311 块是空洞，真实数据只有 56 MB。
    这个函数才是进度。

    返回 (有数据块数, 总块数, 空洞位置前若干)。
    """
    have = tot = 0
    holes: list[int] = []
    with open(path, "rb") as f:
        off = 0
        while True:
            b = f.read(chunk)
            if not b:
                break
            tot += 1
            if any(b):
                have += 1
            elif len(holes) < 20:
                holes.append(off // chunk)
            off += len(b)
    return have, tot, len(holes)


def hf_declared(repo: str, fname: str, host: str = "huggingface.co") -> tuple[str | None, int | None]:
    """问 HF 要官方声明的 sha256 / size。"""
    url = f"https://{host}/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.load(r)
    for s in d.get("siblings", []):
        if s["rfilename"] == fname:
            lfs = s.get("lfs") or {}
            return lfs.get("sha256"), lfs.get("size")
    return None, None


def fetch_small(repo: str, fname: str, dest: Path, rev: str,
                host: str = "huggingface.co") -> bool:
    url = f"https://{host}/{repo}/resolve/{rev}/{fname}"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        (dest / fname).write_bytes(data)
        print(f"  {fname:<28} {len(data):>8} B  OK")
        return True
    except Exception as e:
        print(f"  {fname:<28} 跳过 ({type(e).__name__}: {str(e)[:50]})")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="例如 multimolecule/ernierna")
    ap.add_argument("--file", default="model.safetensors")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--sha256", default=None, help="上游声明的 sha256；不给就自己去问 HF")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--rev", default="main")
    ap.add_argument("--host", default="huggingface.co",
                    help="下载主机。hf-mirror.com 是镜像；**哈希用的是同一份声明值**，"
                         "所以从哪个 host 下都不影响正确性，只影响速度。")
    ap.add_argument("--api-host", default="huggingface.co",
                    help="查元数据用哪个 host（声明 sha256 从这里拿）")
    ap.add_argument("--connections", type=int, default=8,
                    help="aria2 连接数。16 在这条链路上握手失败率太高，默认降到 8")
    ap.add_argument("--method", default="aria2", choices=["aria2", "curl"],
                    help="aria2=多连接续传（快，靠控制文件记进度）；"
                         "curl=单连接续传（慢，但写出来的一定是连续前缀）")
    ap.add_argument("--max-attempts", type=int, default=20,
                    help="失败重试次数。**重试不会删 .part**，所以进度不丢")
    ap.add_argument("--no-verify", action="store_true",
                    help="**危险**：跳过 sha256。只有元数据接口拿不到哈希时才用，且要写进 PROGRESS")
    args = ap.parse_args()

    dest: Path = args.dest if args.dest.is_absolute() else (ROOT / args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    big = dest / args.file
    part = dest / (args.file + ".part")
    url = f"https://{args.host}/{args.repo}/resolve/{args.rev}/{args.file}"

    # ---- 单写者锁 ----
    # 2026-10-09 真踩了：杀 aria2 **没杀它的父 python**，那个 python 的重试循环
    # 立刻又拉起一个 aria2；与此同时我又起了第二个驱动器。
    # 结果两个 aria2 写同一个 .part / .part.aria2。
    # （数据本身没事 —— 两边写的是同样的字节、同样的偏移；但控制文件会互相踩，
    #   进度读数变得不可信。）这个锁就是为这件事加的。
    lock = dest / ".fetch.lock"
    if lock.exists():
        try:
            old = int(lock.read_text(encoding="utf-8").split()[0])
        except Exception:
            old = -1
        alive = False
        if old > 0:
            alive = subprocess.run(["tasklist", "/FI", f"PID eq {old}"],
                                   capture_output=True, text=True).stdout.find(str(old)) >= 0
        if alive:
            print(f"[STOP] {dest} 已经被 PID {old} 的驱动器占着（单写者锁）。")
            print("       两个 aria2 写同一个 .part 会把控制文件搞乱。")
            print("       要么等它结束，要么先杀掉**父 python 进程**再重来。")
            return 3
        print(f"[note] 发现过期锁（PID {old} 已不在），接管")
    lock.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}", encoding="utf-8")

    want_sha, want_size = args.sha256, args.size
    if want_sha is None:
        print(f"[meta] 问 {args.api_host} 要 {args.repo} 的声明值 ...")
        try:
            want_sha, api_size = hf_declared(args.repo, args.file, args.api_host)
            want_size = want_size or api_size
        except Exception as e:
            print(f"[meta] 失败: {e}")

    print("=" * 74)
    print(f"repo   : {args.repo}")
    print(f"file   : {args.file}")
    print(f"host   : {args.host}   (元数据来自 {args.api_host})")
    print(f"dest   : {dest}")
    print(f"期望   : size={want_size}  sha256={want_sha}")
    print("=" * 74)

    if not want_sha and not args.no_verify:
        print("[STOP] 拿不到上游 sha256，拒绝下载大文件。")
        print("       真要跳过请显式加 --no-verify，并把原因写进 PROGRESS.md。")
        return 2

    # 已经好了？
    if big.exists():
        h = sha256_of(big)
        if want_sha and h == want_sha:
            print(f"[skip] 已存在且哈希正确 ({h[:16]}...)")
            for f in SMALL:
                if not (dest / f).exists():
                    fetch_small(args.repo, f, dest, args.rev, args.host)
            return 0
        print(f"[note] 已存在的文件哈希不符 ({h[:16]}... != {want_sha[:16] if want_sha else '?'}...)")
        bad = dest / (args.file + ".corrupt")
        if bad.exists():
            bad.unlink()
        big.rename(bad)
        print(f"[keep] 留档 -> {bad.name}")

    # 下
    exe = shutil.which("aria2c")
    print(f"[down] {url}")

    if args.method == "aria2":
        if not exe:
            print("[STOP] 没有 aria2c。用 --method curl 或先装 aria2。")
            return 2
        # ⚠️ 这里必须循环重试，而且**不能删 .part / .part.aria2**。
        #
        # 为什么（2026-10-09 实测）：
        #   aria2 多连接是按偏移写文件的，`--file-allocation=none` 下逻辑大小会跳到
        #   "最高已写位置"，中间全是零。实测一份 .part：
        #       逻辑大小 325,391,316 B（看起来 94%），311 个 MB 块里 **255 个是空洞**，
        #       真实数据只有 56 MB（16%）。
        #   所以**逻辑大小完全不能当作进度**。
        #   而 `.part.aria2` 控制文件准确记录了哪些分片已完成 —— 它才是进度的真相。
        #   一次失败就删文件，等于把真进度也扔了（上一版就是这么干的）。
        #
        # 正确性最后由 sha256 兜底：只要它没通过，就绝不改名。
        for attempt in range(1, args.max_attempts + 1):
            cmd = [exe, "-x", str(args.connections), "-s", str(args.connections),
                   "-k", "1M",
                   "--continue=true",                 # 用控制文件续传
                   "--file-allocation=none",          # 不预分配
                   "--disable-ipv6=true",
                   "--max-tries=0", "--retry-wait=10", "--timeout=60",
                   "--lowest-speed-limit=1K",         # 死连接快点丢掉换新的
                   "--summary-interval=60", "--console-log-level=warn",
                   "-d", str(dest), "-o", part.name, url]
            print(f"\n[aria2] 第 {attempt}/{args.max_attempts} 次 "
                  f"（.part 逻辑大小 {part.stat().st_size if part.exists() else 0:,} B "
                  f"—— 注意这不是进度）", flush=True)
            print("[aria2] " + " ".join(cmd), flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd)
            el = time.time() - t0
            size_now = part.stat().st_size if part.exists() else 0
            print(f"[aria2] rc={rc} 用时 {el/60:.1f} min  逻辑大小 {size_now:,} B")
            if part.exists():
                have, tot, nh = real_progress(part)
                print(f"[真进度] 有数据块 {have}/{tot}  (约 {have} MB / "
                      f"{args.size//(1<<20) if args.size else '?'} MB)，空洞 {nh}")
            if rc == 0:
                print("[aria2] 退出码 0")
                break
            print(f"[aria2] 非零退出 —— **保留 .part 与控制文件**，重试会接着下")
            time.sleep(5)
        else:
            print(f"[warn] {args.max_attempts} 次都没成功。.part 还在，再跑本命令即可。")
    else:
        # 单连接 curl 续传。慢，但写出来的一定是**连续前缀**，没有空洞歧义。
        if not shutil.which("curl"):
            print("[STOP] 没有 curl。")
            return 2
        for attempt in range(1, args.max_attempts + 1):
            print(f"\n[curl] 第 {attempt}/{args.max_attempts} 次续传 "
                  f"（已有 {part.stat().st_size if part.exists() else 0:,} B）", flush=True)
            rc = subprocess.call(["curl", "-sS", "-L", "-C", "-",
                                  "--max-time", "1800",
                                  "--speed-limit", "1024", "--speed-time", "60",
                                  "-A", "curl/8.0", "-o", str(part), url])
            print(f"[curl] rc={rc}  "
                  f"大小 {part.stat().st_size if part.exists() else 0:,} B")
            if rc == 0:
                break
            time.sleep(5)

    if not part.exists():
        print("[FAIL] 没有 .part，下载没起来")
        return 1

    # 校验
    size = part.stat().st_size
    print()
    print(f"[verify] 大小 {size:,} (期望 {want_size}) "
          f"{'OK' if want_size and size == want_size else '不符' if want_size else '未声明'}")
    got = sha256_of(part)
    print(f"[verify] sha256 {got}")
    if want_sha and got != want_sha:
        print("[FAIL] 哈希不符 —— 绝不改名")
        if want_size and size >= want_size:
            ctl = part.with_name(part.name + ".aria2")
            print("[FAIL] 大小已满但内容错 -> 判定为预分配空洞，清掉重来")
            part.unlink(missing_ok=True)
            if ctl.exists():
                ctl.unlink()
            print("       重跑本命令即可（会从头下）")
        else:
            print(f"[FAIL] 只下了 {size/1e6:.1f} MB，未完成 -> 保留 .part，重跑会接着下")
        return 1

    print("[verify] 哈希一致")
    if big.exists():
        big.unlink()
    part.rename(big)          # 同盘 rename 原子
    print(f"[done] -> {big}")

    print()
    print("[meta] 小文件")
    for f in SMALL:
        if not (dest / f).exists():
            fetch_small(args.repo, f, dest, args.rev, args.host)

    print()
    print("下一步（**必须**）：")
    print(f"  & $PY scripts\\01c_check_encoder_sanity.py --model {dest} \\")
    print(f"      --expect-sha256 {want_sha} --expect-size {want_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
