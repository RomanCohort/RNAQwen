"""自动重试跑一个命令，直到退出码为 0 或达到上限。

为什么不用 PowerShell 包一层
---------------------------
试过了，**嵌套引号在 cmd → powershell → python 三层之间会被吃掉**，导致内层命令
每次都以退出码 1 立刻失败（日志里能看到 attempt 2/30、3/30 都是秒退）。
Python 的 `subprocess` 收的是**参数列表**，不经过 shell 解析，没有这个坑。

用法：
    python scripts/_retry_runner.py --tries 30 --sleep 20 --log logs/x.log -- <命令> [参数...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=30)
    ap.add_argument("--sleep", type=int, default=20)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        print("[retry] 没给命令")
        return 2

    log = args.log
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        line = f"[retry] {msg}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(line, flush=True)
        if log:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    note("cmd = " + " ".join(cmd))
    for i in range(1, args.tries + 1):
        note(f"attempt {i}/{args.tries}")
        try:
            rc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1])).returncode
        except Exception as exc:  # noqa: BLE001
            note(f"spawn failed: {type(exc).__name__}: {exc}")
            rc = -1
        note(f"exit code {rc}")
        if rc == 0:
            note("completed successfully")
            return 0
        time.sleep(args.sleep)
    note("gave up after max tries")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
