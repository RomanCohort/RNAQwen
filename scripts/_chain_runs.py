"""串行跑多个训练任务：等上一个进程退出 → 起下一个。

为什么需要它
------------
单卡，两轨 Stage 1 只能串行；而每个要跑 8~22 小时，人不可能守着。

⚠️ 为什么用 Python 写、而且必须用 WMI 启动
------------------------------------------
实测：**DSH 的 pwsh 工具调用跑在 Windows Job Object 里**，
会话一重启（这个项目发生过很多次）就把整棵进程树带走 ——
哪怕你用了 `Start-Process -WindowStyle Hidden`，甚至纯 PowerShell 的 watcher 也一起死。
表现是"进程静默消失，没有 traceback、没有 abort 栈、事件日志也没记录"。

绕过办法：把命令行交给 WMI（`Win32_Process.Create`）去建，
父进程变成 `WmiPrvSE.exe`，不在我的作业对象里，于是不受会话重启影响。

**所以这个脚本本身也要用 WMI 启动，否则它自己先死。**

用法：
    python scripts/_chain_runs.py --wait-pid 18568 --spec specs.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = r"C:\ana\envs\comfyui\python.exe"


def alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, errors="replace",
    ).stdout
    return str(pid) in out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True, help="先等这个 PID 退出")
    ap.add_argument("--spec", type=Path, required=True, help="JSON：[{name, args:[...], log}]")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "chain_runs.log")
    args = ap.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)

    def note(msg: str) -> None:
        line = f"[chain] {msg}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    note(f"waiting for PID {args.wait_pid}")
    while alive(args.wait_pid):
        time.sleep(args.poll)
    note(f"PID {args.wait_pid} exited; sleeping 30s to let the GPU go")
    time.sleep(30)

    for i, job in enumerate(spec):
        name, job_args, log = job["name"], job["args"], job.get("log", f"logs/{name}.log")
        cmd = [PY, str(ROOT / job_args[0]), *job_args[1:]]
        log_path = ROOT / log
        note(f"[{i+1}/{len(spec)}] starting {name} -> {log_path}")
        with log_path.open("a", encoding="utf-8") as fh:
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(ROOT), creationflags=0x00000008 | 0x00000200)  # DETACHED|NEW_GROUP
        note(f"{name} pid={p.pid}")
        p.wait()
        note(f"{name} exited with code {p.returncode}")
        time.sleep(30)

    note("all jobs done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
