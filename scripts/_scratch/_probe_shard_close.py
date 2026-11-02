"""关掉分片句柄，commit 还回来吗？

装载器要改成「用多少开多少」的前提是：丢掉句柄引用之后，Windows 的 commit
真的会释放。如果 safetensors 内部有缓存，改动就白做。

顺便看一眼句柄对象有哪些方法（有没有显式 close）。
"""
from __future__ import annotations

import ctypes
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "_scratch"))
from _probe_commit import _PERF, MDL, commit_of  # noqa: E402


def commit_gb() -> float:
    return commit_of()


def main() -> int:
    from safetensors import safe_open

    shards = sorted(MDL.glob("model-*.safetensors"))[:5]
    c0 = commit_gb()
    print(f"  基线                       commit {c0:6.1f} GB")

    hs = [safe_open(str(p), framework="pt", device="cpu") for p in shards]
    c1 = commit_gb()
    print(f"  开 5 个分片（{sum(p.stat().st_size for p in shards) / 2 ** 30:.2f} GB）"
          f"  commit {c1:6.1f} GB  ({c1 - c0:+.1f})")

    print(f"  句柄对象的方法: {[m for m in dir(hs[0]) if not m.startswith('__')]}")

    del hs
    gc.collect()
    c2 = commit_gb()
    print(f"  del + gc 之后              commit {c2:6.1f} GB  ({c2 - c1:+.1f})")
    if c2 - c0 < 1.0:
        print("[probe] 结论：commit 完全还回来了 -> 改「用多少开多少」可行")
    else:
        print(f"[probe] 结论：还差 {c2 - c0:.1f} GB 没还 —— 需要别的办法")
    return 0


if __name__ == "__main__":
    sys.exit(main())
