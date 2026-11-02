"""`safe_open(...).get_tensor()` 给的是映射的视图，还是复制出来的？

这决定「装载器能不能随手关分片句柄」：
  是视图  -> 关掉映射会让已读到的张量变成悬空指针，必须先 clone 再关
  是复制  -> 直接关就安全

判据用三个互相独立的量：
  1. 同一个 key 取两次的 data_ptr 是否相同（缓存的视图会相同）
  2. untyped_storage().nbytes()：视图会是整个映射的大小，复制就是张量自身大小
  3. 取张量前后 commit 的增量：复制会涨张量大小，视图不涨
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")


class _PERF(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("CommitTotal", ctypes.c_size_t),
        ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t),
        ("PhysicalTotal", ctypes.c_size_t),
        ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t),
        ("KernelTotal", ctypes.c_size_t),
        ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t),
        ("PageSize", ctypes.c_size_t),
        ("HandleCount", ctypes.c_ulong),
        ("ProcessCount", ctypes.c_ulong),
        ("ThreadCount", ctypes.c_ulong),
    ]


def commit_gb() -> float:
    m = _PERF()
    m.cb = ctypes.sizeof(m)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(m), m.cb)
    return m.CommitTotal * m.PageSize / 2 ** 30


def main() -> int:
    from safetensors import safe_open

    p = sorted(MDL.glob("model-*.safetensors"))[0]
    print(f"[probe] {p.name}  {p.stat().st_size / 2 ** 30:.2f} GB")
    c0 = commit_gb()
    h = safe_open(str(p), framework="pt", device="cpu")
    print(f"  open           commit {c0:6.1f} -> {commit_gb():6.1f} GB")

    keys = list(h.keys())
    # 挑一个不太小的张量，让"复制"看得见
    cand = []
    for k in keys:
        s = h.get_slice(k).get_shape()
        n = 1
        for d in s:
            n *= d
        cand.append((n, k, s))
    cand.sort(reverse=True)
    n, k, s = cand[0]
    print(f"  最大张量 {k}  {s}  {n * 2 / 2 ** 30:.2f} GB(bf16)")

    c1 = commit_gb()
    x = h.get_tensor(k)
    c2 = commit_gb()
    y = h.get_tensor(k)
    c3 = commit_gb()

    print(f"  get_tensor#1   commit {c1:6.1f} -> {c2:6.1f} GB  "
          f"({c2 - c1:+.2f})")
    print(f"  get_tensor#2   commit {c2:6.1f} -> {c3:6.1f} GB  "
          f"({c3 - c2:+.2f})")
    print(f"  data_ptr   #1 {x.data_ptr():#x}   #2 {y.data_ptr():#x}   "
          f"相同={x.data_ptr() == y.data_ptr()}")
    print(f"  storage   #1 {x.untyped_storage().nbytes() / 2 ** 20:.1f} MiB  "
          f"= 张量自身 {x.numel() * x.element_size() / 2 ** 20:.1f} MiB ? "
          f"{x.untyped_storage().nbytes() == x.numel() * x.element_size()}")
    print(f"  dtype {x.dtype}  shape {tuple(x.shape)}  "
          f"前三个值 {x.flatten()[:3].tolist()}")
    del x, y
    print("[probe] 完")
    return 0


if __name__ == "__main__":
    sys.exit(main())
