"""量 commit：谁在吃「提交量（commit）」这张额度。

2026-10-14：76_ 装载在 layer 11 报 `OSError 1455 页面文件太小` 而死。
这台机器的 commit 额度 = 物理 63.6 GB + 固定页面文件 32 GB ≈ 95.6 GB，
所以「谁占了 commit」是可以量的，不用猜。

依次量：
  1. 基线
  2. mmap 68 GB 的副缓存（read-only numpy memmap）
  3. 再 torch.from_numpy 它
  4. 逐个 safe_open 35B 的 26 个分片（**只开句柄，不读张量**）
  5. 真读一个张量
"""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
AUX = ROOT / "data" / "processed" / "_rfa_full_hidden.f16"


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


def perf() -> tuple[float, float, float]:
    m = _PERF()
    m.cb = ctypes.sizeof(m)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(m), m.cb):
        raise OSError("GetPerformanceInfo 失败")
    gb = m.PageSize / 2 ** 30
    return m.CommitTotal * gb, m.CommitLimit * gb, m.PhysicalAvailable * gb


def report(tag: str) -> None:
    used, limit, phys = perf()
    print(f"  {tag:<34} commit 已用 {used:>6.1f} / {limit:>6.1f} GB"
          f"   可用 {limit - used:>6.1f} GB   物理空闲 {phys:>5.1f} GB", flush=True)


def commit_of() -> float:
    """只取「提交量已用」这一个数，供别的探针复用。"""
    used, _, _ = perf()
    return used


def main() -> int:
    print(f"[probe] 副缓存 {AUX}  {AUX.stat().st_size / 2**30:.1f} GB"
          if AUX.exists() else "[probe] 副缓存不存在")
    n_shard = len(list(MDL.glob("model-*.safetensors")))
    print(f"[probe] 35B 分片 {n_shard} 个，合计 "
          f"{sum(p.stat().st_size for p in MDL.glob('model-*.safetensors')) / 2**30:.1f} GB")
    report("0 基线")

    import numpy as np

    n_pos, dim = 47547438, 768
    mm = np.memmap(AUX, dtype=np.float16, mode="r", shape=(n_pos, dim))
    report("1 mmap 68 GB 副缓存（只看一页）")

    import torch

    t = torch.from_numpy(mm)
    report("2 torch.from_numpy 之后")
    del t

    from safetensors import safe_open

    handles = []
    for i, p in enumerate(sorted(MDL.glob("model-*.safetensors"))):
        handles.append(safe_open(str(p), framework="pt", device="cpu"))
        if i % 5 == 4 or i == n_shard - 1:
            report(f"3 开了 {i + 1}/{n_shard} 个分片句柄")

    keys = list(handles[0].keys())
    report("4 列 key（读 header 完成）")
    k0 = keys[0]
    x = handles[0].get_tensor(k0)
    report(f"5 读了一个张量 {k0} {tuple(x.shape)} {x.dtype}")
    del x
    handles.clear()
    report("6 关掉所有句柄")
    print("[probe] 完", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
