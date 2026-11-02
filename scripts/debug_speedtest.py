"""测几种下载通道的实际速度，选一条能用的。

背景：直连 huggingface.co 的 CDN 实测 **0.10 MB/s** ——
Qwen3-8B（16.4 GB）要下 45 小时，不可用。
"""

from __future__ import annotations

import concurrent.futures as cf
import time
import warnings
from pathlib import Path

import requests

warnings.filterwarnings("ignore")

CHANNELS = {
    "hf_direct": "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors",
    "hf_mirror": "https://hf-mirror.com/multimolecule/rnafm/resolve/main/model.safetensors",
    "modelscope_qwen3_8b": "https://modelscope.cn/models/Qwen/Qwen3-8B/resolve/master/model-00001-of-00005.safetensors",
    "modelscope_files": "https://www.modelscope.cn/api/v1/models/Qwen/Qwen3-8B/repo/files?Revision=master",
    "hf_proxy_7890": "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors",
}


def speed(url: str, seconds: float = 8.0, proxies: dict | None = None) -> tuple[str, float, str]:
    t0 = time.time()
    n = 0
    host = "-"
    try:
        r = requests.get(url, stream=True, timeout=(10, 30), proxies=proxies, headers={"User-Agent": "rnaqwen-speedtest/0.1"})
        host = r.url.split("/")[2]
        if r.status_code != 200:
            return f"HTTP {r.status_code}", 0.0, host
        for chunk in r.iter_content(1024 * 512):
            n += len(chunk)
            if time.time() - t0 > seconds:
                break
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {str(exc)[:60]}", 0.0, host
    dt = max(time.time() - t0, 1e-6)
    return "OK", n / dt / 1e6, host


def main() -> None:
    print(f"{'channel':24s} {'MB/s':>8s}  {'status':12s} host")
    for name, url in CHANNELS.items():
        if name == "hf_proxy_7890":
            continue
        status, mbps, host = speed(url)
        print(f"{name:24s} {mbps:8.2f}  {status:12s} {host}", flush=True)

    # 本机有没有 Clash 之类的本地代理端口
    for port in (7890, 7897, 10808, 10809, 1080, 20171):
        proxies = {"http": f"http://127.0.0.1:{port}", "https": f"http://127.0.0.1:{port}"}
        status, mbps, host = speed(CHANNELS["hf_direct"], seconds=5, proxies=proxies)
        print(f"proxy:{port:<16d} {mbps:8.2f}  {status:12s} {host}", flush=True)

    # 并发连接能不能叠加带宽
    def one(_i: int) -> float:
        _s, m, _h = speed(CHANNELS["hf_direct"], seconds=8)
        return m

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        speeds = list(ex.map(one, range(6)))
    print(f"\n6 并发 hf_direct: 单流均值={sum(speeds)/len(speeds):.2f} MB/s  合计={sum(speeds):.2f} MB/s")


if __name__ == "__main__":
    main()
