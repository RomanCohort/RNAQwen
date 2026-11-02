"""探测 HF 单连接 / 多连接 / 镜像 的下载吞吐，决定用哪种下载策略。"""
import sys
import time
import concurrent.futures as cf

import requests

PATH = "/datasets/multimolecule/rnacentral.1024/resolve/main/data.parquet"
BASES = ["https://huggingface.co", "https://hf-mirror.com"]
OFFSET_BASE = 2_000_000_000


def probe(base: str, n: int, dur: float = 10.0):
    url = base + PATH

    def grab(i):
        start = OFFSET_BASE + i * 30_000_000
        end = start + 2_000_000
        t0 = time.time()
        got = 0
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                             stream=True, timeout=(15, 20))
            if r.status_code != 206:
                return (r.status_code, 0.0)
            for c in r.iter_content(262144):
                got += len(c)
                if time.time() - t0 > dur:
                    break
        except Exception as exc:  # noqa: BLE001
            return ("ERR:" + type(exc).__name__, got / max(time.time() - t0, 1e-9))
        return (206, got / max(time.time() - t0, 1e-9))

    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(grab, range(n)))
    agg = sum(r[1] for r in res)
    codes = sorted({str(r[0]) for r in res})
    host = base.split("//")[1][:20]
    print(f"{host:22s} n={n:2d}  agg={agg/1024:8.0f} KB/s  codes={codes}", flush=True)
    return agg


if __name__ == "__main__":
    total = 0.0
    for base in BASES:
        for n in (1, 4, 8, 16, 24):
            total += probe(base, n)
    print("done", total)
    sys.exit(0)
