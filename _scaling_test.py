"""决定性测试：HF 吞吐随连接数缩放吗？
   涨   -> 受高 RTT 下的 BDP（窗口）限制 —— 多加连接能救
   不涨 -> 隧道带宽本身就这么点 —— 加连接没用，只能换路
"""
import subprocess, time, shutil, os

URL = "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors"
ARIA = shutil.which("aria2c")
TMP = os.environ.get("TEMP", ".")


def measure(conns, seconds=25):
    out = os.path.join(TMP, f"_scale_{conns}.part")
    ctl = out + ".aria2"
    for f in (out, ctl):
        if os.path.exists(f):
            os.remove(f)
    cmd = [ARIA, "-x", str(conns), "-s", str(conns), "-k", "1M",
           "--file-allocation=none", "--disable-ipv6=true",
           "--max-tries=1", "--timeout=30", "--lowest-speed-limit=0",
           "--summary-interval=0", "--console-log-level=error",
           "-d", TMP, "-o", os.path.basename(out), URL]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    time.sleep(seconds)
    p.kill()
    try:
        p.wait(timeout=10)
    except Exception:
        pass
    el = time.time() - t0
    # 真实落盘数据量（不是逻辑大小）
    got = 0
    if os.path.exists(out):
        with open(out, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                got += sum(1 for x in b if x)
    for f in (out, ctl):
        if os.path.exists(f):
            os.remove(f)
    return got / el / 1024.0  # KB/s


print("=" * 74)
print("HF 吞吐 vs 连接数（各测 25 秒）")
print("=" * 74)
results = []
for c in [1, 2, 4, 8, 16]:
    kbs = measure(c)
    results.append((c, kbs))
    print("  -x%-3d -> %7.1f KB/s" % (c, kbs), flush=True)

print()
base = results[0][1]
best = max(r[1] for r in results)
print("  单连接 %.1f KB/s   峰值 %.1f KB/s   倍数 %.1fx" % (base, best, best / max(base, 0.01)))
print()
if best / max(base, 0.01) > 2.5:
    print("  => 随连接数明显缩放 = 受 BDP/窗口限制。**加连接有用。**")
else:
    print("  => 基本不缩放 = 隧道带宽本身就这么大。**加连接没用，只能换路。**")
