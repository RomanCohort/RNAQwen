"""网络体检：到底哪一段慢。"""
import json
import socket
import subprocess
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}
MB = 1024 * 1024


def timed_read(url, seconds=8.0, headers=None, label=""):
    h = dict(UA)
    if headers:
        h.update(headers)
    try:
        t0 = time.time()
        got = 0
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=25) as r:
            while time.time() - t0 < seconds:
                b = r.read(256 * 1024)
                if not b:
                    break
                got += len(b)
        el = max(time.time() - t0, 0.01)
        print("  %-40s %8.2f MB/s   (%6.2f MB / %.1fs)" % (label, got / el / 1e6, got / 1e6, el))
        return got / el
    except Exception as e:
        print("  %-40s FAIL %s: %s" % (label, type(e).__name__, str(e)[:60]))
        return 0.0


def latency(host, port=443, tries=3):
    best = None
    for _ in range(tries):
        try:
            t0 = time.time()
            s = socket.create_connection((host, port), timeout=8)
            dt = (time.time() - t0) * 1000
            s.close()
            best = dt if best is None else min(best, dt)
        except Exception:
            pass
    return best


print("=" * 74)
print("1) TCP 连接延迟（不是 ping，Google/CDN 常吞 ICMP）")
print("=" * 74)
for host, label in [("47.79.124.117", "隧道出口 47.79.124.117"),
                    ("www.modelscope.cn", "ModelScope（国内）"),
                    ("huggingface.co", "huggingface.co")]:
    ms = latency(host)
    print("  %-32s %s" % (label, ("%.0f ms" % ms) if ms else "连接失败"))

print()
print("=" * 74)
print("2) 下载速率对比")
print("=" * 74)
# 国内基线
timed_read("https://modelscope.cn/models/ZhejiangLab-LifeScience/rnafm/resolve/master/config.json",
           label="ModelScope 小文件（国内基线）")
timed_read("https://modelscope.cn/models/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/master/config.json",
           label="ModelScope +Range 8MB",
           headers={"Range": "bytes=0-8388607"})
# HF
timed_read("https://huggingface.co/multimolecule/rnafm/resolve/main/config.json",
           label="HF 小文件（走 TUN）")
timed_read("https://huggingface.co/multimolecule/ernierna/resolve/main/model.safetensors",
           label="HF +Range 8MB（走 TUN）",
           headers={"Range": "bytes=0-8388607"})
timed_read("https://hf-mirror.com/multimolecule/ernierna/resolve/main/model.safetensors",
           label="hf-mirror +Range 8MB",
           headers={"Range": "bytes=0-8388607"})

print()
print("=" * 74)
print("3) 本机链路")
print("=" * 74)
try:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-NetAdapter | Where-Object Status -eq 'Up' | "
         "Select-Object Name,LinkSpeed,MediaType | Format-Table -AutoSize | Out-String"],
        capture_output=True, text=True, timeout=30).stdout
    print(out.strip()[:800])
except Exception as e:
    print("  查询失败:", e)

print("=" * 74)
print("4) 谁在吃 CPU（mihomo 加密开销）")
print("=" * 74)
try:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 "
         "Name,Id,@{n='CPU_s';e={[math]::Round($_.CPU,0)}},@{n='MemMB';e={[math]::Round($_.WorkingSet64/1MB,0)}} | "
         "Format-Table -AutoSize | Out-String"],
        capture_output=True, text=True, timeout=30).stdout
    print(out.strip()[:900])
except Exception as e:
    print("  查询失败:", e)

print("=" * 74)
print("5) 当前两条下载的合计速率（三次采样，各 5 秒）")
print("=" * 74)
try:
    for i in range(3):
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$a=(Get-NetAdapterStatistics | Measure-Object -Property ReceivedBytes -Sum).Sum; "
             "Start-Sleep -Seconds 5; "
             "$b=(Get-NetAdapterStatistics | Measure-Object -Property ReceivedBytes -Sum).Sum; "
             "Write-Output (($b-$a)/5)"],
            capture_output=True, text=True, timeout=40).stdout.strip()
        print("  采样 %d: %s B/s = %.1f KB/s" % (i + 1, out, float(out) / 1024 if out else 0))
except Exception as e:
    print("  失败:", e)
