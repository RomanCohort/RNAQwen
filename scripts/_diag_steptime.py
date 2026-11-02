"""一次性调查：逐步耗时的分布、与序列长度的关系、epoch 间漂移。"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "reports" / "_raw_timing_loss.json").read_text(encoding="utf-8-sig"))
r = d["runs"][0]
m = r["batch_meta"]
tot = [x["step_total_s"] for x in m]
fwd = [x["step_fwd_s"] for x in m]
bwd = [x["step_bwd_s"] for x in m]
n = len(tot)
q = lambda xs, p: sorted(xs)[min(int(p * len(xs)), len(xs) - 1)]

print(f"n = {n}")
print(f"total  min={min(tot):.3f} p25={q(tot,.25):.3f} median={st.median(tot):.3f} "
      f"mean={sum(tot)/n:.3f} p75={q(tot,.75):.3f} p90={q(tot,.9):.3f} max={max(tot):.3f}")
print(f"fwd    median={st.median(fwd):.3f} mean={sum(fwd)/n:.3f}  ({100*sum(fwd)/sum(tot):.0f}% of total)")
print(f"bwd    median={st.median(bwd):.3f} mean={sum(bwd)/n:.3f}  ({100*sum(bwd)/sum(tot):.0f}% of total)")
print(f"median(fwd)+median(bwd) = {st.median(fwd)+st.median(bwd):.3f} vs "
      f"median(fwd+bwd) = {st.median([a+b for a,b in zip(fwd,bwd)]):.3f}")
print()
print("per-step (first 14):  nt    fwd    bwd   total")
for x in m[:14]:
    print(f"                    {x['seq_nt']:>4}  {x['step_fwd_s']:.3f}  "
          f"{x['step_bwd_s']:.3f}  {x['step_total_s']:.3f}")
print()
# 与长度的关系
xs = [x["seq_nt"] for x in m]
ys = tot
mx, my = sum(xs) / n, sum(ys) / n
sxx = sum((a - mx) ** 2 for a in xs)
sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
b = sxy / sxx
a = my - b * mx
r_p = sxy / ((sxx ** 0.5) * (sum((v - my) ** 2 for v in ys) ** 0.5))
print(f"回归 total_s ~ seq_nt:  intercept={a:.3f}s  slope={1000*b:.2f} ms/nt  r={r_p:.3f}")
# 只用短序列（<500nt）看噪声底
short = [y for x, y in zip(xs, ys) if x < 500]
print(f"短序列(<500nt, n={len(short)}) total_s: median={st.median(short):.3f} mean={sum(short)/len(short):.3f} "
      f"min={min(short):.3f} max={max(short):.3f}")
mid = [y for x, y in zip(xs, ys) if 500 <= x < 700]
print(f"中序列(500-700nt, n={len(mid)}) total_s: median={st.median(mid):.3f} mean={sum(mid)/len(mid):.3f}")
long_ = [y for x, y in zip(xs, ys) if x >= 700]
print(f"长序列(>=700nt, n={len(long_)}) total_s: median={st.median(long_):.3f} mean={sum(long_)/len(long_):.3f}")
print()
e1 = [x["step_total_s"] for x in m[:50]]
e2 = [x["step_total_s"] for x in m[50:]]
print(f"epoch1 mean={sum(e1)/50:.3f} median={st.median(e1):.3f}")
print(f"epoch2 mean={sum(e2)/50:.3f} median={st.median(e2):.3f}")
dd = [bb - aa for aa, bb in zip(e1, e2)]
print(f"paired (epoch2-epoch1): mean={sum(dd)/50:+.3f} median={st.median(dd):+.3f} "
      f"(负 = epoch2 更快)")
print(f"总墙钟 = {r['seconds']['wall_total']:.1f}s for {n} 步 -> 均值核对 "
      f"{r['seconds']['wall_total']/n:.3f} s/step")
