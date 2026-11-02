"""量化损坏范围：多少个张量是全零的（aria2 预分配留下的空洞）。"""
import glob, os, sys
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
import torch
from safetensors.torch import load_file

MP = ROOT / "models" / "rnafm"
raw = {}
for f in glob.glob(str(MP / "*.safetensors")):
    raw.update(load_file(f))

zeros, ok, tiny = [], [], []
for k in sorted(raw):
    t = raw[k].float()
    if t.numel() == 0:
        continue
    if float(t.abs().max()) == 0.0:
        zeros.append(k)
    else:
        ok.append(k)

print("总张量 :", len(raw))
print("正常   :", len(ok))
print("全零   :", len(zeros))
print()
print("=== 全零的张量（应完全为空）===")
for k in zeros:
    print("   %-52s %s" % (k, tuple(raw[k].shape)))
if zeros:
    print()
    print("=== 全零张量的前缀分布 ===")
    from collections import Counter
    c = Counter(k.rsplit(".", 2)[0] if "." in k else k for k in zeros)
    for k, v in c.most_common(20):
        print("   %-50s %d" % (k, v))

# 关键：编码器主干有多少是零的
body = [k for k in zeros if "encoder." in k or "embeddings." in k]
print()
print("编码器主干里的全零张量 :", len(body))
print("  前 20 个:", body[:20])
