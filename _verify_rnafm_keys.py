"""先看清楚两边到底叫什么名字，别在错误的键集合上下结论。"""
import sys, glob, os
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
import torch
from multimolecule.models.rnafm import RnaFmModel

MP = str(ROOT / "models" / "rnafm")

print("=== 目录里的文件 ===")
for f in sorted(os.listdir(MP)):
    p = os.path.join(MP, f)
    print("  %-34s %12d" % (f, os.path.getsize(p) if os.path.isfile(p) else -1))

from safetensors.torch import load_file
raw = {}
for f in glob.glob(os.path.join(MP, "*.safetensors")):
    part = load_file(f)
    print("  %s -> %d tensors" % (os.path.basename(f), len(part)))
    raw.update(part)

print("\n=== raw 前 12 个键 ===")
for k in sorted(raw)[:12]:
    print("   ", k, tuple(raw[k].shape))

print("\n=== raw 里含 embeddings 的键 ===")
for k in sorted(raw):
    if "embed" in k.lower():
        print("   ", k, tuple(raw[k].shape), "std=%.4f" % float(raw[k].float().std()))

m = RnaFmModel.from_pretrained(MP).eval()
sd = m.state_dict()
print("\n=== 模型前 12 个键 ===")
for k in sorted(sd)[:12]:
    print("   ", k, tuple(sd[k].shape))

print("\n=== 模型里含 embeddings 的键 ===")
for k in sorted(sd):
    if "embed" in k.lower():
        print("   ", k, tuple(sd[k].shape), "std=%.4f" % float(sd[k].float().std()))

print("\n=== 键集合对比 ===")
print("  raw 独有 :", len(set(raw) - set(sd)))
for k in sorted(set(raw) - set(sd))[:12]:
    print("     -", k)
print("  model 独有:", len(set(sd) - set(raw)))
for k in sorted(set(sd) - set(raw))[:12]:
    print("     +", k)
print("  共有     :", len(set(raw) & set(sd)))
