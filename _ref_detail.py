"""1) 那 2 个全零张量到底是哪两个（必须说清楚，不能含糊）
   2) ModelScope 副本 vs HF 副本的 vocab / 结构差异清单
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
REF = ROOT / "models" / "_rnafm_msc_ref"
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401

import torch
from safetensors.torch import load_file
from multimolecule.models.rnafm import RnaFmModel

raw = load_file(str(REF / "model.safetensors"))
norm = {}
for k, v in raw.items():
    for pfx in ("rnafm.", "model."):
        if k.startswith(pfx):
            k = k[len(pfx):]
            break
    norm[k] = v

m = RnaFmModel.from_pretrained(str(REF)).eval()
sd0 = m.state_dict()
m.load_state_dict({k: v for k, v in norm.items() if k in sd0}, strict=False)

print("=== 模型里全零的张量（含来源）===")
sd = m.state_dict()
for k in sorted(sd):
    t = sd[k]
    if t.numel() and float(t.float().abs().max()) == 0.0:
        src = "来自文件" if k in norm else "**新初始化（文件里没有）**"
        print("   %-42s %-26s %s" % (k, tuple(t.shape), src))

print()
print("=== 没装上的键（模型侧）===")
for k in sorted(set(sd) - set(norm)):
    t = sd[k]
    print("   %-42s %-26s 当前 absmax=%.4e" % (k, tuple(t.shape), float(t.float().abs().max())))

print()
print("=== 文件里有、模型用不上的键 ===")
for k in sorted(set(norm) - set(sd)):
    print("   %-52s %s" % (k, tuple(norm[k].shape)))

print()
print("=== vocab 对比 ===")
wf = sd["embeddings.word_embeddings.weight"]
print("   multimolecule(RnaFmModel)  vocab = %d" % wf.shape[0])
print("   ModelScope 文件            vocab = %d" % norm["embeddings.word_embeddings.weight"].shape[0])
try:
    hf = load_file(str(ROOT / "models" / "rnafm" / "model.safetensors"))
    print("   HF 文件(multimolecule/rnafm) vocab = %d" % hf["model.embeddings.word_embeddings.weight"].shape[0])
except Exception as e:
    print("   HF 文件读取失败:", str(e)[:80])

print()
print("=== 两份的 vocab.txt 行数 ===")
for tag, p in [("ModelScope", REF / "vocab.txt"), ("HF", ROOT / "models" / "rnafm" / "vocab.txt")]:
    if p.exists():
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        print("   %-12s %d 行: %s" % (tag, len(lines), lines))
    else:
        print("   %-12s 缺失" % tag)
