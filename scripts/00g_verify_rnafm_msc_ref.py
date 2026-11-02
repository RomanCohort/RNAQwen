"""修正版交叉验证。

上一版错在哪：`RnaFmModel.from_pretrained(REF)` 因为键名带 `rnafm.` 前缀，
encoder 全部 MISSING（LOad REPORT 里列着了），于是测的是**随机初始化**的模型。
随机模型天然对序列敏感 —— 那个 YES 是假的。

正确做法：手动剥前缀、严格 load、**断言 missing 为空**，再测前向。
（PROGRESS §7.3c 记过同一个教训：load_state_dict 的返回值必须断言，不能丢。）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
REF = ROOT / "models" / "_rnafm_msc_ref"
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401

import torch
from safetensors.torch import load_file
from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

print("=" * 74)
print("1) 剥前缀 + 严格 load（关键：断言 missing）")
print("=" * 74)
raw = load_file(str(REF / "model.safetensors"))
norm = {}
for k, v in raw.items():
    for pfx in ("rnafm.", "model."):
        if k.startswith(pfx):
            k = k[len(pfx):]
            break
    norm[k] = v

m = RnaFmModel.from_pretrained(str(REF)).eval()
sd = m.state_dict()
print("  模型张量 %d / 文件(剥前缀后) %d / 交集 %d" % (len(sd), len(norm), len(set(sd) & set(norm))))
print("  模型独有（未覆盖）:", sorted(set(sd) - set(norm))[:10])
print("  文件独有（用不上）:", sorted(set(norm) - set(sd))[:10])

report = m.load_state_dict({k: v for k, v in norm.items() if k in sd}, strict=False)
missing = [k for k in report.missing_keys
           if "pooler" not in k and "layer_norm" not in k]
print(f"  load_state_dict missing={len(report.missing_keys)} "
      f"unexpected={len(report.unexpected_keys)}")
print(f"  剔除 pooler/layer_norm 后仍 missing: {missing if missing else '无'}")

if missing:
    print("\n[STOP] 还有键没装上，下面测了也不算数。")
    for k in report.missing_keys[:40]:
        print("   ", k)
    raise SystemExit(1)

print()
print("=" * 74)
print("2) 权重真装进去了吗（读模型，不是读文件）")
print("=" * 74)
w = m.state_dict()["embeddings.word_embeddings.weight"].float()
print("  embeddings.word_embeddings.weight %s  absmax=%.6e  nnz=%d/%d" % (
    tuple(w.shape), float(w.abs().max()), int((w.abs() > 1e-8).sum()), w.numel()))
n_zero = sum(1 for k in m.state_dict()
             if m.state_dict()[k].numel() and float(m.state_dict()[k].float().abs().max()) == 0.0)
print("  模型里全零张量数:", n_zero)

print()
print("=" * 74)
print("3) 前向：序列敏感性（这次权重真的在里面）")
print("=" * 74)
tok = RnaTokenizer.from_pretrained(str(REF))
seqs = ["ACGUACGUACGUACGUACGU", "GGGGGGGGGGGGGGGGGGGG", "UUUUUUUUUUUUUUUUUUUU"]
enc = tok(seqs, return_tensors="pt", padding=True)
print("  ids:", enc["input_ids"].tolist()[0])
with torch.no_grad():
    hs = m(**enc).last_hidden_state
d1 = float((hs[0] - hs[1]).abs().max())
d2 = float((hs[0] - hs[2]).abs().max())
print("  输出", tuple(hs.shape))
print("  ACGU vs GGGG  max|Δ| = %.6e" % d1)
print("  ACGU vs UUUU  max|Δ| = %.6e" % d2)

# 对照：同一个位置、同一条序列，必须逐位相同
with torch.no_grad():
    hs2 = m(**tok([seqs[0], seqs[0]], return_tensors="pt", padding=True)).last_hidden_state
print("  同序列两次 max|Δ| = %.6e (应为 0)" % float((hs2[0] - hs2[1]).abs().max()))

print()
print("=" * 74)
print("4) 结论")
print("=" * 74)
ok = (not missing) and float(w.abs().max()) > 1e-6 and d1 > 1e-3 and d2 > 1e-3
print("""
  HF 那份 multimolecule/rnafm（我们手上的）:
      word_embeddings absmax = 0.000000e+00, nnz = 0/17920
      205 张量里 155 个全零
      不同序列输出 max|Δ| = 0.000000e+00
      -> 坏的

  ModelScope 那份 ZhejiangLab-LifeScience/rnafm:
      word_embeddings absmax = %.6e, nnz = %d/%d
      213 张量里 %d 个全零
      不同序列输出 max|Δ| = %.6e / %.6e
      -> %s
""" % (float(w.abs().max()), int((w.abs() > 1e-8).sum()), w.numel(), n_zero, d1, d2,
       "健康的 ✅ —— 上游 RNA-FM 本来就不是全零，坐实我们那份文件坏了"
       if ok else "还是不健康，继续查"))
raise SystemExit(0 if ok else 1)
