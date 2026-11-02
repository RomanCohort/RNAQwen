"""1) 剥掉 `model.` 前缀后逐位比对（上一条脚本的"共有键 0"只是前缀问题）。
   2) 查 word_embeddings 是不是真的全零 —— 若全零，模型对输入序列完全不敏感，
      那是比"权重坏了"严重得多的事,必须查清。
"""
import sys, glob, os
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
import torch
from safetensors.torch import load_file
from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

MP = str(ROOT / "models" / "rnafm")
raw = {}
for f in glob.glob(os.path.join(MP, "*.safetensors")):
    raw.update(load_file(f))

# --- 1) 剥前缀后逐位比对 ---
norm = {}
for k, v in raw.items():
    norm[k[len("model."):] if k.startswith("model.") else k] = v

m = RnaFmModel.from_pretrained(MP).eval()
sd = m.state_dict()
common = sorted(set(norm) & set(sd))
print("剥前缀后共有键 :", len(common), "/ 模型", len(sd), "/ 文件", len(norm))
maxd, bad = 0.0, []
for k in common:
    a, b = norm[k].float(), sd[k].float()
    if a.shape != b.shape:
        bad.append((k, tuple(a.shape), tuple(b.shape)))
        continue
    d = float((a - b).abs().max())
    maxd = max(maxd, d)
    if d != 0.0:
        bad.append((k, d))
print("逐位相同       :", "YES" if maxd == 0.0 else "NO  max|Δ|=%.3e" % maxd)
print("形状/数值不一致:", bad if bad else "无")
print("未覆盖的模型键 :", sorted(set(sd) - set(norm)))   # 应该只有 pooler

# --- 2) word_embeddings 到底是不是零 ---
w = sd["embeddings.word_embeddings.weight"].float()
print("\n=== word_embeddings.weight (28, 640) ===")
print("  absmax      :", f"{float(w.abs().max()):.6e}")
print("  mean        :", f"{float(w.mean()):.6e}")
print("  std         :", f"{float(w.std()):.6e}")
print("  nnz(>1e-8)  :", int((w.abs() > 1e-8).sum()), "/", w.numel())
print("  前两行前 8 列:\n", w[:2, :8])

# --- 3) 决定性检验：换序列，输出会不会变 ---
tok = RnaTokenizer.from_pretrained(MP)
seqs = ["ACGUACGUACGUACGUACGU",
        "GGGGGGGGGGGGGGGGGGGG",
        "ACGUACGUACGUACGUACGU"]
enc = tok(seqs, return_tensors="pt", padding=True)
with torch.no_grad():
    out = m(**enc).last_hidden_state
d = float((out[0] - out[1]).abs().max())
s = float((out[0] - out[2]).abs().max())
print("\n=== 序列敏感性 ===")
print("  不同序列 max|Δ| :", f"{d:.6e}", "(必须 > 0)")
print("  相同序列 max|Δ| :", f"{s:.6e}", "(应为 0)")

# --- 4) tokenizer 是否把 A/C/G/U 映到不同 id ---
ids = tok(["ACGU", "UUUU", "GGGG"], return_tensors="pt")["input_ids"]
print("\n  tokenizer ids   :", ids.tolist())
print("  词表大小        :", len(tok))

print("\n[verdict]", "PASS" if maxd == 0.0 and not (set(sd) - set(norm)) - {
    "pooler.dense.weight", "pooler.dense.bias"
} and d > 0 else "FAIL")
