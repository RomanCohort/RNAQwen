"""弄清楚 RNA-FM 的前向到底走不走 word_embeddings。
   如果走且全零 -> 模型对序列完全不敏感（灾难）。
   如果它其实走别的路径 -> 我的敏感性测试写错了。
"""
import sys, inspect
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
import torch
from multimolecule.models.rnafm import RnaFmModel, RnaTokenizer

MP = str(ROOT / "models" / "rnafm")
m = RnaFmModel.from_pretrained(MP).eval()
tok = RnaTokenizer.from_pretrained(MP)

print("=== config 里的关键项 ===")
c = m.config
for k in ["vocab_size", "hidden_size", "num_hidden_layers", "max_position_embeddings",
          "token_dropout", "architectures", "model_type"]:
    print("  %-24s %r" % (k, getattr(c, k, "<无>")))

print("\n=== embeddings 模块结构 ===")
print("  type:", type(m.embeddings).__name__)
for n, mod in m.embeddings.named_children():
    print("    %-22s %s" % (n, type(mod).__name__))
print("  自己的张量:")
for n, p in m.embeddings.named_parameters(recurse=False):
    print("    %-22s %s  std=%.6e  nnz=%d/%d" % (
        n, tuple(p.shape), float(p.float().std()),
        int((p.float().abs() > 1e-8).sum()), p.numel()))

print("\n=== 直接跑 embeddings(ids) 看输出是否依赖 ids ===")
ids_a = torch.tensor([[1, 6, 7, 8, 9, 2]])
ids_b = torch.tensor([[1, 9, 9, 9, 9, 2]])
with torch.no_grad():
    ea = m.embeddings(ids_a)
    eb = m.embeddings(ids_b)
ea = ea[0] if isinstance(ea, tuple) else ea
eb = eb[0] if isinstance(eb, tuple) else eb
print("  embeddings(A) vs embeddings(U) max|Δ| =", f"{float((ea-eb).abs().max()):.6e}")

print("\n=== 完整前向：把 tokenizer 真实的 ids 打出来再比 ===")
seqs = ["ACGUACGUACGUACGUACGU", "GGGGGGGGGGGGGGGGGGGG", "UUUUUUUUUUUUUUUUUUUU"]
enc = tok(seqs, return_tensors="pt", padding=True)
print("  ids:")
for s, row in zip(seqs, enc["input_ids"].tolist()):
    print("    %-22s %s" % (s, row))
print("  enc keys:", list(enc.keys()))
with torch.no_grad():
    o = m(**enc).last_hidden_state
print("  输出形状:", tuple(o.shape))
print("  ACGUpoly vs Gpoly   max|Δ| =", f"{float((o[0]-o[1]).abs().max()):.6e}")
print("  ACGUpoly vs Upoly   max|Δ| =", f"{float((o[0]-o[2]).abs().max()):.6e}")
print("  逐 token 逐维都比较，不是只看 max —— 非零元素占比:",
      "%.4f%%" % (100.0*float((o[0]-o[1]).abs().gt(0).float().mean())))

print("\n=== 前向里有没有别的 token 嵌入来源 ===")
src = inspect.getsource(type(m.embeddings).forward)
print(src)
