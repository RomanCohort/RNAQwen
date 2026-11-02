"""确认 ernierna / rnaernie 的类名、forward 签名、以及能不能像 rnafm 那样直接调。"""
import sys, inspect
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401
import torch

for modname in ["multimolecule.models.rnafm", "multimolecule.models.ernierna",
                "multimolecule.models.rnaernie"]:
    print("=" * 74)
    print(modname)
    print("=" * 74)
    try:
        import importlib
        m = importlib.import_module(modname)
    except Exception as e:
        print("  导入失败:", type(e).__name__, str(e)[:120])
        continue

    names = [n for n in dir(m) if n.endswith("Model") or n == "RnaTokenizer"]
    print("  相关类:", names)

    for cn in ["RnaFmModel", "ErnieRnaModel", "RnaErnieModel", "RnaTokenizer"]:
        if hasattr(m, cn):
            obj = getattr(m, cn)
            try:
                sig = inspect.signature(obj.forward)
                params = list(sig.parameters)
                print(f"  {cn}.forward 参数: {params}")
            except Exception as e:
                print(f"  {cn}: 取签名失败 {e}")
            doc = (obj.__doc__ or "").strip().split("\n")[0][:100]
            print(f"      doc: {doc}")
    print()

print("=" * 74)
print("ErnieRnaConfig 的默认值（判断要不要额外输入）")
print("=" * 74)
from multimolecule.models.ernierna import ErnieRnaConfig
c = ErnieRnaConfig()
for k in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "intermediate_size", "max_position_embeddings", "output_attention_biases",
          "pairwise_alpha", "position_embedding_type", "mask_token_id", "pad_token_id"]:
    print("   %-28s %r" % (k, getattr(c, k, "<无>")))
