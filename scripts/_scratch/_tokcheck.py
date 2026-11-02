import sys, traceback
from pathlib import Path
LLAMA = Path(r"C:\qwen-api\llama-cpp-src\vendor\llama.cpp")
print("--- plain import ---")
try:
    from transformers import AutoTokenizer
    print("plain OK", AutoTokenizer)
except Exception as e:
    traceback.print_exc()
print("--- with LLAMA on path ---")
sys.path.insert(0, str(LLAMA)); sys.path.insert(1, str(LLAMA / "gguf-py"))
try:
    from transformers import AutoTokenizer as AT2
    print("with-llama-path OK")
except Exception as e:
    traceback.print_exc()
print("--- direct module ---")
try:
    import transformers.models.auto.tokenization_auto as ta
    print("direct OK", ta.AutoTokenizer)
except Exception:
    traceback.print_exc()
import transformers, tokenizers
print("transformers", transformers.__version__, "tokenizers", tokenizers.__version__)
