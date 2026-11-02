import sys
from pathlib import Path
LLAMA = Path(r"C:\qwen-api\llama-cpp-src\vendor\llama.cpp")
sys.path.insert(0, str(LLAMA)); sys.path.insert(1, str(LLAMA / "gguf-py"))
import gguf
for p in [r"C:\qwen-api\models\Qwen3.6-35B-A3B-msc\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
          r"C:\qwen-api\RNAQwen\gguf\RNAQwen-preview-Q4_K_M.gguf"]:
    r = gguf.GGUFReader(p)
    names = [t.name for t in r.tensors]
    b40 = [n for n in names if n.startswith("blk.40.")]
    print(Path(p).name, len(names), "blk40:", len(b40))
    print("   ", b40[:14])
    for k in ("qwen35moe.nextn_predict_layers", "qwen35moe.block_count", "general.name"):
        f = r.fields.get(k)
        print(f"    {k} = {f.contents() if f is not None else None}")
