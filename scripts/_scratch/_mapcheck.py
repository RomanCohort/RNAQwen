import sys
from pathlib import Path
LLAMA = Path(r"C:\qwen-api\llama-cpp-src\vendor\llama.cpp")
sys.path.insert(0, str(LLAMA)); sys.path.insert(1, str(LLAMA / "gguf-py"))
import gguf
tm = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35MOE, 41)
for n in ["model.layers.3.self_attn.q_proj.weight",
          "model.layers.0.linear_attn.in_proj_qkv.weight",
          "model.layers.0.linear_attn.out_proj.weight",
          "model.embed_tokens.weight", "lm_head.weight",
          "model.layers.20.mlp.shared_expert.down_proj.weight"]:
    try:
        print(n, "->", tm.get_name(key=n, try_suffixes=(".weight", ".bias")))
    except Exception as e:
        print(n, "!!", type(e).__name__, e)
r = gguf.GGUFReader(r"C:\qwen-api\models\Qwen3.6-35B-A3B-msc\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
print("fields has tokens:", "tokenizer.ggml.tokens" in r.fields)
print("n tensors", len(r.tensors), "n tokens", len(r.fields["tokenizer.ggml.tokens"].data))
print("arch", r.fields["general.architecture"].contents())
names = [t.name for t in r.tensors]
print([n for n in names if "blk.3." in n and "attn" in n][:8])
print([n for n in names if "embd" in n or "output" in n or n.startswith("blk.0.token")][:8])
