import json
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
W = Path(r"C:\qwen-api\RNAQwen\work\gguf")
man = json.loads((W / "manifest.json").read_text(encoding="utf-8"))
p = W / "patch.safetensors"
d = load_file(str(p))
k = "lm_head.weight"
n = man["base_vocab"]
before = d[k][n:].abs().max().item()
d[k][n:] = 0.0
after = d[k][n:].abs().max().item()
save_file(d, str(p), metadata={"format": "pt", "producer": "scripts/98_export_gguf.py"})
man["sentinel"] = 0.0
man["sentinel_note"] = ("constant rows are not safe: logit(c) = c*sum(h), so -100 "
                        "became a huge positive logit when sum(h)<0 (measured in the "
                        "smoke test). zeros -> logit 0.")
(W / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"lm_head rows >= {n}: max|v| {before:.3f} -> {after:.3f}; manifest sentinel {man['sentinel']}")
