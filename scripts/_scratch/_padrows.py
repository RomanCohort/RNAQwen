import json
import numpy as np
import torch
from safetensors import safe_open
from pathlib import Path
BASE = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
idx = json.loads((BASE / "model.safetensors.index.json").read_text(encoding="utf-8"))
for key in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
    shard = idx["weight_map"][key]
    with safe_open(str(BASE / shard), framework="pt", device="cpu") as f:
        t = f.get_tensor(key).float()
    pad = t[248070:248320]
    print(key, tuple(t.shape), "rows 248070..248319: max|v|", float(pad.abs().max()),
          "mean|v|", float(pad.abs().mean()))
    print("   row 248319 =", float(t[248319].abs().max()), " first 3 elems", t[248319][:3].tolist())
    print("   typical row |v| mean:", float(t[:1000].abs().mean()))
