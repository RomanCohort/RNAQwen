"""按统一约定的验证命令加载本地模型并跑一次前向。

用法:
    python scripts/verify_models.py <本地目录> [<本地目录> ...]

注意：本机 torch（ROCm Windows nightly）没编入分布式后端，
`from transformers import ...` 会直接 ImportError ——
所以这里先 import 本项目的 shim（详见 `src/rnaqwen/env/shim.py`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— 自动装载 env.shim

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


def verify(path: str) -> int:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).eval()
    with torch.no_grad():
        out = model(**tok("the RNA regulates gene expression.", return_tensors="pt"))
    print(f"{path}  hidden={model.config.hidden_size}  shape={tuple(out.last_hidden_state.shape)}")
    return 0


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 2
    rc = 0
    for p in paths:
        try:
            rc |= verify(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{p}  FAIL {type(exc).__name__}: {str(exc)[:200]}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
