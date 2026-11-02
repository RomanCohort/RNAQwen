"""把 pubmedbert 目录补全为 HF 标准文件集。

背景
----
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` 上游（HF 和 ModelScope
的文件清单完全一致）本身就**没有** model.safetensors，也**没有** tokenizer.json：

    GET https://huggingface.co/api/models/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
    -> .gitattributes LICENSE.md README.md config.json flax_model.msgpack
       pytorch_model.bin tokenizer_config.json vocab.txt

所以这两个文件只能本地生成：
    model.safetensors  <- 用 pytorch_model.bin 加载后由 transformers 导出（张量值一致）
    tokenizer.json     <- 用 vocab.txt + tokenizer_config.json 构造成 BertTokenizerFast 后导出

注意 transformers 5.9.0 的 `save_pretrained` 已经写死导 safetensors
(`SAFE_WEIGHTS_NAME`)，没有 `safe_serialization` 参数。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dtensor_shim  # noqa: E402  (torch 缺 _distributed_c10d，必须先补)

import torch  # noqa: F401,E402  (torch 必须先导入，safetensors 才拿得到后端)
from transformers import AutoModel, BertTokenizerFast  # noqa: E402


def main() -> int:
    d = Path(sys.argv[1]).resolve()
    print(f"[fix] target = {d}", flush=True)

    # ---- 1. tokenizer.json -------------------------------------------------
    tj = d / "tokenizer.json"
    if tj.exists():
        print(f"[fix] tokenizer.json already present ({tj.stat().st_size} bytes)", flush=True)
    else:
        tok = BertTokenizerFast.from_pretrained(str(d))
        tok.save_pretrained(str(d))
        if not tj.exists():
            raise RuntimeError("save_pretrained did not emit tokenizer.json")
        print(f"[fix] wrote tokenizer.json    {tj.stat().st_size} bytes", flush=True)

    # ---- 2. model.safetensors ---------------------------------------------
    st = d / "model.safetensors"
    if st.exists():
        print(f"[fix] model.safetensors already present ({st.stat().st_size} bytes)", flush=True)
    else:
        model = AutoModel.from_pretrained(str(d))
        model.save_pretrained(str(d))
        if not st.exists():
            raise RuntimeError("save_pretrained did not emit model.safetensors")
        print(f"[fix] wrote model.safetensors {st.stat().st_size} bytes", flush=True)

    print("[fix] dir now contains:", flush=True)
    for f in sorted(d.rglob("*")):
        if f.is_file():
            print(f"   {f.stat().st_size:>12d}  {f.relative_to(d).as_posix()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
